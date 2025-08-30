import os
import sys

import tiktoken
import torch
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

from utils import get_project_root

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from args import get_command_line_args
from data_utils.load import load_data


API_KEY = "your api key"
API_BASE = "base-url"
client = OpenAI(api_key=API_KEY, base_url=API_BASE)
encoder = tiktoken.get_encoding("cl100k_base")


@retry(wait=wait_exponential(multiplier=1.5), stop=stop_after_attempt(5))
def get_batch_embeddings(batch):
    response = client.embeddings.create(
        input=batch,
        model="text-embedding-3-large"
    )
    return [item.embedding for item in response.data]


def get_embeddings_resume(texts,file, max_tokens_per_batch=8192, batch_size=32):

    if os.path.exists(file):
        existing = torch.load(file)
        start_idx = len(existing)
        all_embeddings = existing
        print(f"Detected {start_idx} saved embeddings. Continue processing from the {start_idx}th entry.")
    else:
        start_idx = 0
        all_embeddings = []
        print("No existing embeddings file detected, starting from scratch.")

    embed_dim = None
    if all_embeddings:
        embed_dim = all_embeddings[0].shape[0]

    current_batch = []
    current_tokens = 0
    idx = start_idx

    while idx < len(texts):
        text = texts[idx].replace("\n", " ").strip()
        idx += 1

        token_count = len(encoder.encode(text)) if text else 0

        if not text or token_count > max_tokens_per_batch:
            if embed_dim is None and all_embeddings:
                embed_dim = all_embeddings[0].shape[0]
            zeros = torch.zeros(embed_dim) if embed_dim else None
            all_embeddings.append(zeros)
            print(f'The {idx-1}th text {"empty" if not text else f"overlimit{token_count}" } is appended with a zero vector.')
            continue


        if current_tokens + token_count > max_tokens_per_batch or len(current_batch) >= batch_size:
            embs = get_batch_embeddings(current_batch)

            if embed_dim is None and embs:
                embed_dim = len(embs[0])
            all_embeddings.extend([torch.tensor(e) for e in embs])
            torch.save(all_embeddings, file)
            print(f"{len(all_embeddings)} embeddings have been saved (up to idx={idx-1}).")
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += token_count

    if current_batch:
        embs = get_batch_embeddings(current_batch)
        if embed_dim is None and embs:
            embed_dim = len(embs[0])
        all_embeddings.extend([torch.tensor(e) for e in embs])
        torch.save(all_embeddings, file)
        print(f"All done, a total of {len(all_embeddings)} embeddings saved.")

    return torch.stack(all_embeddings) if embed_dim else torch.tensor([])


if __name__ == "__main__":
    args = get_command_line_args()
    root = get_project_root()
    if args.text_attack is not None:
        data = torch.load(f"{root}/attack/data/orig/{args.dataset}_{args.text_attack}_orig.pt")
        texts = [item["content"] for item in data]
        file = f"{root}/data/emb/ChatGPT/openai_{args.dataset}_{args.text_attack}_embeddings.pt"
    else:
        data,_,texts = load_data(args.dataset,use_text=True,seed=0)
        file = f"{root}/data/emb/ChatGPT/openai_{args.dataset}_clean_embeddings.pt"
    embeddings = get_embeddings_resume(texts,
                                       file=file,
                                       max_tokens_per_batch=8192,
                                       batch_size=32)

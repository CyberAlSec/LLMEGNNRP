import argparse
import os
import gc
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from typing import List, Optional

from utils import get_project_root, get_model_name

# You can set a default model path here or override it at runtime with --model_path
root = get_project_root()
# -------------------- Helper functions --------------------
def load_embedding_model(model_path: str, device: torch.device, use_device_map: bool = True, dtype=torch.float16):
    """
    Load tokenizer and model. For large models, device_map is used to auto-place model shards on available GPUs.
    """
    print(f"[info] Loading tokenizer and model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # try to load with device_map for big models, fallback to moving model to device
    try:
        if use_device_map and torch.cuda.is_available():
            model = AutoModel.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModel.from_pretrained(model_path)
            model.to(device)
    except Exception as e:
        print("[warn] device_map load failed, falling back to simple load:", e)
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
    model.eval()
    return tokenizer, model

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Last-token pooling: pick the last non-pad token embedding for each sequence.
    last_hidden_states: [batch, seq_len, hidden]
    attention_mask: [batch, seq_len]
    Returns: [batch, hidden]
    """
    seq_lens = attention_mask.sum(dim=1) - 1  # index of last non-pad token
    batch_idx = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
    return last_hidden_states[batch_idx, seq_lens, :]

def mean_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Mean-pool across non-pad tokens.
    """
    mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
    masked = last_hidden_states * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts

def text_to_embeddings_batch(texts: List[str],
                             tokenizer,
                             model,
                             device: torch.device,
                             max_length: int = 2048,
                             batch_size: int = 1,
                             pool: str = "mean") -> torch.Tensor:
    """
    Convert a list of texts to embedding tensors (returned on CPU).
    pool: "mean" | "last_token"
    """
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch = tokenizer(batch_texts,
                          padding=True,
                          truncation=True,
                          max_length=max_length,
                          return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            # prefer last_hidden_state if present
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
            else:
                # fallback: use final hidden state from hidden_states
                hidden_states = outputs.hidden_states
                last_hidden = hidden_states[-1]

            attention_mask = batch.get("attention_mask", None)
            if attention_mask is None:
                # all tokens are non-pad
                attention_mask = torch.ones(last_hidden.size()[:2], dtype=torch.long, device=last_hidden.device)

            if pool == "last_token":
                emb = last_token_pool(last_hidden, attention_mask)
            else:
                emb = mean_pool(last_hidden, attention_mask)

            emb = emb.cpu()
            all_embs.append(emb)

        # free
        del outputs, last_hidden, batch
        torch.cuda.empty_cache()
        gc.collect()

    if len(all_embs) == 0:
        return torch.empty((0, 0))
    embeddings = torch.cat(all_embs, dim=0)
    return embeddings

# -------------------- Main processing flow --------------------
def process_dataset(model_type: str,
                    model_path: str,
                    dataset_name: str,
                    style: Optional[str],
                    seed: int,
                    batch_size: int,
                    max_length: int):
    """
    Load texts (from an attack file or load_data), generate embeddings in batches, and save the result.
    This version processes ALL texts (no start slicing).
    """
    # delay import of load_data to avoid heavy imports at module load
    from data_utils.load import load_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_embedding_model(model_path, device)

    # prepare texts:
    texts = None
    # try reading attack file if style provided
    if style is not None:
        default_attack_path = os.path.join(f"{root}/attack/data/orig", dataset_name,
                                           f"{dataset_name}_{style}_orig.pt")
        if os.path.exists(default_attack_path):
            try:
                adv = torch.load(default_attack_path)
                # expect list of dicts with "content" key
                try:
                    texts = [item["content"] for item in adv]
                except Exception:
                    # if structure is different, try to convert items to strings directly
                    texts = [str(x) for x in adv]
                print(f"[info] Loaded {len(texts)} texts from attack file: {default_attack_path}")
            except Exception as e:
                print("[warn] Unable to load attack file, reason:", e)

    # fallback: use load_data
    if texts is None:
        _, _, texts = load_data(dataset=dataset_name, use_text=True, attack_type=None, seed=seed)
        print(f"[info] Loaded {len(texts)} texts from load_data() for dataset {dataset_name}")

    # NOTE: no slicing by start anymore — process ALL texts
    total_samples = len(texts)

    # ensure output_dir exists
    output_dir = f"{root}/data/emb/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_type}/{model_type}_{dataset_name}_{style or 'clean'}_emb.pt")

    # process in chunks to avoid memory spikes
    chunk_size = 1000  # number of samples to accumulate before saving; adjust if needed
    embeddings_parts = []
    for i in range(0, total_samples, chunk_size):
        chunk_texts = texts[i:i+chunk_size]
        print(f"[info] Processing samples {i}..{i+len(chunk_texts)-1} (chunk_size={len(chunk_texts)})")
        embs = text_to_embeddings_batch(chunk_texts, tokenizer, model, device,
                                        max_length=max_length, batch_size=batch_size)
        if embs.numel() == 0:
            continue
        embeddings_parts.append(embs)
        # free GPU
        del embs
        torch.cuda.empty_cache()
        gc.collect()

    if len(embeddings_parts) == 0:
        print("[warn] No embeddings were generated, exiting.")
        return

    embeddings = torch.cat(embeddings_parts, dim=0)
    torch.save(embeddings, output_file)
    print(f"[done] Saved embeddings to {output_file}, shape: {embeddings.shape}")

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="Llama", choices=["Linq","Llama"], help="Pooling/model type hint ('Linq' uses last-token pooling; others use mean)")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name (passed to load_data as dataset argument)")
    parser.add_argument("--style", type=str, default=None, help="Attack style name (optional). If a matching attack file exists it will be used.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Tokenizer batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Tokenizer max_length")
    parser.add_argument("--batch_chunk", type=int, default=1000, help="Chunk size for batching (affects memory)")
    args = parser.parse_args()
    model_path,_ = get_model_name(args.model_type)
    process_dataset(model_type = args.model_type,
                    model_path=model_path,
                    dataset_name=args.dataset,
                    style=args.style,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    max_length=args.max_length)

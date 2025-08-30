import asyncio
import json
import os
import torch
import tiktoken
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import time
import numpy as np

from args import get_command_line_args
from data_utils.load import load_data
from utils import generate_pred, get_project_root
root = get_project_root()
API_KEY = "your-apikey"
API_BASE = "api-url"

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)
inst = {
    "cora":" The extracted terms should be relevant to artificial intelligence, machine learning",
    "pubmed":"The extracted terms should be relevant to biomedicine, life sciences, and clinical medicine.",
    "arxiv_2023":"The extracted terms should be relevant to computer science, physics, mathematics, and engineering disciplines.",
    "ogbn-products":"The extracted terms should be relevant to product categories, e-commerce attributes, and consumer behavior."
}


async def gpt_entity_extraction_batch(args,  batch_size=5, max_retries=3):
    style = args.feature_type
    dataset_name = args.dataset
    instr = inst[dataset_name]
    if args.text_attack is not None:
        data = torch.load(f"{root}/attack/data/orig/{dataset_name}/{dataset_name}_{args.text_attack}_orig.pt")
        texts = [item["content"] for item in data]
    else:
        data,_,texts = load_data(dataset_name,use_text=True,seed=0)

    prompts = build_prompts(dataset_name, texts, instruction=instr)


    batches = create_safe_batches(prompts, model="gpt-3.5-turbo", batch_size=batch_size)

    results = [None] * len(prompts)
    total_batches = len(batches)

    pbar = tqdm(total=len(prompts), desc=f"Processing progress ({dataset_name})")

    # Processing each batch
    for i, batch in enumerate(batches):
        # Get the index of the current batch in the original list
        start_idx = sum(len(b) for b in batches[:i])
        end_idx = start_idx + len(batch)

        # Creation of tasks for the current batch
        tasks = []
        for j, prompt in enumerate(batch):
            idx = start_idx + j
            tasks.append(process_single_prompt(client, prompt, idx, max_retries))

        # Concurrent execution of tasks
        batch_results = await asyncio.gather(*tasks)

        #  update result
        for idx, content in batch_results:
            results[idx] = content
            pbar.update(1)

    pbar.close()

    save_final_results(dataset_name, results, args.text_attack,style)

    print(f"Entity extraction complete! A total of {len(prompts)} text was processed.")
    return results


async def process_single_prompt(client, prompt, idx, max_retries=3):
    """
    Handling of individual prompts, including retry mechanism
    :return: (original index, response content)
    """
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return (idx, resp.choices[0].message.content)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                tqdm.write(f"Error {idx}: {str(e)}, wait {wait_time}s and retry...")
                await asyncio.sleep(wait_time)
            else:
                tqdm.write(f"Error {idx}: {str(e)}")
                return (idx, None)


def create_safe_batches(prompts, model="gpt-3.5-turbo", batch_size=10, max_tokens=4096, safety_margin=500):
    """
    Create secure batch processing lists (for grouping only, actual requests are single)
    """
    try:
        encoder = tiktoken.encoding_for_model(model)
    except:
        encoder = tiktoken.get_encoding("cl100k_base")

    token_counts = [len(encoder.encode(p)) for p in prompts]

    max_per_prompt = max_tokens - safety_margin

    valid_batches = []
    current_batch = []
    current_tokens = 0

    for idx, (prompt, count) in enumerate(zip(prompts, token_counts)):
        # Check if the current prompt is too long
        if count > max_per_prompt:
            print(f"WARNING: Skipping long prompts #{idx} (tokens={count}>{max_per_prompt})")
            continue

        if len(current_batch) >= batch_size or current_tokens + count > max_tokens:
            valid_batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(prompt)
        current_tokens += count

    if current_batch:
        valid_batches.append(current_batch)

    print(f"Create {len(valid_batches)} batches (up to {batch_size} hints per group)")
    return valid_batches


def save_final_results(dataset_name, results,  attack_name, style):
    if attack_name is not None:
        torch_path = f"{root}/attack/data/{style}/{dataset_name}/{dataset_name}_{attack_name}.pt"
    else:
        torch_path = f"{root}/attack/data/{style}/{dataset_name}/{dataset_name}_clean.pt"
    torch.save(results, torch_path)
    if style == "TAPE":
        generate_pred(results,dataset_name,attack_name)
    print(f"Final results saved: {torch_path}")

def build_prompts(dataset_name,texts, style="KEA", instruction = "The extracted terms should be relevant to artificial intelligence, machine learning"):
    prompts = []
    for t in texts:
        #KEA
        if style == "KEA":
            prompt = f"You should work like a named entity recognizer. \n Paper: \n {t} \n Extract the technical terms from this paper and output a description for each terms in the format of a python dict, with the format {{'XX': 'XXX', 'YY': 'YYY'}}. {instruction} \n "
        #TAPE
        elif style == "TAPE":
            if dataset_name == "cora":
                prompt = f"{t} \n Question：Which of the following sub-categories of AI does this paper belong to: Case Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text. \n \n Answer:"
            elif dataset_name == "pubmed":
                prompt = f"{t} \n Question: Does the paper involve any cases of Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes? Please give one or more answers of either Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes; if multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, give a detailed explanation with quotes from the text explaining why it is related to the chosen option. \n \n Answer:"
            elif dataset_name == "ogbn-products":
                prompt = f"Product description：{t} \n Question: Which of the following category does this product belong to: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn &Garden, Toys&Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies &TV,15) Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, NAN, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Kindle Apps, Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning. \n \n Answer:"
            else:
                prompt = f"{t} \n Question: Which arXiv CS subcategory does this paper belong to? Give 5 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely, in the form “cs.XX”, and provide your reasoning. \n \n Answer:"
        prompts.append(prompt)
    return prompts


if __name__ == "__main__":
    args = get_command_line_args()
    asyncio.run(gpt_entity_extraction_batch(args,  batch_size=20 ))
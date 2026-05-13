import torch
from datasets import load_dataset


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    elif data == "dummy":
        dummy_texts = [
            "Represent this sentence for retrieval: activation aware quantization compresses transformer weights.",
            "Qwen embedding models convert passages and queries into dense vectors for semantic search.",
            "This calibration text is synthetic and local so the notebook does not download a dataset.",
            "INT4 weight-only quantization stores grouped linear weights with scales and zero points.",
        ]
        samples = []
        for i in range(n_samples):
            text = dummy_texts[i % len(dummy_texts)]
            encoded = tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=block_size,
                truncation=True,
            )
            if len(encoded) < block_size:
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = tokenizer.eos_token_id
                if pad_token_id is None:
                    pad_token_id = 0
                encoded = encoded + [pad_token_id] * (block_size - len(encoded))
            samples.append(torch.tensor([encoded[:block_size]], dtype=torch.long))
        return samples
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]

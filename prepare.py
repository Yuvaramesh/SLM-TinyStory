import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Number of workers in parallel operations
num_proc = 8
num_proc_load_dataset = num_proc

# Get GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

# Output directory (optional: adjust as needed)
output_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()

if __name__ == '__main__':
    print("🔹 Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc_load_dataset)

    # Split into train/val
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    # Tokenization function
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    print("🔹 Tokenizing...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Save binary files
    for split, dset in tokenized.items():
        print(f"\n🔹 Saving {split}.bin...")
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"✅ Saved {filename} with {arr_len} tokens.")

        # Sanity check: decode a portion to verify English text
        preview_tokens = np.memmap(filename, dtype=dtype, mode='r')[:200]
        preview_text = enc.decode(preview_tokens.tolist())
        print(f"\n🧪 Decoded preview from {split}.bin:")
        print("--------------------------------------------------")
        print(preview_text)
        print("--------------------------------------------------")

    print("\n✅ All splits processed and saved successfully.")

import os
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from datasets import Dataset, Audio
import logging
import requests
import torch
from transformers import pipeline
import torch
from datasets import Dataset, Audio
from torch.utils.data import DataLoader

# Define device (GPU if available, otherwise CPU)
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Initialize generator pipeline (only one model now)
generator = pipeline(task="automatic-speech-recognition", model="wadhwani-ai/tibet-stt-wav2vec2-mms300m", device=device, token="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")

# Function to download and validate audio
def download_audio(row):
    file_name = os.path.basename(row["url"])
    save_path = f"../data/volume/wav_16k/{file_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):  # Skip if file exists
        return save_path

    try:
        response = requests.get(row["url"], timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path

    except Exception as e:
        logging.error(f"Failed to download {row['file_name']}: {e}")
        return None  # Return None if download failed

# Function to process inference in batches
def process_inference_batch(batch):
    # Perform inference with the selected generator (only one model now)
    results = generator(batch["path"])
    return results

if __name__ == "__main__":
    chunk_size = 1000
    input_file = "../benchmark_v3.csv"
    output_dir = "chunks_v3"
    output_file = "benchmark_v3_inference.csv"
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size), start=1):
        if chunk.empty:
            print(f"Chunk {i} is empty. Skipping.")
            continue

        chunk_file = os.path.join(output_dir, f"chunk_{i}.csv")
        if os.path.exists(chunk_file):
            print(f"Skipping chunk {i}, already processed.")
            continue

        tqdm.pandas(desc="Downloading audio files")
        # Add required columns
        chunk['Path'] = './data/volume/wav_16k/' + chunk['file_name'] + ".wav"
        chunk['url'] = 'https://d38pmlk0v88drf.cloudfront.net/wav16k/' + chunk['file_name'] + ".wav"
        chunk['inf'] = ""
        chunk["path"] = chunk.progress_apply(download_audio, axis=1)
        chunk = chunk[chunk["path"].notnull()]  # Remove rows with failed downloads

        # Reset index to avoid duplicate field errors
        chunk.reset_index(drop=True, inplace=True)

        # Convert to Dataset
        dataset = Dataset.from_pandas(chunk)

        # Create DataLoader for batching
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        # Perform batched inference
        results = []
        for batch in tqdm(dataloader, desc="Running inference"):
            batch_results = process_inference_batch(batch)
            results.extend(batch_results)

        # Add inference results to dataset
        chunk['inf'] = results

        # Save processed chunk to CSV
        chunk.to_csv(chunk_file, index=False)
        print(f"Saved chunk {i} to {chunk_file}")

    # Merge all chunk files into final output
    all_chunks = [pd.read_csv(os.path.join(output_dir, f)) for f in sorted(os.listdir(output_dir)) if f.endswith(".csv")]
    final_df = pd.concat(all_chunks, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"All chunks merged and saved to {output_file}")

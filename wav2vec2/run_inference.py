import os
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from datasets import Dataset, Audio
import logging
import requests
import torch

# Configure logging
logging.basicConfig(filename='failed_downloads.log', level=logging.ERROR, format='%(asctime)s - %(message)s')

# Define device (GPU if available, otherwise CPU)
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Initialize generator pipeline (only one model now)
generator = pipeline(task="automatic-speech-recognition", model="ganga4364/mms_300_khentse_Rinpoche-Checkpoint-58000", device=device)

# Function to download and validate audio
def download_audio(row):
    file_name = os.path.basename(row["url"])
    save_path = f"./data/volume/wav_16k/{file_name}"
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
def process_inference(batch):
    # Perform inference with the selected generator (only one model now)
    results = generator(batch["audio"]["array"])
    batch["inf"] = [result["text"] for result in results]  # Store inference results in the 'inf' column

    return batch

if __name__ == "__main__":
     # Load dataset in chunks
    chunk_size = 1000
    input_file = "../benchmark_v1.csv"
    output_dir = "chunks"
    output_file = "benchmark_v1_inference.csv"
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=1000), start=1):
        if chunk.empty:
            print(f"Chunk {i} is empty. Skipping.")
            continue

        chunk_file = os.path.join(output_dir, f"chunk_{i}.csv")
        if os.path.exists(chunk_file):
            print(f"Skipping chunk {i}, already processed.")
            continue

        tqdm.pandas(desc="Downloading audio files")
        chunk["path"] = chunk.progress_apply(download_audio, axis=1)
        chunk = chunk[chunk["path"].notnull()]  # Remove rows with failed downloads

        # Reset index to avoid duplicate field errors
        chunk.reset_index(drop=True, inplace=True)

        # Convert to Dataset
        dataset = Dataset.from_pandas(chunk)
        dataset = dataset.cast_column("path", Audio())

        # Perform batched inference
        dataset = dataset.map(process_inference, batched=True, batch_size=8)

        # Save processed chunk to CSV
        dataset.to_pandas().to_csv(chunk_file, index=False)
        print(f"Saved chunk {i} to {chunk_file}")

    # Merge all chunk files into final output
    all_chunks = [pd.read_csv(os.path.join(output_dir, f)) for f in sorted(os.listdir(output_dir)) if f.endswith(".csv")]
    final_df = pd.concat(all_chunks, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"All chunks merged and saved to {output_file}")

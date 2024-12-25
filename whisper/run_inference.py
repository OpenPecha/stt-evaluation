import os
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import pyewts
from tqdm import tqdm
import soundfile as sf
import requests
import logging

# Configure logging
logging.basicConfig(filename='failed_downloads.log', level=logging.ERROR, format='%(asctime)s - %(message)s')

# Initialize the Tibetan EWTs converter
converter = pyewts.pyewts()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and processor onto GPU
model = WhisperForConditionalGeneration.from_pretrained("ganga4364/whipser-small-reft").to(device)
processor = WhisperProcessor.from_pretrained("ganga4364/whipser-small-reft")

# Function to download and validate audio
def download_audio(url, file_name, save_path):
    if os.path.exists(save_path):  # Skip if file exists
        return

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        with open(save_path, 'wb') as f:
            f.write(response.content)

        # Validate audio file
        with sf.SoundFile(save_path) as file:
            pass  # If no exception, the file is valid
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {file_name}: {e}")
    except Exception as e:
        logging.error(f"Invalid audio file {file_name}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)  # Remove invalid file

# Function to run batch inference
def run_batch_inference(audio_paths, model, processor):
    inputs_list = []
    valid_paths = []
    for audio_path in audio_paths:
        try:
            audio_input, sr = sf.read(audio_path)
            if sr != 16000:
                logging.error(f"Invalid sampling rate for {audio_path}: expected 16kHz, got {sr}")
                continue
            inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")
            inputs_list.append(inputs.input_features)
            valid_paths.append(audio_path)
        except Exception as e:
            logging.error(f"Error processing audio {audio_path}: {e}")

    if not inputs_list:
        return {path: "" for path in audio_paths}

    # Stack inputs and move to GPU
    input_features = torch.cat(inputs_list).to(device)

    # Generate predictions
    generated_ids = model.generate(input_features)
    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Convert transcriptions to Unicode
    unicode_transcriptions = {path: converter.toUnicode(transcription)
                               for path, transcription in zip(valid_paths, transcriptions)}
    # Add empty results for invalid paths
    for path in audio_paths:
        if path not in unicode_transcriptions:
            unicode_transcriptions[path] = ""

    return unicode_transcriptions

# Function to process a batch of rows
def process_batch(batch_df):
    batch_size = 8  # Adjust batch size based on your GPU memory
    results = {}

    # Download audio files
    for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Downloading audio"):
        url = row['url']
        file_name = os.path.basename(url)
        save_path = f"/data/volume/wav_16k/{file_name}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        download_audio(url, file_name, save_path)

    # Perform inference in batches
    audio_paths = batch_df['Path'].tolist()
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Running inference"):
        batch_paths = audio_paths[i:i + batch_size]

        # Inference with the model
        batch_results = run_batch_inference(batch_paths, model, processor)
        results.update(batch_results)

    # Update DataFrame with inference results
    batch_df['inf'] = batch_df['Path'].map(results)

    return batch_df

if __name__ == "__main__":
    # Load dataset in chunks
    chunk_size = 1000
    input_file = "../benchmark_v1.csv"
    output_dir = "chunks"
    output_file = "benchmark_v1_inference.csv"

    os.makedirs(output_dir, exist_ok=True)

    # Process the dataset in chunks
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size), start=1):
        chunk_file = os.path.join(output_dir, f"chunk_{i}.csv")

        # Skip processing if the chunk file already exists
        if os.path.exists(chunk_file):
            print(f"Skipping chunk {i}, already processed.")
            continue

        # Add required columns
        chunk['Path'] = '/data/volume/wav_16k/' + chunk['file_name'] + ".wav"
        chunk['url'] = 'https://d38pmlk0v88drf.cloudfront.net/wav16k/' + chunk['file_name'] + ".wav"
        chunk['inf'] = ""

        # Process the current batch
        processed_chunk = process_batch(chunk)

        # Save the processed chunk
        processed_chunk.to_csv(chunk_file, index=False)
        print(f"Saved chunk {i} to {chunk_file}")

    # Merge all chunk files into the final output file
    all_chunks = [pd.read_csv(os.path.join(output_dir, f)) for f in sorted(os.listdir(output_dir)) if f.endswith(".csv")]
    final_df = pd.concat(all_chunks, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"All chunks merged and saved to {output_file}")

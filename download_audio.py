import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import requests
import os
import requests
import logging

# Configure logging
logging.basicConfig(filename='failed_downloads.log', level=logging.ERROR, format='%(asctime)s - %(message)s')

# Function to download audio file
def download_audio(url, file_name, save_path):
    # Full path where the audio will be saved
    
    # Skip download if file already exists
    if os.path.exists(save_path):
        return
    
    try:
        # Download the file from the URL and save it to the specified path
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        with open(save_path, 'wb') as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        # Log the error and file name if the download fails
        logging.error(f"Failed to download {file_name}: {e}")


# Function to download the audio file and perform inference
def process_row(row):
    url = row['url']
    file_name = os.path.basename(url)
    
    # Create the save path for the temporary audio file
    save_path = f"/data/volume/wav_16k/{file_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Function to download the audio file
    download_audio(url, file_name, save_path)
    
    return row

# Function to process the DataFrame using Pool
def process_dataframe(df):
    # Number of processes to run, set to the number of available CPU cores or adjust as needed
    num_workers = min(cpu_count(), 5)  # Adjust based on your system capabilities

    # Use Pool to process rows in parallel
    with Pool(processes=num_workers) as pool:
        rows = list(tqdm(pool.imap(process_row, [row for _, row in df.iterrows()]), total=len(df)))

    # Convert the processed rows back into a DataFrame
    return pd.DataFrame(rows)



if __name__ == "__main__":

    df = pd.read_csv("benchmark_v1.csv")
    # Add required columns
    df['Path'] = './data/volume/wav_16k/' + df['file_name'] + ".wav"
    df['url'] = 'https://d38pmlk0v88drf.cloudfront.net/wav16k/' + df['file_name'] + ".wav"

    processed_df = process_dataframe(df)
   

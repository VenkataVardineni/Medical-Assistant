#!/usr/bin/env python3
"""
Medical Data Preparation Script

This script fetches and prepares medical Q&A data from the MedQuad dataset
using the Hugging Face datasets API.
"""

import requests
import pandas as pd
import json
import time
from typing import List, Dict, Any
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedQuadDataFetcher:
    """A class for fetching and preparing MedQuad medical Q&A dataset."""
    
    def __init__(self, base_url: str = "https://datasets-server.huggingface.co/rows"):
        """
        Initialize the data fetcher.
        
        Args:
            base_url: Base URL for the Hugging Face datasets API
        """
        self.base_url = base_url
        self.dataset_name = "keivalya/MedQuad-MedicalQnADataset"
        self.config = "default"
        self.split = "train"
        
    def fetch_dataset_batch(self, offset: int = 0, length: int = 100, max_retries: int = 3) -> Dict[str, Any]:
        """
        Fetch a batch of data from the MedQuad dataset with retry logic.
        
        Args:
            offset: Starting index for the batch
            length: Number of records to fetch
            max_retries: Maximum number of retry attempts
            
        Returns:
            API response as dictionary
        """
        params = {
            'dataset': self.dataset_name,
            'config': self.config,
            'split': self.split,
            'offset': offset,
            'length': length
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching data batch: offset={offset}, length={length} (attempt {attempt + 1})")
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    logger.warning(f"Rate limited. Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"HTTP error fetching data batch: {e}")
                    raise
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching data batch: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        
        raise Exception(f"Failed to fetch data batch after {max_retries} attempts")
    
    def fetch_full_dataset(self, batch_size: int = 100, max_records: int = None) -> pd.DataFrame:
        """
        Fetch the full MedQuad dataset in batches.
        
        Args:
            batch_size: Number of records to fetch per batch
            max_records: Maximum number of records to fetch (None for all)
            
        Returns:
            DataFrame containing all fetched data
        """
        all_data = []
        offset = 0
        total_expected = 16407  # Known total from dataset info
        
        logger.info(f"Starting to fetch dataset. Expected total: {total_expected} records")
        
        # Save progress every 500 records
        save_interval = 500
        
        while True:
            # Fetch batch
            batch_response = self.fetch_dataset_batch(offset, batch_size)
            
            # Extract rows from response
            if 'rows' in batch_response:
                batch_data = batch_response['rows']
                if not batch_data:
                    logger.info("No more data available")
                    break
                
                # Extract the actual data from each row
                for row in batch_data:
                    if 'row' in row:
                        all_data.append(row['row'])
                
                # Calculate progress
                progress = (len(all_data) / total_expected) * 100
                logger.info(f"Fetched {len(batch_data)} records (total: {len(all_data)}/{total_expected}, {progress:.1f}%)")
                
                # Save progress periodically
                if len(all_data) % save_interval == 0:
                    temp_df = pd.DataFrame(all_data)
                    temp_filename = f"medquad_temp_{len(all_data)}.csv"
                    temp_df.to_csv(temp_filename, index=False)
                    logger.info(f"Saved progress checkpoint: {temp_filename}")
                
                # Check if we've reached the maximum records
                if max_records and len(all_data) >= max_records:
                    all_data = all_data[:max_records]
                    logger.info(f"Reached maximum records limit: {max_records}")
                    break
                
                # Move to next batch
                offset += batch_size
                
                # Add a much longer delay to be respectful to the API and avoid rate limiting
                time.sleep(3.0 + random.uniform(0, 2.0))  # 3-5 seconds between requests
            else:
                logger.warning("No 'rows' found in response")
                break
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            logger.info(f"Successfully created DataFrame with {len(df)} records and {len(df.columns)} columns")
            return df
        else:
            logger.warning("No data was fetched")
            return pd.DataFrame()
    
    def clean_medical_text(self, text: str) -> str:
        """
        Clean medical text data.
        
        Args:
            text: Raw medical text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep medical terms and punctuation
        import re
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}]', '', text)
        
        # Normalize spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        return text.strip()
    
    def preprocess_qa_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the Q&A data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        # Clean text columns - use the actual column names from the dataset
        text_columns = ['Question', 'Answer']
        for col in text_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].apply(self.clean_medical_text)
                logger.info(f"Cleaned text column: {col}")
        
        # Remove rows with empty questions or answers
        initial_count = len(df_processed)
        df_processed = df_processed.dropna(subset=['Question', 'Answer'])
        df_processed = df_processed[df_processed['Question'].str.strip() != '']
        df_processed = df_processed[df_processed['Answer'].str.strip() != '']
        
        final_count = len(df_processed)
        removed_count = initial_count - final_count
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with empty questions or answers")
        
        # Reset index
        df_processed = df_processed.reset_index(drop=True)
        
        return df_processed
    
    def save_data(self, df: pd.DataFrame, filename: str, format: str = 'json') -> str:
        """
        Save data to file.
        
        Args:
            df: DataFrame to save
            filename: Name of the file
            format: File format ('json', 'csv', or 'parquet')
            
        Returns:
            Path to saved file
        """
        if format == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                df.to_json(f, orient='records', lines=True, force_ascii=False)
        elif format == 'csv':
            df.to_csv(filename, index=False, encoding='utf-8')
        elif format == 'parquet':
            df.to_parquet(filename, index=False)
        else:
            raise ValueError("Unsupported format. Use 'json', 'csv', or 'parquet'")
        
        logger.info(f"Saved data to: {filename}")
        return filename
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dataset information
        """
        try:
            # Try to get dataset info
            info_url = f"https://datasets-server.huggingface.co/info?dataset={self.dataset_name}"
            response = requests.get(info_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not fetch dataset info: {e}")
            return {}


def main():
    """Main function to fetch and prepare MedQuad dataset."""
    
    # Initialize data fetcher
    fetcher = MedQuadDataFetcher()
    
    # Get dataset info
    logger.info("Getting dataset information...")
    dataset_info = fetcher.get_dataset_info()
    if dataset_info:
        logger.info(f"Dataset info: {json.dumps(dataset_info, indent=2)}")
    
    # Fetch a sample batch first to see the structure
    logger.info("Fetching sample batch to examine data structure...")
    sample_batch = fetcher.fetch_dataset_batch(offset=0, length=5)
    
    if 'rows' in sample_batch and sample_batch['rows']:
        sample_data = sample_batch['rows'][0]['row']
        logger.info(f"Sample data structure: {list(sample_data.keys())}")
        logger.info(f"Sample question: {sample_data.get('Question', 'N/A')[:100]}...")
        logger.info(f"Sample answer: {sample_data.get('Answer', 'N/A')[:100]}...")
    
    # Fetch the full dataset
    logger.info("Fetching full dataset...")
    df = fetcher.fetch_full_dataset(batch_size=50, max_records=None)  # Fetch all records with smaller batches
    
    if not df.empty:
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Preprocess the data
        logger.info("Preprocessing data...")
        df_processed = fetcher.preprocess_qa_data(df)
        
        # Save raw data
        fetcher.save_data(df, 'medquad_raw.json', 'json')
        
        # Save processed data
        fetcher.save_data(df_processed, 'medquad_processed.json', 'json')
        fetcher.save_data(df_processed, 'medquad_processed.csv', 'csv')
        
        # Display some statistics
        logger.info("Dataset Statistics:")
        logger.info(f"Total records: {len(df_processed)}")
        logger.info(f"Average question length: {df_processed['Question'].str.len().mean():.1f} characters")
        logger.info(f"Average answer length: {df_processed['Answer'].str.len().mean():.1f} characters")
        
        # Show some examples
        logger.info("\nSample Questions and Answers:")
        for i in range(min(3, len(df_processed))):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Q: {df_processed.iloc[i]['Question']}")
            logger.info(f"A: {df_processed.iloc[i]['Answer'][:200]}...")
        
        logger.info("\nData preparation completed successfully!")
    else:
        logger.error("Failed to fetch data from the API")


if __name__ == "__main__":
    main()
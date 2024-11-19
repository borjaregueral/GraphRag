import re
import tiktoken
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import unicodedata
import ftfy
import codecs
import polars as pl
from pathlib import Path
from tqdm import tqdm
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def clean_text(text: str) -> str:
    """
    Clean text by handling various encoding issues and escape sequences.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
        
    Raises:
        ValueError: If input is not a string
        UnicodeError: If text cannot be properly decoded
    """
    # Input validation
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    if not text:
        return ""
    
    try:
        # 1. Fix encoding issues using ftfy
        text = ftfy.fix_text(text)
        
        # 2. Remove specific disclaimer patterns
        text = re.sub(r"Transcript:\s+\(disclaimer:.*?\)", "", text, flags=re.IGNORECASE)
        
        # 3. Remove speaker labels (e.g., "Ben:", "David:")
        text = re.sub(r'\b[A-Z][a-z]*:\s', '', text)
        
        # 4. Unescape any remaining escaped characters
        # This will convert sequences like \' to ', \" to ", etc.
        text = codecs.decode(text, 'unicode_escape', errors='ignore')
        
        # 5. Remove any remaining problematic characters
        text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"').replace('Â', '')
        
        # 6. Normalize Unicode characters to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # 7. Remove non-breaking spaces and other non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # 8. Normalize whitespace by replacing multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    except UnicodeError as e:
        logger.error(f"Failed to process text encoding: {str(e)}")
        raise UnicodeError(f"Failed to process text encoding: {str(e)}")
    except re.error as e:
        logger.error(f"Invalid regex pattern: {str(e)}")
        raise ValueError(f"Invalid regex pattern: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise e

def num_tokens_from_string(string: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def read_text_file(text_folder: Path, file_name: str) -> str:
    """Read text from a file."""
    file_path = text_folder / f"{file_name}.txt"  # Append .txt extension
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"An error occurred while reading {file_path}: {str(e)}")
        return ""

def process_podcast(podcast: dict, text_folder: Path) -> dict:
    """Process a single podcast entry."""
    if podcast.get("has_transcript"):
        text = read_text_file(text_folder, podcast["file_name"])
        cleaned_text = clean_text(text)
        tokens = num_tokens_from_string(f"{podcast['post_title']} {podcast['blog_title']} {cleaned_text}")
        return {
            **podcast,
            "text": text, 
            "cleaned_text": cleaned_text,
            "tokens": tokens
        }
    return None

def load_and_process_podcasts(csv_path: str, text_folder: Path) -> pl.DataFrame:
    """Load and process podcasts from a CSV file."""
    try:
        # Load the CSV file into a Polars DataFrame
        podcasts = pl.read_csv(csv_path)
        logger.info(f"Loaded {len(podcasts)} podcasts from {csv_path}")

        # Convert DataFrame to list of dictionaries
        podcasts_list = podcasts.to_dicts()

        # Process podcasts using `process_podcast` with list comprehension
        processed_podcasts = [
            result for podcast in tqdm(podcasts_list, desc="Processing Podcasts")
            if (result := process_podcast(podcast, text_folder)) is not None
        ]

        # Create the DataFrame
        podcasts_clean = pl.DataFrame(processed_podcasts)
        logger.info(f"Processed {len(podcasts_clean)} podcasts with transcripts")

        # Parse 'blog_date' to Date type
        podcasts_clean = podcasts_clean.with_columns([
            pl.col("blog_date").str.strptime(pl.Date, "%B %d, %Y").alias("blog_date")
        ])

        return podcasts_clean

    except Exception as e:
        logger.error(f"Failed to load and process podcasts: {str(e)}")
        raise e
# Standard library imports
import os
import time
import random
from collections import OrderedDict

# Third-party imports
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
import polars as pl
from tqdm import tqdm
from openai import RateLimitError
from typing import List, Dict, Any, Tuple

# LangChain imports
from langchain.chains import RetrievalQA, GraphCypherQAChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.vectorstores import Neo4jVector
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_driver(uri: str, user: str, password: str):
    """
    Creates and returns a Neo4j driver instance.
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Neo4j driver created successfully.")
        return driver
    except Neo4jError as e:
        logger.error(f"Failed to create Neo4j driver: {e}")
        raise

def initialize_embeddings(model_name: str = "text-embedding-ada-002"):
    """
    Initializes and returns OpenAI embeddings.
    """
    try:
        embeddings = OpenAIEmbeddings(model=model_name)
        logger.info(f"Embeddings initialized with model: {model_name}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        raise

def initialize_llm(model_name: str = "gpt-4o", temperature: float = 0.0):
    """
    Initializes and returns the Language Model.
    """
    try:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        logger.info(f"Language model initialized with model: {model_name}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize language model: {e}")
        raise

def initialize_vector_store(embeddings, uri: str, user: str, password: str, index_name: str = "vector_index"):
    """
    Initializes and returns the Neo4jVector store.
    """
    try:
        vector_store = Neo4jVector(
            embedding=embeddings,
            url=uri,
            username=user,
            password=password,
            index_name=index_name,
            embedding_node_property="embedding"
        )
        logger.info(f"Neo4jVector store initialized with index: {index_name}")
        return vector_store
    except Neo4jError as e:
        logger.error(f"Failed to initialize Neo4jVector store: {e}")
        raise

def create_vector_index(driver, cypher_query: str, index_name: str = "vector_index"):
    """
    Creates a vector index in Neo4j if it does not already exist.
    """
    try:
        with driver.session() as session:
            # Check if the index already exists
            result = session.run("SHOW INDEXES YIELD name WHERE name = $name", name=index_name)
            exists = any(record["name"] == index_name for record in result)
            
            if exists:
                logger.info(f"Index '{index_name}' already exists. Skipping creation.")
            else:
                session.run(cypher_query)
                logger.info(f"Index '{index_name}' created successfully.")
    except Neo4jError as e:
        logger.error(f"Failed to create vector index '{index_name}': {e}")
        raise

def create_documents_from_df(chunk_df: pl.DataFrame) -> List[Document]:
    """
    Converts a Polars DataFrame of chunks into a list of Document objects.

    Args:
        chunk_df (pl.DataFrame): DataFrame containing chunk data with required columns:
            - 'chunk_text'
            - 'post_url'
            - 'post_title'
            - 'series_number'
            - 'blog_date'
            - 'blog_title'
            - 'file_name'

    Returns:
        List[Document]: List of Document objects created from the dataframe.
    """
    required_columns = {
        'chunk_text',
        'post_url',
        'post_title',
        'series_number',
        'blog_date',
        'blog_title',
        'file_name'
    }

    # Validate that all required columns are present
    missing_columns = required_columns - set(chunk_df.columns)
    if missing_columns:
        logger.error(f"The following required columns are missing from the dataframe: {missing_columns}")
        raise ValueError(f"Missing columns: {missing_columns}")

    documents = []
    skipped_rows = 0

    logger.info("Starting to create Document objects from DataFrame.")

    for row in chunk_df.iter_rows(named=True):
        chunk_text = row.get('chunk_text', '').strip()
        if not chunk_text:
            skipped_rows += 1
            logger.debug(f"Skipping row with empty 'chunk_text': {row}")
            continue  # Skip rows without valid 'chunk_text'

        try:
            document = Document(
                page_content=chunk_text,
                metadata={
                    "post_url": row.get('post_url', ''),
                    "post_title": row.get('post_title', ''),
                    "series_number": row.get('series_number', ''),
                    "blog_date": row.get('blog_date', ''),
                    "blog_title": row.get('blog_title', ''),
                    "file_name": row.get('file_name', ''),
                }
            )
            documents.append(document)
        except Exception as e:
            skipped_rows += 1
            logger.warning(f"Failed to create Document for row {row}: {e}")

    logger.info(f"Created {len(documents)} Document objects.")
    if skipped_rows > 0:
        logger.warning(f"Skipped {skipped_rows} rows due to missing or invalid data.")

    return documents

# Function to estimate token count (simple estimation)
def estimate_tokens(text):
    return len(text.split())

# Function to create batches without exceeding max_tokens_per_batch
def create_batches(documents, max_tokens_per_batch=100000):
    batches = []
    current_batch = []
    current_tokens = 0

    for doc in documents:
        tokens = estimate_tokens(doc.page_content)
        if current_tokens + tokens > max_tokens_per_batch:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
        current_batch.append(doc)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches

def delete_all_indexes(driver):
    """
    Deletes all indexes in the Neo4j database.
    """
    try:
        with driver.session() as session:
            # Retrieve all index names
            indexes = session.run("SHOW INDEXES YIELD name")
            index_names = [record["name"] for record in indexes]
            
            if not index_names:
                logger.info("No indexes found to delete.")
                return
            
            # Iterate and drop each index individually
            for index_name in index_names:
                try:
                    session.run(f"DROP INDEX `{index_name}` IF EXISTS")
                    logger.info(f"Index '{index_name}' deleted successfully.")
                except Neo4jError as e:
                    logger.error(f"Failed to delete index '{index_name}': {e}")
    except Neo4jError as e:
        logger.error(f"Failed to retrieve indexes: {e}")
        raise

# Update the DELETE_ALL_INDEXES_CYPHER to use the new function
def delete_all_nodes_and_indexes(driver):
    """
    Deletes all nodes and indexes in the Neo4j database.
    """
    try:
        with driver.session() as session:
            # Delete all nodes
            session.run("MATCH (n) DETACH DELETE n;")
            logger.info("All nodes have been deleted successfully.")
        
        # Delete all indexes
        delete_all_indexes(driver)
        logger.info("All indexes have been deleted successfully.")
    except Neo4jError as e:
        logger.error(f"Failed to delete nodes or indexes: {e}")
        raise
    
# Function to embed and add a batch to Neo4jVector with retry logic
def embed_and_add_batch(batch, embeddings, db, batch_num, max_retries=5, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            db.add_texts(texts=texts, metadatas=metadatas)
            print(f"Batch {batch_num}: Successfully added to Neo4jVector.")
            break  # Exit retry loop on success
        except RateLimitError as e:
            wait_time = backoff_factor ** attempt + random.uniform(0, 1)
            print(f"Batch {batch_num}: Rate limit exceeded. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Batch {batch_num}: Unexpected error occurred: {e}")
            break
    else:
        print(f"Batch {batch_num}: Failed to add to Neo4jVector after {max_retries} attempts.")


def qa_pipeline(qa, question, max_retries=2, backoff_factor=2):
    """
    Takes a question as input and returns an answer using the RetrievalQA system.
    Implements retry logic with exponential backoff to handle rate limits.
    
    Args:
        qa: The RetrievalQA instance.
        question (str): The user's question.
        max_retries (int): Maximum number of retry attempts.
        backoff_factor (int): Factor for exponential backoff.
    
    Returns:
        tuple: (answer, source_documents)
    """
    max_retries = int(max_retries)
    backoff_factor = int(backoff_factor)
    # Ensure max_retries and backoff_factor are integers
    if not isinstance(max_retries, int):
        raise TypeError(f"max_retries must be an integer, got {type(max_retries).__name__}")
    if not isinstance(backoff_factor, int):
        raise TypeError(f"backoff_factor must be an integer, got {type(backoff_factor).__name__}")

    for attempt in range(max_retries):
        try:
            result = qa.invoke({"query": question})
            answer = result['result']
            source_documents = result['source_documents']
            return answer, source_documents
        except RateLimitError:
            wait_time = backoff_factor ** attempt + random.uniform(0, 1)
            logger.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return "I'm sorry, I couldn't process your request at this time.", []
    else:
        logger.error("Failed to process the request after multiple attempts.")
        return "I'm sorry, I couldn't process your request at this time.", []
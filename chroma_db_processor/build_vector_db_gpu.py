import chromadb
from chromadb.config import Settings
from chromadb.utils.batch_utils import create_batches
from sentence_transformers import SentenceTransformer
import torch
import uuid
import threading
import queue
import orjson
import json
import time
import logging

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
else:
    print("CUDA not available, using CPU")

# Constants
CHROMA_DB_NAME = 'chromadb-exp-test'
CHROMA_DB_DIR= f"../chromadbs/{CHROMA_DB_NAME}"
DEVICE = 'cuda'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DATASET_REVIEW_FILE = '../datasets/Amazon_Fashion.jsonl'
DATASET_META_FILE = '../datasets/meta_Amazon_Fashion.jsonl'
BATCH_SIZE = 5000
QUEUE_SIZE = 1000

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Progress tracking
progress_lock = threading.Lock()
processed_items = 0
total_items = 0

def count_total_lines(file_path):
    """Count total lines in a file (approximates total items)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

class ChromaEmbeddingFunction:
    """Embedding function for ChromaDB using SentenceTransformers on GPU."""

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

    def __call__(self, input):
        return self.model.encode(input, show_progress_bar=False, device=DEVICE).tolist()

    def name(self):
        return f"sentence-transformers-{EMBEDDING_MODEL_NAME}-{DEVICE}"

embedding_function = ChromaEmbeddingFunction()

def ppprint(x):
    if isinstance(x, dict):
        print(json.dumps(x, indent=2))
    else:
        print(x)

def safe_lower(val):
    if isinstance(val, str):
        return val.lower()
    elif isinstance(val, list):
        return ','.join(str(v).lower() for v in val)
    elif val is not None:
        return str(val).lower()
    else:
        return ''

def create_chroma_collections():
    """Creates and returns ChromaDB client and collections for product metadata and reviews."""
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    product_meta_col = client.get_or_create_collection(
        name="product_meta",
        embedding_function=embedding_function,
        metadata={"description": "Product metadata collection"}
    )
    product_review_col = client.get_or_create_collection(
        name="product_review",
        embedding_function=embedding_function,
        metadata={"description": "Product review collection"}
    )
    parent_asin_to_title = {}
    return client, product_meta_col, product_review_col, parent_asin_to_title

def print_progress(line_count, data_type):
    print(f"{data_type.capitalize()} line: {line_count}", end='\r')

def read_reviews(batch_size=BATCH_SIZE):
    """Generator that reads review data in batches from the dataset file."""
    batch_docs = []
    batch_reviews = []
    line_count = 0
    with open(DATASET_REVIEW_FILE, 'rb') as f:
        for line in f:
            line_count += 1
            if line_count % 1000 == 0:
                print_progress(line_count, 'reviews')
            review = orjson.loads(line.strip())
            fields = []
            if review.get('text'):
                fields.append(safe_lower(review['text']))
            else:
                continue
            review_doc = fields[0]
            batch_docs.append(review_doc)
            batch_reviews.append(review)
            if len(batch_docs) >= batch_size:
                yield batch_docs, batch_reviews
                batch_docs, batch_reviews = [], []
        if batch_docs:
            yield batch_docs, batch_reviews

def read_meta(batch_size=BATCH_SIZE):
    """Generator that reads product metadata in batches from the dataset file."""
    batch_docs = []
    batch_products = []
    line_count = 0
    with open(DATASET_META_FILE, 'rb') as f:
        for line in f:
            line_count += 1
            if line_count % 1000 == 0:
                print_progress(line_count, 'meta')
            product = orjson.loads(line.strip())
            fields = []
            for k in ['title']:
                v = product.get(k)
                if v:
                    fields.append(safe_lower(v))
            if not fields or not fields[0]:
                continue
            product_meta_doc = fields[0]
            batch_docs.append(product_meta_doc)
            batch_products.append(product)
            if len(batch_docs) >= batch_size:
                yield batch_docs, batch_products
                batch_docs, batch_products = [], []
        if batch_docs:
            yield batch_docs, batch_products

def producer_reviews(job_queue):
    """Producer thread: reads review batches and puts them into the job queue."""
    for docs, reviews in read_reviews():
        job_queue.put(('review', docs, reviews))
    logger.info("Producer-Reviews: Finished reading reviews")

def producer_meta(job_queue):
    """Producer thread: reads meta batches and puts them into the job queue."""
    for docs, products in read_meta():
        job_queue.put(('meta', docs, products))
    logger.info("Producer-Meta: Finished reading meta")

def encoder(job_queue, insert_queue_reviews, insert_queue_meta):
    """Encoder thread: gets batches from job_queue, encodes them individually, and puts into insert queues."""
    while True:
        item = job_queue.get()
        if item is None:
            logger.info("Encoder: Stopping")
            insert_queue_reviews.put(None)
            insert_queue_meta.put(None)
            return
        batch_type, docs, data = item
        logger.info(f"Encoding {len(docs)} documents")
        print(f"GPU now processing batch of {len(docs)} items")
        embeddings = embedding_function(docs)
        print(f"GPU processed batch, embeddings generated for {len(docs)} items")
        logger.info(f"Encoded {len(docs)} documents")
        if batch_type == 'review':
            # Prepare metadatas for reviews
            metadatas = [{"parent_asin": r['parent_asin']} for r in data]
            # Prepare ids for reviews
            ids = [f"review_{r['parent_asin']}_{uuid.uuid4()}" for r in data]
            # Put into reviews insert queue
            insert_queue_reviews.put((docs, metadatas, ids, embeddings))
        else:
            # Prepare metadatas for meta
            metadatas = [{"parent_asin": p['parent_asin'], "average_rating": p.get('average_rating')} for p in data]
            # Prepare ids for meta
            ids = [f"meta_{p['parent_asin']}_{uuid.uuid4()}" for p in data]
            # Put into meta insert queue
            insert_queue_meta.put((docs, metadatas, ids, embeddings))

def inserter_reviews(insert_queue, collection):
    """Inserter thread for reviews: gets from insert_queue and upserts into collection."""
    while True:
        item = insert_queue.get()
        if item is None:
            return
        docs, metadatas, ids, embeddings = item
        collection.upsert(
            documents=docs,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        logger.info(f"Inserted review batch of {len(docs)} items")
        with progress_lock:
            global processed_items
            processed_items += len(docs)
            print(f"Progress: {processed_items}/{total_items} items processed", end='\r')

def inserter_meta(insert_queue, collection):
    """Inserter thread for meta: gets from insert_queue and upserts into collection."""
    while True:
        item = insert_queue.get()
        if item is None:
            return
        docs, metadatas, ids, embeddings = item
        collection.upsert(
            documents=docs,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        logger.info(f"Inserted meta batch of {len(docs)} items")
        with progress_lock:
            global processed_items
            processed_items += len(docs)
            print(f"Progress: {processed_items}/{total_items} items processed", end='\r')

def populate_chroma_db(product_meta_col, product_review_col):
    """Run the pipelined population process for ChromaDB."""
    logger.info("Starting ChromaDB population with GPU optimization")

    # Initialize progress tracking
    global processed_items, total_items
    processed_items = 0
    total_reviews = count_total_lines(DATASET_REVIEW_FILE)
    total_meta = count_total_lines(DATASET_META_FILE)
    total_items = total_reviews + total_meta
    print(f"Total lines to process: {total_items}")

    # Create queues
    job_queue = queue.Queue(maxsize=QUEUE_SIZE)
    insert_queue_reviews = queue.Queue(maxsize=QUEUE_SIZE)
    insert_queue_meta = queue.Queue(maxsize=QUEUE_SIZE)

    # Number of encoder threads (configurable for GPU saturation)
    num_encoders = 50

    # Start threads for producers, encoders, and inserters
    encoder_threads = [threading.Thread(target=encoder, args=(job_queue, insert_queue_reviews, insert_queue_meta)) for _ in range(num_encoders)]
    threads = [
        threading.Thread(target=producer_reviews, args=(job_queue,)),
        threading.Thread(target=producer_meta, args=(job_queue,)),
        threading.Thread(target=inserter_reviews, args=(insert_queue_reviews, product_review_col)),
        threading.Thread(target=inserter_meta, args=(insert_queue_meta, product_meta_col))
    ] + encoder_threads

    for t in threads:
        t.start()

    # Wait for producers to finish
    threads[0].join()  # reviews producer
    threads[1].join()  # meta producer

    # Signal encoders to stop: put None for each encoder
    for _ in range(num_encoders):
        job_queue.put(None)

    # Wait for encoders and inserters
    for t in threads[2:]:
        t.join()

    logger.info("ChromaDB population completed")

if __name__ == "__main__":
    client, product_meta_col, product_review_col, parent_asin_to_title = create_chroma_collections()
    print("ChromaDB collections created and hashmap initialized.")
    populate_chroma_db(product_meta_col, product_review_col)
    query_text = "recommend me compression sleeves"
    print(f"\nQuerying ChromaDB for: '{query_text}'\n")
    try:
        results = product_meta_col.query(
            query_texts=[query_text],
            n_results=5
        )
        print("Query Results:")
        ppprint(results)
    except Exception as e:
        print(f"ChromaDB query failed: {e}")

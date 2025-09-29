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

# Constants
CHROMA_DB_NAME = 'chromadb-exp-test'
CHROMA_DB_DIR= f"../chromadbs/{CHROMA_DB_NAME}"
DEVICE = 'cuda'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DATASET_REVIEW_FILE = '../datasets/Amazon_Fashion.jsonl'
DATASET_META_FILE = '../datasets/meta_Amazon_Fashion.jsonl'
BATCH_SIZE = 5000

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class ChromaDBPopulator:
    """Handles GPU-optimized population of ChromaDB collections using pipelined processing."""

    def __init__(self, product_meta_col, product_review_col):
        self.product_meta_col = product_meta_col
        self.product_review_col = product_review_col
        self.job_queue = queue.Queue(maxsize=20)
        self.insert_queue_reviews = queue.Queue(maxsize=20)
        self.insert_queue_meta = queue.Queue(maxsize=20)
        self.batch_group_size = 10

    def populate(self):
        """Run the pipelined population process."""
        logger.info("Starting ChromaDB population with GPU optimization")

        # Start threads
        threads = [
            threading.Thread(target=self._producer_reviews),
            threading.Thread(target=self._producer_meta),
            threading.Thread(target=self._encoder),
            threading.Thread(target=self._inserter_reviews),
            threading.Thread(target=self._inserter_meta)
        ]

        for t in threads:
            t.start()

        # Wait for producers
        threads[0].join()  # reviews
        threads[1].join()  # meta

        # Signal encoder
        self.job_queue.put(None)

        # Wait for remaining
        for t in threads[2:]:
            t.join()

        logger.info("ChromaDB population completed")

    def _producer_reviews(self):
        """Producer thread: reads review batches."""
        for batch_docs, batch_reviews in read_reviews():
            self.job_queue.put(('review', batch_docs, batch_reviews))
        logger.info("Producer-Reviews: Finished reading reviews")

    def _producer_meta(self):
        """Producer thread: reads meta batches."""
        for batch_docs, batch_products in read_meta():
            self.job_queue.put(('meta', batch_docs, batch_products))
        logger.info("Producer-Meta: Finished reading meta")

    def _encoder(self):
        """Encoder thread: processes batches and encodes."""
        while True:
            collected_batches = []
            all_docs = []
            for _ in range(self.batch_group_size):
                item = self.job_queue.get()
                if item is None:
                    if collected_batches:
                        self._encode_and_queue_group(collected_batches, all_docs)
                    logger.info("Encoder: Stopping")
                    self.insert_queue_reviews.put(None)
                    self.insert_queue_meta.put(None)
                    return
                collected_batches.append(item)
                all_docs.extend(item[1])

            self._encode_and_queue_group(collected_batches, all_docs)

    def _encode_and_queue_group(self, collected_batches, all_docs):
        """Encode group and queue for insertion."""
        logger.info(f"Encoding {len(all_docs)} documents")
        all_embeddings = embedding_function(all_docs)
        logger.info(f"Encoded {len(all_docs)} documents")

        offset = 0
        for batch_type, batch_docs, batch_data in collected_batches:
            embeddings = all_embeddings[offset:offset + len(batch_docs)]
            if batch_type == 'review':
                metadatas = [{"parent_asin": r['parent_asin']} for r in batch_data]
                ids = [f"review_{r['parent_asin']}_{uuid.uuid4()}" for r in batch_data]
                queue = self.insert_queue_reviews
            else:
                metadatas = [{"parent_asin": p['parent_asin'], "average_rating": p.get('average_rating')} for p in batch_data]
                ids = [f"meta_{p['parent_asin']}_{uuid.uuid4()}" for p in batch_data]
                queue = self.insert_queue_meta
            queue.put((batch_docs, metadatas, ids, embeddings))
            offset += len(batch_docs)

    def _inserter_reviews(self):
        """Inserter thread for reviews."""
        while True:
            item = self.insert_queue_reviews.get()
            if item is None:
                return
            batch_docs, metadatas, ids, embeddings = item
            self.product_review_col.upsert(
                documents=batch_docs,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            logger.info(f"Inserted review batch of {len(batch_docs)} items")

    def _inserter_meta(self):
        """Inserter thread for meta."""
        while True:
            item = self.insert_queue_meta.get()
            if item is None:
                return
            batch_docs, metadatas, ids, embeddings = item
            self.product_meta_col.upsert(
                documents=batch_docs,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            logger.info(f"Inserted meta batch of {len(batch_docs)} items")


def populate_chroma_db(product_meta_col, product_review_col):
    """Wrapper function for backward compatibility."""
    populator = ChromaDBPopulator(product_meta_col, product_review_col)
    populator.populate()

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

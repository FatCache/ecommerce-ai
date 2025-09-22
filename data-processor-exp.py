import chromadb
from chromadb.config import Settings
from chromadb.utils.batch_utils import create_batches
chroma_db_name = 'chromadb-exp'  # New persistent DB name

# Embedding function using sentence-transformers on CUDA
from sentence_transformers import SentenceTransformer
import torch

# Load model on GPU if available
device = 'cuda'
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
#embedding_model = SentenceTransformer('llmware/all-mini-lm-l6-v2-ov', device='npu')

# ChromaDB expects the embedding_function to have a .name() method

# ChromaDB expects the embedding_function to have __call__(self, input: List[str])
class ChromaEmbeddingFunction:
    def __call__(self, input):
        return embedding_model.encode(input, show_progress_bar=False, device=device).tolist()
    def name(self):
        return "sentence-transformers-all-MiniLM-L6-v2-gpu"

embedding_function = ChromaEmbeddingFunction()
import uuid
import threading

def create_chroma_collections():
    # Create persistent ChromaDB client with embedding function
    client = chromadb.PersistentClient(path=f"./chromadbs/{chroma_db_name}")

    # Create or get collections with embedding function
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

    # Hashmap for parent_asin to title
    parent_asin_to_title = {}
    return client, product_meta_col, product_review_col, parent_asin_to_title

import orjson
import json

file_review = 'datasets/Amazon_Fashion.jsonl'
file_meta = 'datasets/meta_Amazon_Fashion.jsonl'

ppprint = lambda x: print(json.dumps(x, indent=2)) if isinstance(x, dict) else print(x)

curr_line_review = 0

def read_reviews(batch_size=1000):
    batch_docs, batch_reviews = [], []
    global curr_line_review
    with open(file_review, 'rb') as f:
        for line in f:
            curr_line_review += 1
            if curr_line_review % 1000 == 0:
                print_progress_line()
            review = orjson.loads(line.strip())
            #ppprint(review)
            # Only include non-empty fields
            fields = []
            # if review.get('title'):
            #     fields.append(f"title:{safe_lower(review['title'])}")
            if review.get('text'):
                fields.append(f"{safe_lower(review['text'])}")
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
        curr_line_review = 'FINISHED'

def read_meta(batch_size=1000):
    batch_docs, batch_products = [], []
    global curr_line_meta
    with open(file_meta, 'rb') as f:
        for line in f:
            curr_line_meta += 1
            if curr_line_meta % 1000 == 0:
                print_progress_line()
            product = orjson.loads(line.strip())
            
            fields = []
            for k in ['title']:
                v = product.get(k)
                if v:
                    fields.append(f"{safe_lower(v)}")
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
        curr_line_meta = 'FINISHED'

curr_line_meta = 0

def print_progress_line():
    global curr_line_review, curr_line_meta
    print(f"Reviews line: {curr_line_review} | Meta line: {curr_line_meta}", end='\r') 



def safe_lower(val):
    if isinstance(val, str):
        return val.lower()
    elif isinstance(val, list):
        return ','.join(str(v).lower() for v in val)
    elif val is not None:
        return str(val).lower()
    else:
        return ''

BATCH_SIZE = 5000
def populate_chroma_db(product_meta_col, product_review_col):
    def insert_reviews():
        for batch_docs, batch_reviews in read_reviews(BATCH_SIZE):
            metadatas = [{"parent_asin": review['parent_asin']} for review in batch_reviews]
            ids = [f"review_{review['parent_asin']}_{uuid.uuid4()}" for review in batch_reviews]
            print("\nInserting product review start...")
            product_review_col.upsert(
                documents=batch_docs,
                metadatas=metadatas,
                ids=ids
            )
            print("Inserting product review finished...")

    def insert_meta():
        for batch_docs, batch_products in read_meta(BATCH_SIZE):
            metadatas = [{"parent_asin": product['parent_asin'], "average_rating": product['average_rating']} for product in batch_products]
            ids = [f"meta_{product['parent_asin']}_{uuid.uuid4()}" for product in batch_products]
            product_meta_col.upsert(
                documents=batch_docs,
                metadatas=metadatas,
                ids=ids
            )
            
            # this should be externalize to say database for later faster retrival duing chat
            # for product in batch_products:
            #     if product['parent_asin'] not in parent_asin_to_title:
            #         parent_asin_to_title[product['parent_asin']] = product['title']
    t1 = threading.Thread(target=insert_reviews)
    t2 = threading.Thread(target=insert_meta)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
if __name__ == "__main__":
    # Example: create persistent ChromaDB collections and hashmap
    client, product_meta_col, product_review_col, parent_asin_to_title = create_chroma_collections()
    print("ChromaDB collections created and hashmap initialized.")

    # Persist the database to disk
    populate_chroma_db(product_meta_col, product_review_col)

    # Example query to ChromaDB
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

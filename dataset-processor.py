import chromadb
from chromadb.config import Settings
from chromadb.utils.batch_utils import create_batches
import orjson  # Add this import

chroma_db_name = 'chromadb_v1'

def create_chroma_collections():
    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(path=f"./chromadbs/{chroma_db_name}")

    # Create or get collections
    product_meta_col = client.get_or_create_collection(
        name="product_meta",
        metadata={"description": "Product metadata collection"}
    )
    product_review_col = client.get_or_create_collection(
        name="product_review",
        metadata={"description": "Product review collection"}
    )

    # Hashmap for parent_asin to title
    parent_asin_to_title = {}
    return client, product_meta_col, product_review_col, parent_asin_to_title

import json
import uuid
import threading

file_review = 'datasets/Amazon_Fashion.jsonl'
file_meta = 'datasets/meta_Amazon_Fashion.jsonl'

ppprint = lambda x: print(json.dumps(x, indent=2)) if isinstance(x, dict) else print(x)

def read_reviews(batch_size=1000):
    batch_docs, batch_reviews = [], []
    with open(file_review, 'rb') as f:
        for line in f:
            review = orjson.loads(line.strip())
            # Only include non-empty fields
            fields = []
            if review.get('title'):
                fields.append(f"title:{safe_lower(review['title'])}")
            if review.get('text'):
                fields.append(f"review:{safe_lower(review['text'])}")
            if review.get('rating'):
                fields.append(f"rating:{safe_lower(review['rating'])}")
            review_doc = "\n".join(fields)
            batch_docs.append(review_doc)
            batch_reviews.append(review)
            if len(batch_docs) >= batch_size:
                yield batch_docs, batch_reviews
                batch_docs, batch_reviews = [], []
        if batch_docs:
            yield batch_docs, batch_reviews

def read_meta(batch_size=1000):
    batch_docs, batch_products = [], []
    with open(file_meta, 'rb') as f:
        for line in f:
            product = orjson.loads(line.strip())
            # Only include selected fields
            fields = []
            for k in ['main_category', 'title', 'rating_number']:
                v = product.get(k)
                if v:
                    fields.append(f"{k}={safe_lower(v)}")
            product_meta_doc = "|".join(fields)
            batch_docs.append(product_meta_doc)
            batch_products.append(product)
            if len(batch_docs) >= batch_size:
                yield batch_docs, batch_products
                batch_docs, batch_products = [], []
        if batch_docs:
            yield batch_docs, batch_products
            

def safe_lower(val):
    if isinstance(val, str):
        return val.lower()
    elif isinstance(val, list):
        return ','.join(str(v).lower() for v in val)
    elif val is not None:
        return str(val).lower()
    else:
        return ''

def populate_chroma_db():
    def insert_reviews():
        for batch_docs, batch_reviews in read_reviews():
            metadatas = [{"parent_asin": review['parent_asin']} for review in batch_reviews]
            ids = [f"review_{review['parent_asin']}_{uuid.uuid4()}" for review in batch_reviews]
            product_review_col.upsert(
                documents=batch_docs,
                metadatas=metadatas,
                ids=ids
            )

    def insert_meta():
        for batch_docs, batch_products in read_meta():
            metadatas = [{"parent_asin": product['parent_asin']} for product in batch_products]
            ids = [f"meta_{product['parent_asin']}_{uuid.uuid4()}" for product in batch_products]
            product_meta_col.upsert(
                documents=batch_docs,
                metadatas=metadatas,
                ids=ids
            )
            for product in batch_products:
                if product['parent_asin'] not in parent_asin_to_title:
                    parent_asin_to_title[product['parent_asin']] = product['title']

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
    #populate_chroma_db()

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

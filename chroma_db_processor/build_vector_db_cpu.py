import chromadb
from chromadb.config import Settings
chroma_db_name = 'chromadb_v1'
import uuid
import threading

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

curr_line_meta = 0

def print_progress_line():
    global curr_line_review, curr_line_meta
    print(f"Reviews line: {curr_line_review} | Meta line: {curr_line_meta}", end='\r') 

def read_meta(batch_size=1000):
    batch_docs, batch_products = [], []
    global curr_line_meta
    with open(file_meta, 'rb') as f:
        for line in f:
            curr_line_meta += 1
            if curr_line_meta % 1000 == 0:
                print_progress_line()
            product = orjson.loads(line.strip())
            #ppprint(product)
            # Only include selected fields
            fields = []
            for k in ['title']:
                v = product.get(k)
                if v:
                    fields.append(f"{safe_lower(v)}")
            if not fields[0]:
                continue
            product_meta_doc = fields[0]
            batch_docs.append(product_meta_doc)
            batch_products.append(product)
            if len(batch_docs) >= batch_size:
                yield batch_docs, batch_products
                batch_docs, batch_products = [], []
        if batch_docs:
            yield batch_docs, batch_products

def read_meta_by_line():

    with open(file_meta, 'rb') as f:
        for line in f:
            global curr_line
            curr_line += 1
            print(f"Reading line {curr_line}", end='\r')
            review = orjson.loads(line.strip())
            #ppprint(review)
            # Only include non-empty fields
            fields = []
            if review.get('title'):
                fields.append(f"title:{safe_lower(review['title'])}")
            if review.get('text'):
                fields.append(f"review:{safe_lower(review['text'])}")
            if review.get('rating'):
                fields.append(f"rating:{safe_lower(review['rating'])}")
            review_doc = "\n".join(fields)
            yield review_doc, review
            

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
        for batch_docs, batch_reviews in read_reviews(5000):
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
        for batch_docs, batch_products in read_meta(5000):
            metadatas = [{"parent_asin": product['parent_asin'], "average_rating": product['average_rating']} for product in batch_products]
            ids = [f"meta_{product['parent_asin']}_{uuid.uuid4()}" for product in batch_products]
            print("\nInserting product meta start...")
            product_meta_col.upsert(
                documents=batch_docs,
                metadatas=metadatas,
                ids=ids
            )
            print("Inserting product meta finished...")
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



# if __name__ == "__main__":
#     meta_iter =  read_meta_by_line()
#     set_main_categories = set()

#     # while True: #
#     #     docs, product = next(meta_iter)
#     #     if not product:
#     #         break
#     #     if product.get('main_category'):
#     #         set_main_categories.add(product['main_category'])
#         # with open('main_categories.txt', 'w', encoding='utf-8') as f:
#     #     for cat in sorted(set_main_categories):
#     #         f.write(f"{cat}\n")

#     for _ in range(10):
#         try:
#             docs, product = next(meta_iter)
#             if product.get('title'):
#                 ppprint(product['title'])
#                 ppprint(product['average_rating'])
#         except StopIteration:
#             break


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

import chromadb

def get_chromadb():
    chroma_db_name = 'chromadb_v1'
    client = chromadb.PersistentClient(path=f"./chromadbs/{chroma_db_name}")
    product_meta_collection = client.get_or_create_collection(
        name="product_meta",
        metadata={"description": "Product metadata collection"}
    )
    product_review_collection = client.get_or_create_collection(
        name="product_review",
        metadata={"description": "Product review collection"}
    )
    print("Models configured and ChromaDB initialized.")
    
    return client, product_meta_collection, product_review_collection
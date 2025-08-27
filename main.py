import chromadb

client = chromadb.PersistentClient(path="chroma-db")




'''
We will build two collections. 
collection_meta based on
- main_category
- title
- average rating
- feature
- description
- price
- asin
collection_review
- 
'''
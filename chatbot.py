import chromadb
from chromadb.config import Settings
from chromadb.utils.batch_utils import create_batches
import google.generativeai as genai
import os

#genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
genai.configure(api_key='AIzaSyCvN4cbmBn6969Q8jaSAyCIuw0P_yVi8WU')

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

# model selection gemini-2.5-pro, gemini-2.5-flash, geminigemini-2.5-flash-lite
model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite",
                              generation_config=generation_config,
                              safety_settings=safety_settings)


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
    
    return client, product_meta_collection, product_review_collection

def start_chat():
    client, product_meta_collection, product_review_collection = get_chromadb()
    query_text = "recommend me compression sleeves for running" # comes from input 

    context_prompt = "You are a chat bot that specializes in providing recommendation for user query. For each user input, you must ascertain if" \
    "more data is required or not from user to do a search on RAG database. It is important to never query database all the time. Query to database or RAG should be" \
    "be only performed when all necessary material as been gathered. This reply will be structured into YAML. Current action that can be performed is <SEARCH_RAG> or <ASK_USER>" \
    "<ASK_USER> means, you, the chatbot will ask another to gather more information until enough information has been gatehered to query RAG database. You need to take input from the" \
    "user to get their feedback on result from RAG that you provide. If user is not satisfied. Ask another question to understand their need to create better statement to seerch" \
    "vector database. Result from RAG will be enclosed in <RESULT_RAG> tags. "

    print(f"\nQuerying ChromaDB for: '{query_text}'\n")
    while True:
      try:
          # need to add some stuff here to tell its a bot and keep reponse in specific format
          # intelligent, somehow response needs to convert to use chromadb searches??
          print(f"\nSending prompt to Google Gemini...\n")
          
          convo = model.start_chat(history=[])
          convo.send_message(query_text)
          gemini_response = convo.last.text
          print(f"\nGemini Response: {gemini_response}\n")

          results = product_meta_collection.query(
              query_texts=[query_text],
              n_results=5
          )
          print("Query Results:")
          print(results)
      except Exception as e:
          print(f"ChromaDB query failed: {e}")


if __name__ == "__main__":
    start_chat()
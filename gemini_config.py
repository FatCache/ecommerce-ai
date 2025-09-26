import google.generativeai as genai
import os
from prompts import context_prompt

def configure_gemini():
    # Use environment variable for API key
    # api_key = os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     raise ValueError("GOOGLE_API_KEY environment variable not set.")
    # genai.configure(api_key=api_key)

    genai.configure(api_key='AIzaSyCvN4cbmBn6969Q8jaSAyCIuw0P_yVi8WU')
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

    # Main model for chat and complex tasks
    main_model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings,
                                  system_instruction=context_prompt)
    
    # Cheaper model for summarization tasks
    # Using gemini-1.0-flash-001 as a potentially cheaper alternative
    summarization_model = genai.GenerativeModel(model_name="gemini-1.0-flash-001",
                                                generation_config=generation_config, # Reusing generation config, but could be tuned separately
                                                safety_settings=safety_settings) # Reusing safety settings

    return main_model, summarization_model
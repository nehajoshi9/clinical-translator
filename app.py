import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

# Retrieve the key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# Check if the key is present
if not GEMINI_API_KEY:
    st.error("FATAL ERROR: GEMINI_API_KEY not found in .env file or environment variables.")
    st.stop()

# Initialize the client using the key
client = genai.Client(api_key=GEMINI_API_KEY)

# ... rest of your code ...
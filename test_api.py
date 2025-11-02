import os
from google import genai
from dotenv import load_dotenv

# Load key securely from your .env file
load_dotenv() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found. Check your .env file.")
else:
    try:
        # 1. Initialize Client with the key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # 2. Test with the simplest request and a stable model
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'API key validation successful' and nothing else."
        )
        
        # 3. Print the result
        print("--- API Test Result ---")
        print(f"✅ SUCCESS! Model Response:\n{response.text.strip()}")
        
    except Exception as e:
        print(f"❌ API Test Failed! Error: {e}")
        print("Your key may be invalid, restricted, or billing may not be enabled.")
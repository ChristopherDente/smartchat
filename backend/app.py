import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

# Load environment variables from .env file
load_dotenv(override=True)

app = FastAPI()

# Enable CORS for the local dev server on port 8080
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure the Gemini API
api_key = os.getenv("GEMINI_API_KEY")
is_configured = False

if api_key and api_key != "your_gemini_api_key_here":
    genai.configure(api_key=api_key)
    # Ensure the AI adopts the SmartChat persona
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction="You are SmartChat AI, a highly advanced, helpful, and friendly virtual assistant built into the SmartChat messaging platform. Always answer politely and concisely. If asked who you are, say you are SmartChat AI. If asked who developed you or created you, say you were developed by Christopher Comteq."
    )
    # Start a persistent chat session to remember conversation history
    chat_session = model.start_chat(history=[])
    is_configured = True
else:
    model = None
    chat_session = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not is_configured:
        raise HTTPException(
            status_code=500, 
            detail="Gemini API key is not configured. Please add it to the backend/.env file and restart."
        )
        
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    try:
        # Give the AI context of the exact Philippine Time before passing the message
        current_time = datetime.now(ZoneInfo('Asia/Manila')).strftime('%A, %Y-%m-%d %I:%M %p (PHT)')
        context_injected_message = f"[System Context - Current Real-Time: {current_time}]\nUser: {user_message}"
        
        response = chat_session.send_message(context_injected_message)
        # Using response.text to extract standard text payload
        return ChatResponse(response=response.text)
    except Exception as e:
        print(f"Error during generative request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error connecting to the model")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5001, reload=True)

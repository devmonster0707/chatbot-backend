import os
import openai
import uuid  # To generate unique user session IDs
from fastapi import FastAPI,  Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from mangum import Mangum
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment configuration
access_token = os.getenv("OPENAI_API_KEY", "")
if not access_token:
    raise KeyError("OPENAI_API_KEY not found in environment.")
openai.api_key = access_token

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
if not PINECONE_API_KEY :
    raise KeyError("PINECONE_API_KEY not found in environment.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "chatbot-memory"  # You can choose a custom name for the Pinecone index

# Check if the index exists, create it if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI's embedding dimension size
        metric='euclidean',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Create an Index object
index = pc.Index(index_name)

# FastAPI app setup
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

# Create handler for AWS Lambda
handler = Mangum(app)

origins = [
    "http://localhost:3000",
    "https://ai-chatbot-five-bice.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)


EXCLUDED_CHATS = ["image", "info"]
INITIAL_CHATLOG = [
    {"role": "system", "content": "You are a helpful assistant"},
    {
        "role": "info",
        "content": (
            "Hello!\n\nI'm a personal assistant chatbot. I will respond as"
            " best I can to any messages you send me.\n\nHow may I assist you today?"
        ),
    },
]



class UserInputIn(BaseModel):
    prompt: str



@app.get("/")
async def get_chat_logs():
    return list(INITIAL_CHATLOG)

# Function to generate vector embeddings
def generate_embeddings(text: str) -> list:
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    embeddings = response["data"][0]["embedding"]
    return embeddings

@app.post("/")
async def chat(
    request: Request,
    user_input: UserInputIn,
     sessionId: str = Query(..., description="Session ID from the frontend")
):
    # global chat_log

    # Generate a unique session ID for the user (this can be passed from the frontend or generated dynamically)
    user_session_id = sessionId  # Generate a unique ID for each user

    # Append user input to chat log
    user_message = user_input.prompt

    # Generate embeddings for the user message
    user_embeddings = generate_embeddings(user_message)

    # Use the user's session ID as the namespace to separate data
    namespace = user_session_id

    # Upsert the vector embeddings into Pinecone index (storing only embeddings and metadata)
    vector_id = str(uuid.uuid4())
    # We store only the vector and metadata (raw content of the message)
    index.upsert([(vector_id, user_embeddings, {"content": user_message})], namespace=namespace)

    # Corrected query with keyword arguments, using the user's session as the namespace
    query_results = index.query(vector=user_embeddings, top_k=5, include_metadata=True, namespace=namespace)
    
    relevant_memory = ""
    for match in query_results["matches"]:
        relevant_memory += match["metadata"]["content"] + "\n"

    # Create AI prompt combining relevant memory and new input
    ai_prompts = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": relevant_memory}
    ]

    # Make call to OpenAI API for the response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=ai_prompts,
        temperature=0.8,
        max_tokens=500,
    )

    # Get response from OpenAI API
    bot_response = response.get("choices", [])[0].get("message").get("content")

    return {"role": "assistant", "content": bot_response}

@app.delete("/")
async def delete_history(sessionId: str = Query(..., description="Session ID from the frontend")):
    # Use the session ID passed from the frontend to delete the data
    namespace = sessionId

    # Check if the index exists
    if index_name not in pc.list_indexes().names():
        raise HTTPException(status_code=404, detail="Index not found")

    # Delete all vectors associated with the given session's namespace
    try:
        index.delete(delete_all=True, namespace=namespace)
        return list(INITIAL_CHATLOG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session history: {str(e)}")

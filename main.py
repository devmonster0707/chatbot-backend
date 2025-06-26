import os
import openai
from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import APIKeyHeader, APIKeyQuery
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

api_key = os.getenv("ACCEPTED_API_KEY", "")
if not api_key:
    raise KeyError("ACCEPTED_API_KEY not found in environment.")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise KeyError("PINECONE_API_KEY or PINECONE_ENVIRONMENT not found in environment.")

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

# Create handler for AWS Lambda
handler = Mangum(app)

origins = [
    "http://localhost:8000",
    "http://localhost:3000",
    "https://ai-chatbot-git-main-nathanrcobb.vercel.app",
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

api_key_query = APIKeyQuery(name="api-key", auto_error=False)
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

EXCLUDED_CHATS = ["image", "info"]
INITIAL_CHATLOG = [
    {"role": "system", "content": "You are a helpful assistant"},
    {
        "role": "info",
        "content": (
            "Hello!\n\nI'm a personal assistant chatbot. I will respond as"
            " best I can to any messages you send me.\n\nI can also generate"
            " images based on a prompt you send using '/image ' followed by"
            " the prompt.\n\nHow may I assist you today?"
        ),
    },
]

# Log of all user, system, and assistant prompts
chat_log = list(INITIAL_CHATLOG)

class UserInputIn(BaseModel):
    prompt: str

def get_api_key(
    api_key_query: str = Security(api_key_query),
    api_key_header: str = Security(api_key_header),
) -> str:
    if api_key_query == api_key:
        return api_key_query
    if api_key_header == api_key:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

@app.get("/docs", include_in_schema=False)
async def get_documentation(api_key: str = Security(get_api_key)):
    openapi_url = "/openapi.json"
    if api_key:
        openapi_url += f"?api-key={api_key}"
    return get_swagger_ui_html(openapi_url=openapi_url, title="docs")

@app.get("/openapi.json", include_in_schema=False)
async def openapi(api_key: str = Security(get_api_key)):
    return get_openapi(title="FastAPI", version="0.1.0", routes=app.routes)

@app.get("/")
async def get_chat_logs():
    global chat_log
    return chat_log

# Function to generate vector embeddings
def generate_embeddings(text: str) -> list:
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    embeddings = response["data"][0]["embedding"]
    return embeddings

@app.post("/")
async def chat(
    request: Request,
    user_input: UserInputIn,
    api_key: str = Security(get_api_key),
):
    global chat_log

    # Append user input to chat log
    user_message = user_input.prompt
    chat_log.append({"role": "user", "content": user_message})

    # Generate embeddings for the user message
    user_embeddings = generate_embeddings(user_message)

    # Upsert the data (user_embeddings) into the Pinecone index
    vector_id = str(len(chat_log))  # Unique ID for each message
    index.upsert([(vector_id, user_embeddings, {"content": user_message})])

    # Corrected query with keyword arguments
    query_results = index.query(vector=user_embeddings, top_k=5, include_metadata=True)
    
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

    # Append assistant response to chat log
    chat_log.append({"role": "assistant", "content": bot_response})

    return chat_log

@app.delete("/")
async def clear_chat_log(api_key: str = Security(get_api_key)):
    global chat_log, INITIAL_CHATLOG
    chat_log = list(INITIAL_CHATLOG)
    return chat_log

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)

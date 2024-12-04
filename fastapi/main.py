from fastapi import FastAPI, HTTPException, Depends, Query
from typing import Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import snowflake.connector
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
import json
import logging
import logging
from fastapi.logger import logger as fastapi_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fastapi_logger.setLevel(logging.INFO)


# Load environment variables
load_dotenv()

# Initialize clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or get Pinecone index
INDEX_NAME = "test"
if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )
index = pc.Index(INDEX_NAME)

# Snowflake connection configuration
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA')
}

# JWT and security configurations
SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15

# FastAPI app
app = FastAPI()

# OAuth2PasswordBearer for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Snowflake database connection
def get_db_connection():
    return snowflake.connector.connect(
        user=SNOWFLAKE_CONFIG['user'],
        password=SNOWFLAKE_CONFIG['password'],
        account=SNOWFLAKE_CONFIG['account'],
        warehouse=SNOWFLAKE_CONFIG['warehouse'],
        database=SNOWFLAKE_CONFIG['database'],
        schema=SNOWFLAKE_CONFIG['schema']
    )

# Utility functions
def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def get_user(username: str):
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT USERNAME, PASSWORD, CREATED_AT FROM USERS WHERE USERNAME = %s", (username,))
        row = cursor.fetchone()
        return {"username": row[0], "password": row[1], "created_at": row[2]} if row else None
    finally:
        cursor.close()
        connection.close()

def create_user(username: str, password: str):
    hashed_password = get_password_hash(password)
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            """INSERT INTO USERS (USERNAME, PASSWORD, CREATED_AT) VALUES (%s, %s, %s)""",
            (username, hashed_password, datetime.utcnow())
        )
        connection.commit()
    finally:
        cursor.close()
        connection.close()

def inspect_index():
    results = index.query(vector=[0] * 1536, top_k=10, include_metadata=True)
    return results

# Utility Functions (unchanged from your original code)
def get_embedding(text, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    MAX_TOKENS = 8192

    if len(tokens) > MAX_TOKENS:
        text = encoding.decode(tokens[:MAX_TOKENS])

    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def retrieve_information(user_query):
    try:
        logging.info("Retrieving information for query: %s", user_query)
        query_embedding = get_embedding(user_query)
        results = index.query(vector=query_embedding, top_k=4, include_metadata=True)

        logging.debug("Query results: %s", results)

        RELEVANCE_THRESHOLD = 0.8
        relevant_matches = [
            match for match in results["matches"] if match.get('score', 0) >= RELEVANCE_THRESHOLD
        ]

        if not relevant_matches:
            logging.info("No relevant matches found for query: %s", user_query)
            return None

        logging.info("Relevant matches found: %s", relevant_matches)
        context_info = " ".join([f"Chunk ID: {match['metadata'].get('chunk_id', 'Unknown')}. Text: {match['metadata'].get('text', '').strip()}"
                               for match in relevant_matches])
        return context_info
    except Exception as e:
        logging.error("Error in retrieve_information: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error during information retrieval")


def generate_plan(user_query, context_info, current_plan=None):
    try:
        if not context_info:
            logging.info("No relevant context found, skipping plan generation")
            return "No relevant information found in the Knowledge Base"

        system_prompt = (
            "You are an assistant that creates specific and actionable MVP-style structured JSON learning plans based on the user's query. "
            "Focus on clear objectives, key topics, step-by-step actions, a realistic timeline, and clear expected outcomes. "
            "Adjust the number of lessons to suit the user's query and complexity of the topic. "
            "Do not include any explanations or comments. Return only valid JSON output in this format:\n"
            "{\n"
            '  "Objective": "...",\n'
            '  "KeyTopics": ["...", "..."],\n'
            '  "Weeks": [\n'
            '    {"week": "Week 1", "title": "...", "details": "..."},\n'
            '    {"week": "Week 2", "title": "...", "details": "..."},\n'
            "    ...\n"
            "  ],\n"
            '  "Timeline": "...",\n'
            '  "ExpectedOutcome": "..."'
            "\n}"
        )

        if current_plan:
            current_plan_text = json.dumps(current_plan, indent=2)
            user_prompt = (
                f"Here is the current learning plan:\n{current_plan_text}\n\n"
                f"User Query: {user_query}\n"
                "Update the learning plan based on the query. Make sure to incorporate the user's new focus or requirements "
                "while retaining the structure and relevant parts of the existing plan."
            )
        else:
            user_prompt = (
                f"User Query: {user_query}\nContext: {context_info}\n"
                "Create a new structured learning plan based on the query and context."
            )

        # Log the user prompt to see the input to OpenAI API
        logging.debug("User prompt: %s", user_prompt)

        plan_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        logging.debug("Plan response: %s", plan_response)

        return plan_response.choices[0].message.content
    except Exception as e:
        logging.error("Error in generate_plan: %s", str(e))
        return '{"error": "Plan generation failed due to an internal error."}'

def validate_and_clean_json(response_text):
    try:
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx == -1 or end_idx == -1:
            return None

        json_text = response_text[start_idx:end_idx]
        return json.loads(json_text)
    except Exception as e:
        return None

def summarize_plan(plan):
    summary_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that creates simple, logical, and professional summaries of learning plans. "
                    "The output should be concise, focused, and practical. Avoid lists or unnecessary detail—describe the plan logically and clearly in a way that feels natural."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Learning Plan: {plan}\n"
                    "Summarize the learning plan into a practical and logical overview that explains what it achieves and how it is structured. "
                    "Keep it brief and easy to understand without sounding overly simplified or verbose."
                )
            },
        ]
    )
    return summary_response.choices[0].message.content

# API Endpoints
@app.post("/signup")
async def signup(username: str = Query(...), password: str = Query(...)):
    if get_user(username):
        raise HTTPException(status_code=400, detail="Username already registered")
    create_user(username, password)
    return {"message": "User created successfully"}

@app.post("/login")
async def login(username: str = Query(...), password: str = Query(...)):
    user = get_user(username)
    if not user or not verify_password(password, user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/query")
async def query_router(request: dict):
    user_query = request.get("user_query")
    current_plan = request.get("current_plan", None)
    current_summary = request.get("current_summary", None)  # Capture the current summary to retain it

    if not user_query:
        raise HTTPException(status_code=400, detail="User query is required.")

    logging.info("Processing query: %s", user_query)

    # Initialize response object
    response_data = {
        "plan": None,
        "summary": current_summary,
        "response": "Unable to process the query. Please try again later."
    }

    try:
        # Check for relevance in the knowledge base
        context_info = retrieve_information(user_query)

        if context_info:
            # Case 1: Relevant context found → Generate/Update learning plan
            learning_plan_json = generate_plan(user_query, context_info, current_plan)
            plan = validate_and_clean_json(learning_plan_json)

            if not plan:
                response_data["response"] = "Failed to generate or parse learning plan."
            else:
                summary = summarize_plan(plan)
                response_data.update({
                    "plan": plan,
                    "summary": summary,
                    "response": "I've updated the plan based on your query."
                })

        else:
            # Case 2: No relevant context found → Retain the previous plan and summary
            if current_plan:
                fallback_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                   messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant specializing in data science. The knowledge base contains vectorized content from "
                                "GeeksforGeeks covering all data science topics. If the user's query is relevant to the knowledge base, provide a concise, "
                                "accurate, and helpful response. If the query is unrelated to data science, politely explain that the knowledge base is "
                                "focused on data science topics and suggest the user reframe their query if applicable."
                            )
                        },
                        {"role": "user", "content": user_query},
                    ] ,
                )
                response_data.update({
                    "plan": current_plan,
                    "summary": current_summary,
                    "response": fallback_response.choices[0].message.content
                })
            else:
                # No plan exists
                fallback_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant specializing in data science. The knowledge base contains vectorized content from "
                                "GeeksforGeeks covering all data science topics. If the user's query is relevant to the knowledge base, provide a concise, "
                                "accurate, and helpful response. If the query is unrelated to data science, politely explain that the knowledge base is "
                                "focused on data science topics and suggest the user reframe their query if applicable."
                            )
                        },
                        {"role": "user", "content": user_query},
                    ],
                )
                response_data.update({
                    "response": fallback_response.choices[0].message.content
                })
    except Exception as e:
        logging.error("Error processing query: %s", str(e))
        response_data["response"] = "An error occurred while processing your query. Please try again later."

    return response_data


@app.get("/")
async def root():
    return {"message": "Welcome to the API!"}

@app.middleware("http")
async def log_requests(request, call_next):
    logging.info("Request URL: %s %s", request.method, request.url)
    response = await call_next(request)
    logging.info("Response status: %s", response.status_code)
    return response

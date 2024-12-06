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
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends

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

# Dependency for extracting username from the token
async def get_current_username(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

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
        if not context_info and not current_plan:
            logging.info("No relevant context or existing plan found, skipping plan generation")
            return "No relevant information found in the Knowledge Base"

        system_prompt = (
            "You are an assistant that creates specific and actionable MVP-style structured JSON learning plans based on the user's query. "
            "Focus on clear objectives, key topics, step-by-step actions, and clear expected outcomes. "
            "If the query refers to updating an existing plan, modify only the relevant sections to reflect the user's request. "
            "Do not include any explanations or comments. Return only valid JSON output in this format:\n"
            "{\n"
            '  "Objective": "...",\n'
            '  "KeyTopics": ["...", "..."],\n'
            '  "Modules": [\n'
            '    {"module": 1, "title": "...", "description": "..."},\n'
            '    {"module": 2, "title": "...", "description": "..."},\n'
            "    ...\n"
            "  ],\n"
            '  "ExpectedOutcome": "..."'
            "\n}"
        )

        if current_plan:
            current_plan_text = json.dumps(current_plan, indent=2)
            user_prompt = (
                f"Here is the current learning plan:\n{current_plan_text}\n\n"
                f"User Query: {user_query}\n"
                "Update the learning plan based on the query. Modify only the relevant sections (e.g., adjust focus, add emphasis, or refine topics). "
                "Ensure that the updated plan reflects the user's request while retaining the original structure and relevant parts."
            )
        else:
            user_prompt = (
                f"User Query: {user_query}\nContext: {context_info}\n"
                "Create a new structured learning plan based on the query and context."
            )

        # Log the user prompt for debugging
        logging.debug("User prompt: %s", user_prompt)

        plan_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        logging.debug("Plan response: %s", plan_response)

        # Clean and validate JSON response
        plan_content = plan_response.choices[0].message.content
        logging.info("Raw Generated Plan: %s", plan_content)

        # Strip trailing commas (fix invalid JSON)
        plan_content = plan_content.replace(",\n  ]", "\n  ]").replace(",\n}", "\n}")

        logging.info("Cleaned Plan: %s", plan_content)
        return plan_content
    except Exception as e:
        logging.error("Error in generate_plan: %s", str(e))
        return '{"error": "Plan generation failed due to an internal error."}'


def validate_and_clean_json(response_text):
    try:
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx == -1 or end_idx == -1:
            logging.error("JSON validation failed: No JSON object found in response.")
            return None

        json_text = response_text[start_idx:end_idx]
        plan = json.loads(json_text)

        # Ensure Modules are present and valid
        if "Modules" in plan:
            if not isinstance(plan["Modules"], list):
                logging.error("Modules validation failed: Modules must be a list.")
                return None
            for module in plan["Modules"]:
                if not all(key in module for key in ["module", "title", "description"]):
                    logging.error("Modules validation failed: Each module must have 'module', 'title', and 'description'.")
                    return None

        logging.info("Validated and cleaned JSON: %s", plan)
        return plan
    except json.JSONDecodeError as e:
        logging.error("Error decoding JSON: %s", str(e))
        logging.error("Raw JSON Response: %s", response_text)
        return None
    except Exception as e:
        logging.error("Unexpected error during JSON validation: %s", str(e))
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
    return {"access_token": access_token, "token_type": "bearer", "username": username}

@app.post("/refresh_token")
async def refresh_token(username: str = Depends(get_current_username)):
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
        "plan": current_plan,  # Retain the current plan by default
        "summary": current_summary,
        "response": "Unable to process the query. Please try again later."
    }

    try:
        # Step 1: Check for relevance in the knowledge base
        context_info = retrieve_information(user_query)

        if context_info:
            # Case 1: Relevant context found → Generate/Update learning plan
            learning_plan_json = generate_plan(user_query, context_info, current_plan)
            plan = validate_and_clean_json(learning_plan_json)

            if plan:
                summary = summarize_plan(plan)

                # Generate dynamic response based on user query and context using LLM
                response_prompt = (
                    "You are a helpful assistant specializing in creating and updating learning plans. Based on the following information, "
                    "generate a professional and relevant response summarizing the action taken:\n"
                    f"User Query: {user_query}\n"
                    f"Existing Plan: {current_plan}\n"
                    f"Generated Plan: {plan}\n"
                    "Ensure the response clearly communicates the action taken and its relevance to the user's input. Be specific."
                )
                response_generation = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": response_prompt},
                        {"role": "user", "content": user_query},
                    ],
                )
                response_text = response_generation.choices[0].message.content.strip()

                response_data.update({
                    "plan": plan,
                    "summary": summary,
                    "response": response_text
                })
                logging.info("Successfully updated the plan.")
            else:
                response_data["response"] = "Failed to generate or parse learning plan."
                logging.error("Plan generation failed. Raw response: %s", learning_plan_json)

        elif current_plan:
            # Case 2: Indirectly relevant query or context to refine the plan
            refine_prompt = (
                "You are an assistant specializing in refining learning plans. "
                "Determine if the user's query provides additional context to refine the current plan. "
                "If yes, generate an updated plan. If no, explain the relevance of the current plan or address the query conversationally.\n"
                f"User Query: {user_query}\n"
                f"Current Plan: {current_plan}"
            )
            refine_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": refine_prompt},
                    {"role": "user", "content": user_query},
                ],
            )

            refinement = refine_response.choices[0].message.content.strip()

            if refinement.startswith("{"):  # Check if the response includes a refined plan
                updated_plan = validate_and_clean_json(refinement)
                if updated_plan:
                    response_data.update({
                        "plan": updated_plan,
                        "summary": summarize_plan(updated_plan),
                        "response": f"Plan updated based on your input: '{user_query}'."
                    })
                else:
                    response_data["response"] = "Failed to refine the existing plan."
            else:
                # Retain the existing plan and provide a response
                response_data.update({
                    "response": refinement,
                    "plan": current_plan,  # Explicitly retain the current plan
                    "summary": current_summary  # Keep the existing summary
                })

        else:
            # Case 3: Irrelevant or general conversational input
            general_prompt = (
                "You are a helpful assistant specializing in general conversational responses. "
                "The user's input is unrelated to learning plans or the knowledge base. "
                "Respond politely and professionally, focusing on the role of a data science assistant."
            )
            general_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": general_prompt},
                    {"role": "user", "content": user_query},
                ],
            )
            response_data.update({
                "response": general_response.choices[0].message.content.strip(),
                "plan": current_plan,  # Explicitly retain the current plan
                "summary": current_summary  # Keep the existing summary
            })

    except Exception as e:
        logging.error("Error processing query: %s", str(e))
        response_data["response"] = "An error occurred while processing your query. Please try again later."

    logging.info("Final response data: %s", response_data)
    return response_data


@app.post("/save_plan")
async def save_plan(request: dict, username: str = Depends(get_current_username)):
    """
    Save the learning plan and module details to Snowflake.
    """
    plan = request.get("plan")
    summary = request.get("summary")

    if not plan or not summary:
        raise HTTPException(status_code=400, detail="Plan and summary are required.")

    connection = get_db_connection()
    try:
        cursor = connection.cursor()

        # Insert plan into the database
        plan_id = plan.get("PlanID", str(datetime.utcnow().timestamp()))
        key_topics = json.dumps(plan.get("KeyTopics", []))
        learning_outcomes = plan.get("LearningOutcomes", "N/A")  # Include learning outcomes

        cursor.execute(
            """
            INSERT INTO PLANS (PLAN_ID, USERNAME, SUMMARY, KEY_TOPICS, LEARNING_OUTCOMES)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (plan_id, username, summary, key_topics, learning_outcomes)
        )

        # Insert module details
        for module in plan.get("Modules", []):  # Rename "Weeks" to "Modules"
            module_id = module.get("ModuleID", str(datetime.utcnow().timestamp()))
            module_number = module.get("module", 0)
            title = module.get("title", "No Title")
            description = module.get("description", "No Description")
            cursor.execute(
                """
                INSERT INTO MODULES (MODULE_ID, PLAN_ID, MODULE, TITLE, DESCRIPTION)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (module_id, plan_id, module_number, title, description)
            )

        connection.commit()
        return {"message": "Plan saved successfully.", "plan_id": plan_id}
    except Exception as e:
        connection.rollback()
        logging.error("Error saving plan: %s", str(e))
        raise HTTPException(status_code=500, detail="An error occurred while saving the plan.")
    finally:
        cursor.close()
        connection.close()

@app.get("/get_plans")
def get_plans(username: str = Depends(get_current_username)):
    """
    Fetch all plans for the currently logged-in user.
    """
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        # Query only the plans associated with the current user
        cursor.execute(
            """
            SELECT plan_id, summary, key_topics, learning_outcomes
            FROM plans
            WHERE username = %s
            """,
            (username,),
        )
        plans = [
            {
                "plan_id": row[0],
                "summary": row[1],
                "key_topics": json.loads(row[2]) if row[2] else [],
                "learning_outcomes": row[3],  # Add this line to include learning outcomes
            }
            for row in cursor.fetchall()
        ]

        if not plans:
            logging.info(f"No plans found for user: {username}")
            return {"message": "No plans available"}

        return plans
    except Exception as e:
        logging.error(f"Error fetching plans: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch plans")
    finally:
        cursor.close()
        connection.close()


@app.get("/get_modules/{plan_id}")
def get_modules(plan_id: str):
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT MODULE_ID, PLAN_ID, MODULE, TITLE, DESCRIPTION FROM MODULES WHERE PLAN_ID = %s", 
            (plan_id,)
        )
        modules = cursor.fetchall()

        if not modules:
            logging.warning(f"No modules found for plan_id: {plan_id}")
            return {"message": f"No modules available for plan ID: {plan_id}"}

        return [
            {
                "module_id": row[0],
                "plan_id": row[1],
                "module": row[2],
                "title": row[3],
                "description": row[4],
            }
            for row in modules
        ]
    except Exception as e:
        logging.error(f"Error fetching modules for plan_id {plan_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch modules")
    finally:
        cursor.close()
        connection.close()


@app.get("/")
async def root():
    return {"message": "Welcome to the API!"}

@app.middleware("http")
async def log_requests(request, call_next):
    logging.info("Request URL: %s %s", request.method, request.url)
    response = await call_next(request)
    logging.info("Response status: %s", response.status_code)
    return response

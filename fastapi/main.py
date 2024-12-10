from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends
import os
from dotenv import load_dotenv
import json
from typing import Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import snowflake.connector
from queue import Queue
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from pydantic import BaseModel, ValidationError
from typing import List, Dict
from googleapiclient.discovery import build
import logging

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Define index parameters
INDEX_NAME = os.getenv("INDEX_NAME")
YOUTUBE_INDEX = os.getenv("YOUTUBE_INDEX")
DIMENSION = os.getenv("DIMENSION")
METRIC = os.getenv("METRIC")
CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER")
REGION = os.getenv("REGION")

# Check if the index exists
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    # Create the index if it doesn't exist
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD_PROVIDER, region=REGION),
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

if YOUTUBE_INDEX not in existing_indexes:
    # Create the YouTube index if it doesn't exist
    pc.create_index(
        name=YOUTUBE_INDEX,
        dimension=DIMENSION,  # Same dimension as embeddings
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD_PROVIDER, region=REGION),
    )

youtube_index = pc.Index(YOUTUBE_INDEX)

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
ACCESS_TOKEN_EXPIRE_MINUTES = 160

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

class Module(BaseModel):
    module: int
    title: str
    description: str

class Plan(BaseModel):
    Title: str
    Objective: str
    KeyTopics: List[str]
    Modules: List[Module]
    ExpectedOutcome: str

class SnowflakeConnectionPool:
    def __init__(self, config, maxsize=10):
        self.config = config
        self.pool = Queue(maxsize)
        for _ in range(maxsize):
            self.pool.put(self._create_connection())

    def _create_connection(self):
        return snowflake.connector.connect(
            user=self.config['user'],
            password=self.config['password'],
            account=self.config['account'],
            warehouse=self.config['warehouse'],
            database=self.config['database'],
            schema=self.config['schema'],
        )

    def get_connection(self):
        try:
            return self.pool.get(timeout=10)
        except Exception:
            raise HTTPException(status_code=500, detail="No available database connections in the pool.")

    def release_connection(self, connection):
        try:
            self.pool.put(connection, timeout=10)
        except Exception:
            connection.close()  # Close connection if pool is full

    def close_all_connections(self):
        while not self.pool.empty():
            connection = self.pool.get()
            connection.close()

pool = SnowflakeConnectionPool(SNOWFLAKE_CONFIG, maxsize=10)

def get_db_connection():
    connection = pool.get_connection()
    return connection


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
        with connection.cursor() as cursor:
            cursor.execute("SELECT USERNAME, PASSWORD, CREATED_AT FROM USERS WHERE USERNAME = %s", (username,))
            row = cursor.fetchone()
            return {"username": row[0], "password": row[1], "created_at": row[2]} if row else None
    finally:
        pool.release_connection(connection)  # Ensure the connection is released back to the pool

def create_user(username: str, password: str):
    hashed_password = get_password_hash(password)
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO USERS (USERNAME, PASSWORD, CREATED_AT) VALUES (%s, %s, %s)""",
                (username, hashed_password, datetime.utcnow())
            )
            connection.commit()
    finally:
        pool.release_connection(connection)  # Ensure the connection is released back to the pool

def inspect_index():
    try:
        results = index.describe_index_stats()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error inspecting index")


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
        query_embedding = get_embedding(user_query)
        results = index.query(vector=query_embedding, top_k=4, include_metadata=True)

        RELEVANCE_THRESHOLD = 0.8
        relevant_matches = [
            match for match in results["matches"] if match.get('score', 0) >= RELEVANCE_THRESHOLD
        ]

        if not relevant_matches:
            return None

        context_info = " ".join([
            f"Chunk ID: {match['metadata'].get('chunk_id', 'Unknown')}. Text: {match['metadata'].get('text', '').strip()}"
            for match in relevant_matches
        ])
        return context_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving information: {str(e)}")


def generate_plan(user_query, context_info, current_plan=None):
    try:
        if not context_info and not current_plan:
            return "No relevant information found in the Knowledge Base"

        system_prompt = (
            "You are a highly skilled assistant specializing in crafting structured and actionable MVP-style JSON learning plans that inspire users to learn and achieve their goals. "
            "Your plans should strike a balance between brevity and depth, providing clear and engaging content that motivates users to progress. "
            "Descriptions should be intellectual yet accessible, with relatable examples to demonstrate real-world applications and outcomes. "
            "Each learning plan should include:\n"
            "- A concise, compelling title that encapsulates the learning journey.\n"
            "- A clear and motivational objective outlining what the user will accomplish, written in an approachable and inspiring manner.\n"
            "- Key topics broken into digestible concepts to serve as milestones for learning progress.\n"
            "- Step-by-step modules with detailed yet exciting descriptions that highlight the value, relevance, and potential real-world impact of each step. Use relatable examples to illustrate key concepts and outcomes.\n"
            "- A measurable and inspiring expected outcome to give the user a sense of achievement and direction upon completing the plan.\n"
            "If updating an existing plan, carefully refine the relevant sections based on the user's query while preserving the overall structure and coherence of the plan. "
            "Strictly return a valid JSON output only. Do not include any introductory text, explanations, or comments. The response should consist solely of the JSON structure in this format:\n"
            "{\n"
            '  "Title": "An engaging and concise title for the learning plan",\n'
            '  "Objective": "A clear, inspiring summary of what the user will achieve",\n'
            '  "KeyTopics": ["Key topic 1", "Key topic 2", "Key topic 3"],\n'
            '  "Modules": [\n'
            '    {"module": 1, "title": "Catchy module title", "description": "A motivating, easy-to-read, and intellectually stimulating description that highlights the value of this module. Include a relatable example to showcase its practical application."},\n'
            '    {"module": 2, "title": "Another engaging module title", "description": "An actionable, detailed, and exciting description that keeps users interested and demonstrates the real-world impact of this module through examples."},\n'
            "    ...\n"
            "  ],\n"
            '  "ExpectedOutcome": "A clear, measurable, and inspiring outcome to help users stay focused and motivated."\n'
            "}\n"
            "Focus on providing content that is practical, motivating, and relatable while ensuring the JSON structure is complete, valid, and error-free. Tailor each plan to the user's query to make the learning journey personal and impactful. Ensure that the output contains only the JSON and nothing else."
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

        plan_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )

        plan_content = plan_response.choices[0].message.content

        plan_content = plan_content.replace(",\n  ]", "\n  ]").replace(",\n}", "\n}")  # Fix invalid JSON

        return plan_content
    except Exception as e:
        return '{"error": "Plan generation failed due to an internal error."}'

def validate_and_clean_json(response_text: str) -> Optional[dict]:
    try:
        response_json = json.loads(response_text)
        plan = Plan(**response_json)
        return plan.dict()
    except (ValidationError, json.JSONDecodeError) as e:
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

# ---  API Endpoints  ---
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
            else:
                response_data["response"] = "Failed to generate or parse learning plan."

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
        response_data["response"] = "An error occurred while processing your query. Please try again later."

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
        with connection.cursor() as cursor:
            # Insert plan into the database
            plan_id = plan.get("PlanID", str(datetime.utcnow().timestamp()))
            title = plan.get("Title", "Untitled Plan")
            key_topics = json.dumps(plan.get("KeyTopics", []))
            learning_outcomes = plan.get("ExpectedOutcome", "N/A")

            cursor.execute(
                """
                INSERT INTO PLANS (PLAN_ID, USERNAME, TITLE, SUMMARY, KEY_TOPICS, LEARNING_OUTCOMES)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (plan_id, username, title, summary, key_topics, learning_outcomes)
            )

            # Insert module details
            for module in plan.get("Modules", []):
                module_id = module.get("ModuleID", str(datetime.utcnow().timestamp()))
                module_number = module.get("module", 0)
                module_title = module.get("title", "No Title")
                description = module.get("description", "No Description")
                cursor.execute(
                    """
                    INSERT INTO MODULES (MODULE_ID, PLAN_ID, MODULE, TITLE, DESCRIPTION)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (module_id, plan_id, module_number, module_title, description)
                )

            connection.commit()
            return {"message": "Plan saved successfully.", "plan_id": plan_id}
    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail="An error occurred while saving the plan.")
    finally:
        connection.close()

@app.get("/get_plans")
def get_plans(username: str = Depends(get_current_username), page: int = 1, size: int = 10):
    """
    Fetch paginated plans for the currently logged-in user.
    """
    connection = get_db_connection()
    try:
        offset = (page - 1) * size
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT plan_id, title, summary, key_topics, learning_outcomes
                FROM plans
                WHERE username = %s
                LIMIT %s OFFSET %s
                """,
                (username, size, offset),
            )
            plans = [
                {
                    "plan_id": row[0],
                    "title": row[1],
                    "summary": row[2],
                    "key_topics": json.loads(row[3]) if row[3] else [],
                    "learning_outcomes": row[4],
                }
                for row in cursor.fetchall()
            ]

            if not plans:
                return {"message": "No plans available"}

            return plans
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch plans")
    finally:
        connection.close()

@app.get("/get_modules/{plan_id}")
def get_modules(plan_id: str, page: int = 1, size: int = 10):
    connection = get_db_connection()
    try:
        offset = (page - 1) * size
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT MODULE_ID, PLAN_ID, MODULE, TITLE, DESCRIPTION
                FROM MODULES
                WHERE PLAN_ID = %s
                LIMIT %s OFFSET %s
                """,
                (plan_id, size, offset)
            )
            modules = cursor.fetchall()

            if not modules:
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
        raise HTTPException(status_code=500, detail="Failed to fetch modules")
    finally:
        connection.close()

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrieve_detailed_explanation(title: str, description: str, top_k: int = 50) -> str:
    """
    Retrieve detailed explanations for the given title and description by querying Pinecone dynamically.
    """
    try:
        logging.info("Starting retrieval of detailed explanations.")
        logging.info(f"Title: {title}")
        logging.info(f"Description: {description}")
        logging.info(f"Top_k parameter: {top_k}")

        # Dynamically create a query based on the title and description
        query_text = f"Find detailed explanations relevant to the following context:\nTitle: {title}\nDescription: {description}"
        logging.info("Generated query text for embedding.")
        
        query_embedding = get_embedding(query_text)
        if not query_embedding:
            logging.error("Failed to generate embedding. Query embedding is empty.")
            raise ValueError("Query embedding is empty.")
        logging.info("Successfully generated query embedding.")

        # Query Pinecone
        logging.info("Querying Pinecone with generated embedding.")
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Process results
        if results and results.get("matches"):
            logging.info(f"Received {len(results['matches'])} matches from Pinecone.")
            chunks = []
            for match in results["matches"]:
                score = match["score"]
                chunk_id = match["metadata"].get("chunk_id", "N/A")
                text = match["metadata"].get("text", "").strip()

                logging.info(f"Processing match with score: {score}, Chunk ID: {chunk_id}")
                if score >= 0.85:  # Threshold for relevance
                    if text:
                        logging.info(f"Adding chunk with score {score} and Chunk ID {chunk_id}.")
                        chunks.append(f"Chunk ID: {chunk_id}. {text}")

            if chunks:
                logging.info("Successfully processed and filtered relevant chunks.")
                return "\n\n".join(chunks)
            else:
                logging.warning("No relevant chunks found after filtering.")
                return "No relevant explanation found."

        logging.warning("No matches received from Pinecone.")
        return "No relevant explanation found."

    except Exception as e:
        logging.error(f"Error occurred during explanation retrieval: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving explanation: {str(e)}")


@app.get("/get_module_details/{module_id}")
def get_module_details(module_id: str):
    """
    Fetch module details and generate a structured, article-like detailed explanation dynamically.
    """
    connection = get_db_connection()
    try:
        # Step 1: Retrieve module details from the database
        cursor = connection.cursor()
        cursor.execute(
            "SELECT MODULE_ID, MODULE, TITLE, DESCRIPTION FROM MODULES WHERE MODULE_ID = %s",
            (module_id,)
        )
        module = cursor.fetchone()
        if not module:
            return {"message": f"No details found for module ID: {module_id}"}

        title, description = module[2], module[3]  # Extract title and description

        # Step 2: Retrieve raw explanation from Pinecone
        raw_explanation = retrieve_detailed_explanation(title, description, top_k=30)

        # Step 3: Use OpenAI to format the explanation into an article
        try:
            
            system_prompt = f"""
                You are an AI assistant tasked with synthesizing a detailed and professional article based on the provided explanation text. 
                The explanations are curated to include highly relevant information (relevance score > 0.85) and represent key insights about the topic.

                Context:
                - Module Title: {title}
                - Module Description: {description}

                **Purpose**:
                Your goal is to generate a comprehensive and insightful article that educates readers on the topic. 
                The article should be structured naturally and cohesively, presenting the information in a logical and engaging manner.

                **Content Guidelines**:
                1. Use all the provided explanation text, ensuring the original meaning and context are retained.
                2. Highlight key points and elaborate where necessary to ensure clarity and understanding.
                3. Avoid introducing external information or data outside of what is provided.

                **Style**:
                - Write in a professional and engaging tone suitable for an educational audience.
                - Organize the content in a manner that flows logically, allowing readers to follow the topic easily.
                - Use paragraphs, headings, or bullet points where necessary to improve readability.
                - Ensure smooth transitions between ideas and sections for coherence.

                **Additional Instructions**:
                - Focus on clarity, coherence, and logical progression of ideas.
                - Avoid unnecessary repetition and ensure the article stays on topic.

                Provided Explanation Text:
                {raw_explanation}
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_prompt}],
                max_tokens=1000,
                temperature=0
            )
            formatted_explanation = response.choices[0].message.content.strip()

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error formatting explanation: {str(e)}")

        # Step 4: Return module details with formatted explanation
        return {
            "module_id": module[0],
            "module": module[1],
            "title": title,
            "description": description,
            "detailed_explanation": formatted_explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch module details")
    finally:
        cursor.close()
        connection.close()



# --------------------------- YOUTUBE VIDEO LOGIC -----------------------------------------------

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Define YouTubeVideoResponse model
class YouTubeVideoResponse(BaseModel):
    video_url: Optional[str]
    relevance_score: Optional[float]

# Function for summarizing text
def summarize_text(text: str, max_length: int = 100) -> str:
    try:
        logger.info("Summarizing input text")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or another model if you prefer
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize this text in {max_length} characters: {text}"}
            ],
            max_tokens=max_length,
        )

        # Access the content of the response correctly
        summary = response.choices[0].message.content.strip()
        logger.info(f"Summarized text: {summary}")
        return summary

    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")


# Function for chunking text
def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Fetching relevant YouTube videos
def fetch_youtube_videos(query: str, max_results: int = 3) -> List[dict]:
    try:
        logger.info("Fetching YouTube videos for query: %s", query)
        search_response = youtube.search().list(
            q=query, part="snippet", type="video", maxResults=max_results
        ).execute()
        videos = [
            {
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
            }
            for item in search_response.get("items", [])
        ]
        return videos
    except Exception as e:
        logger.error(f"Error fetching YouTube videos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching YouTube videos: {str(e)}")

# Fetch video transcript

def fetch_video_transcript(video_id: str) -> Optional[str]:
    try:
        logger.info("Fetching transcript for video ID: %s", video_id)
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item["text"] for item in transcript_data])
    except Exception as e:
        # Check if the error is due to transcripts being disabled
        if "TranscriptsDisabled" in str(e):
            logger.warning(f"Transcript disabled for video ID: {video_id}")
            return None
        logger.error(f"Error fetching transcript for video {video_id}: {str(e)}")
        return None


# Generate embeddings for text
def generate_embedding(text: str) -> List[float]:
    try:
        logger.info("Generating embeddings for text")
        response = client.embeddings.create(model="text-embedding-ada-002", input=text)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

# Upsert video data to Pinecone
def upsert_to_pinecone(video_id: str, title: str, description: str, transcript_chunks: List[str]) -> None:
    try:
        logger.info("Upserting data to Pinecone for video ID: %s", video_id)
        for idx, chunk in enumerate(transcript_chunks):
            embedding = generate_embedding(chunk)
            youtube_index.upsert([
                {
                    "id": f"{video_id}_chunk_{idx}",
                    "values": embedding,
                    "metadata": {
                        "title": title,
                        "description": description,
                        "transcript_chunk": chunk,
                        "video_id": video_id,
                    },
                }
            ])
    except Exception as e:
        logger.error(f"Error upserting to Pinecone for video {video_id}: {str(e)}")

@app.get("/get_relevant_youtube_video/{module_id}", response_model=YouTubeVideoResponse)
async def get_relevant_youtube_video(module_id: str):
    try:
        logger.info("Fetching relevant YouTube video for module ID: %s", module_id)
        # Simulated database module details

        connection = get_db_connection()
        with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT TITLE, DESCRIPTION FROM MODULES WHERE MODULE_ID = %s",
                    (module_id,)
                )
                module = cursor.fetchone()
                if not module:
                    raise HTTPException(status_code=404, detail=f"No module found with ID {module_id}")

                title, description = module

        module_title = title
        module_description = description
        module_detailed_explanation = retrieve_detailed_explanation(title, description)  # Use existing function to get explanation
        combined_text = f"{module_title} {module_description} {module_detailed_explanation}"

        # Step 1: Summarize input text
        summarized_text = summarize_text(combined_text)

        # Step 2: Fetch YouTube videos
        videos = fetch_youtube_videos(summarized_text)
        if not videos:
            logger.warning("No relevant YouTube videos found")
            raise HTTPException(status_code=404, detail="No relevant YouTube videos found.")

        # Step 3: Process transcripts and upsert to Pinecone
        for video in videos:
            transcript_text = fetch_video_transcript(video["video_id"])
            if transcript_text:
                transcript_chunks = chunk_text(transcript_text)
                upsert_to_pinecone(video["video_id"], video["title"], video["description"], transcript_chunks)

        # Step 4: Generate embedding for the module content
        module_embedding = generate_embedding(combined_text)

        # Step 5: Query Pinecone for the most relevant video
        search_results = youtube_index.query(vector=module_embedding, top_k=1, include_metadata=True)
        if not search_results["matches"]:
            logger.warning("No matches found in Pinecone")
            return {"video_url": None, "relevance_score": None}

        # Extract the most relevant match
        best_match = search_results["matches"][0]
        video_url = f"https://www.youtube.com/watch?v={best_match['metadata']['video_id']}"
        relevance_score = best_match["score"]

        return {"video_url": video_url, "relevance_score": relevance_score}

    except Exception as e:
        logger.error(f"Error in /get_relevant_youtube_video endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Welcome to the API!"}

@app.middleware("http")
async def log_requests(request, call_next):
    response = await call_next(request)
    return response

@app.on_event("shutdown")
async def close_connection_pool():
    pool.close_all_connections()

import logging
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import HTTPException
import tiktoken
from typing import List, Optional
from snowflake.connector import connect
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from config import (
    SNOWFLAKE_CONFIG,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    YOUTUBE_API_KEY,
    INDEX_NAME,
    YOUTUBE_INDEX,
    DIMENSION,
    METRIC,
    CLOUD_PROVIDER,
    REGION,
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    client,
    pc,
    youtube,
    index,
    youtube_index,
)
from queue import Queue
import snowflake.connector
from fastapi import Depends

# OAuth2PasswordBearer for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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

class YouTubeVideoResponse(BaseModel):
    video_url: Optional[str]
    relevance_score: Optional[float]

# -------- Utility Functions --------

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

async def get_current_username(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

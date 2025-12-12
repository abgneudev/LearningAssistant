# AIVY - AI Learning Assistant: Comprehensive Codebase Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Core Components](#core-components)
6. [Data Flow](#data-flow)
7. [API Endpoints](#api-endpoints)
8. [Database Schema](#database-schema)
9. [Setup and Deployment](#setup-and-deployment)
10. [Development Workflow](#development-workflow)
11. [Key Features Explained](#key-features-explained)

---

## Project Overview

**AIVY** (AI-Powered Learning Assistant) is a next-generation educational platform that integrates various learning tools into a unified, personalized experience. The platform combines text content, videos, flashcards, and quizzes to provide comprehensive learning paths.

### Core Value Propositions:
- **Unified Learning Tools**: All learning content (text, videos, quizzes) accessible in one place
- **Personalized Assistance**: AI-tailored learning content and assessments
- **Advanced Search**: RAG (Retrieval-Augmented Generation) for context-aware results
- **Engaging UI**: Intuitive and interactive user experience

---

## Architecture

The application follows a microservices architecture with three main components:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Streamlit  │─────▶│   FastAPI   │─────▶│  Snowflake  │
│  (Frontend) │      │  (Backend)  │      │  (Database) │
└─────────────┘      └─────────────┘      └─────────────┘
                            │
                            ├─────▶ OpenAI API (GPT-3.5)
                            ├─────▶ Pinecone (Vector DB)
                            ├─────▶ YouTube API
                            └─────▶ ArXiv API
       
┌─────────────┐
│   Airflow   │─────▶ Web Scraping (GeeksforGeeks)
│ (ETL/DAGs)  │─────▶ Data Processing & Embeddings
└─────────────┘─────▶ Pinecone Upload
```

### Key External Services:
- **Snowflake**: Stores user data, learning plans, and modules
- **Pinecone**: Vector database for similarity search (text, images, videos)
- **OpenAI**: GPT-3.5 for content generation and embeddings
- **YouTube API**: Fetches relevant educational videos
- **ArXiv API**: Retrieves academic papers

---

## Technology Stack

### Backend (FastAPI)
- **Framework**: FastAPI 0.115.5
- **Python**: >=3.10, <3.14
- **Key Dependencies**:
  - `openai` (1.57.2) - AI content generation
  - `pinecone` (5.4.2) - Vector database
  - `snowflake-connector-python` (3.12.4) - Database connectivity
  - `youtube-transcript-api` (0.6.3) - Video transcript extraction
  - `tiktoken` (0.8.0) - Token counting for embeddings
  - `passlib` & `python-jose` - Authentication & JWT

### Frontend (Streamlit)
- **Framework**: Streamlit 1.40.2
- **Python**: ^3.12
- **Key Dependencies**:
  - `requests` (2.32.3) - API communication
  - `streamlit-extras` (0.5.0) - Enhanced UI components

### Data Pipeline (Airflow)
- **Framework**: Apache Airflow 2.10.3
- **Python**: Compatible with Airflow
- **Key Dependencies**:
  - `beautifulsoup4` & `bs4` - Web scraping
  - `transformers` (4.47.0) - CLIP model for image embeddings
  - `torch` (2.5.1) - Deep learning framework
  - `pillow` (11.0.0) - Image processing

---

## Project Structure

```
LearningAssistant/
├── Airflow/                          # Data extraction and processing
│   ├── dags/
│   │   ├── GFG_Data_Extraction_DAG.py    # Main DAG definition
│   │   └── extraction_files/
│   │       ├── extraction.py             # Text extraction logic
│   │       ├── image_extraction.py       # Image processing
│   │       └── links.py                  # Web scraping utilities
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── entrypoint.sh
│   └── requirements.txt
│
├── fastapi/                          # Backend API
│   ├── main.py                       # FastAPI app & endpoints
│   ├── config.py                     # Configuration & initialization
│   ├── syllabus.py                   # Learning plan generation
│   ├── lessons.py                    # Lesson content retrieval
│   ├── utils.py                      # Helper functions & models
│   ├── pyproject.toml               # Poetry dependencies
│   └── Dockerfile
│
├── streamlit/                        # Frontend UI
│   ├── app.py                        # Main entry point
│   ├── ui/
│   │   ├── planner.py               # Learning plan interface
│   │   ├── lesson.py                # Lesson display
│   │   ├── plans.py                 # Saved plans view
│   │   └── quiz.py                  # Quiz interface
│   ├── pyproject.toml               # Poetry dependencies
│   └── Dockerfile
│
├── diagram/                          # Architecture diagrams
├── docker-compose.yml               # Multi-service orchestration
└── README.md                        # Project documentation
```

---

## Core Components

### 1. Airflow (Data Pipeline)

**Purpose**: Extract, process, and index educational content from web sources.

**Main DAG (`GFG_Data_Extraction_DAG.py`)**:
- **Schedule**: Runs daily (`timedelta(days=1)`)
- **Tasks**:
  1. `scrape_links` - Scrapes GeeksforGeeks for tech-related links
  2. `fetch_new_links` - Retrieves unprocessed links from Snowflake
  3. `process_new_links` - Processes content and generates embeddings

**Processing Pipeline**:
```python
# Text Processing
1. Scrape webpage content → Clean text → Chunk text
2. Generate Ada embeddings (OpenAI)
3. Upload to Pinecone text index

# Image Processing (CLIP Model)
1. Extract .png images → Filter by relevance
2. Generate CLIP embeddings (openai/clip-vit-base-patch32)
3. Upload to Pinecone image index
```

**Key Functions** (from `extraction.py`):
- `scrape_webpage()` - Extracts content and images from URLs
- `chunk_text()` - Splits text into manageable chunks
- `get_ada_embedding()` - Generates text embeddings
- `upload_to_pinecone()` - Uploads vectors to Pinecone

### 2. FastAPI (Backend API)

**Purpose**: Provides RESTful API for user management, learning plan generation, and content retrieval.

**Architecture Patterns**:
- **Connection Pooling**: `SnowflakeConnectionPool` manages database connections
- **Caching**: LRU cache for embeddings (`@lru_cache`)
- **Dependency Injection**: FastAPI's `Depends()` for shared resources
- **JWT Authentication**: OAuth2 with bearer tokens

**Key Modules**:

#### `config.py`
Initializes all external services:
```python
- OpenAI client (GPT-3.5)
- Pinecone indexes (text, YouTube, images)
- YouTube API client
- Snowflake configuration
- JWT settings
```

#### `main.py` 
Contains all API endpoints (detailed in [API Endpoints](#api-endpoints) section).

#### `syllabus.py`
Handles learning plan generation:
- `retrieve_information()` - Queries Pinecone for relevant content
- `generate_plan()` - Uses GPT-3.5 to create structured learning plans
- `validate_and_clean_json()` - Parses and validates plan JSON
- `get_embedding()` - Generates embeddings with caching

#### `lessons.py`
Manages lesson content:
- `retrieve_detailed_explanation()` - Fetches relevant chunks from Pinecone
- `fetch_youtube_videos()` - Searches YouTube for educational videos
- `fetch_video_transcript()` - Retrieves video transcripts
- `upsert_to_pinecone()` - Indexes YouTube content
- `summarize_text()` - Uses GPT to condense content

#### `utils.py`
Utility functions and data models:
- Password hashing (`get_password_hash`, `verify_password`)
- JWT token management (`create_access_token`, `decode_token`)
- Database operations (`get_db_connection`, `get_user`, `create_user`)
- Pydantic models (`Module`, `Plan`, `QuizGeneration`, `FlashcardGeneration`)
- Connection pool implementation (`SnowflakeConnectionPool`)

### 3. Streamlit (Frontend)

**Purpose**: User-friendly interface for learners.

**Main Application (`app.py`)**:
- User authentication (login/signup)
- Session management
- Page navigation (sidebar)

**UI Pages**:

#### `planner.py`
- Chat-based interface for learning plan generation
- Displays generated plans with modules
- Supports plan saving and refinement
- Uses conversational AI for natural interaction

#### `plans.py`
- Lists all saved learning plans
- Displays plan metadata (title, summary, topics)
- Module navigation

#### `lesson.py`
- Displays module details and detailed explanations
- Shows relevant YouTube videos
- Presents academic papers from ArXiv
- Formatted article-style content

#### `quiz.py`
- Generates quizzes based on module content
- Multiple-choice questions
- Instant feedback and scoring

---

## Data Flow

### User Learning Journey:

```
1. User Login (Streamlit)
   ↓
2. Create Learning Plan (Planner)
   - User query → FastAPI /query
   - Retrieves context from Pinecone
   - GPT generates structured plan
   ↓
3. Save Plan (FastAPI /save_plan)
   - Stores in Snowflake (PLANS & MODULES tables)
   ↓
4. View Saved Plans (Plans page)
   - Fetches from FastAPI /get_plans
   ↓
5. Select Module (Lesson page)
   - GET /get_module_details/{module_id}
   - Retrieves detailed explanation from Pinecone
   - GPT formats as article
   ↓
6. Enhance Learning
   - YouTube video: /get_relevant_youtube_video/{module_id}
   - Academic papers: /get_relevant_arxiv_paper/{module_id}
   ↓
7. Test Knowledge
   - Flashcards: /generate_flashcards/{module_id}
   - Quiz: /generate_quiz/{module_id}
```

### Content Indexing Flow (Airflow):

```
1. Scrape Links from GeeksforGeeks
   ↓
2. Store in Snowflake (STATUS='NEW')
   ↓
3. Fetch unprocessed links
   ↓
4. Process each link:
   - Extract text & images
   - Generate embeddings
   - Upload to Pinecone
   ↓
5. Mark as processed (STATUS='PROCESSED')
```

---

## API Endpoints

### Authentication

#### POST `/signup`
Create a new user account.
- **Parameters**: `username`, `password` (query params)
- **Response**: `{"message": "User created successfully"}`

#### POST `/login`
Authenticate user and receive JWT token.
- **Parameters**: `username`, `password` (query params)
- **Response**: `{"access_token": "...", "token_type": "bearer", "username": "..."}`

#### POST `/refresh_token`
Refresh JWT access token.
- **Authentication**: Required
- **Response**: New access token

### Learning Plans

#### POST `/query`
Generate or update learning plan based on user query.
- **Request Body**:
  ```json
  {
    "user_query": "string",
    "current_plan": "object (optional)",
    "current_summary": "string (optional)"
  }
  ```
- **Response**:
  ```json
  {
    "plan": {
      "Title": "string",
      "Objective": "string",
      "KeyTopics": ["string"],
      "Modules": [{"module": 1, "title": "string", "description": "string"}],
      "ExpectedOutcome": "string"
    },
    "summary": "string",
    "response": "string"
  }
  ```

#### POST `/save_plan`
Save learning plan to database.
- **Authentication**: Required
- **Request Body**: `{"plan": {...}, "summary": "string"}`
- **Response**: `{"message": "Plan saved successfully.", "plan_id": "string"}`

#### GET `/get_plans`
Retrieve all plans for the logged-in user.
- **Authentication**: Required
- **Query Parameters**: `page` (default: 1), `size` (default: 0 for all)
- **Response**: Array of plan objects

#### GET `/get_modules/{plan_id}`
Fetch modules for a specific plan.
- **Parameters**: `plan_id`, `page`, `size`
- **Response**: Array of module objects

### Module Content

#### GET `/get_module_details/{module_id}`
Retrieve detailed module content.
- **Response**:
  ```json
  {
    "module_id": "string",
    "module": "int",
    "title": "string",
    "description": "string",
    "detailed_explanation": "string (formatted article)"
  }
  ```

#### GET `/get_relevant_youtube_video/{module_id}`
Find most relevant YouTube video for module.
- **Response**:
  ```json
  {
    "video_url": "string",
    "relevance_score": "float",
    "transcript": "string"
  }
  ```

#### GET `/get_relevant_arxiv_paper/{module_id}`
Fetch relevant academic papers from ArXiv.
- **Response**: Array of paper objects with title, authors, summary, links

### Assessment

#### GET `/generate_flashcards/{module_id}`
Generate flashcards for module.
- **Response**:
  ```json
  {
    "flashcards": [
      {"question": "string", "answer": "string"}
    ]
  }
  ```

#### GET `/generate_quiz/{module_id}`
Generate multiple-choice quiz.
- **Response**:
  ```json
  {
    "quiz": [
      {
        "question": "string",
        "options": ["A", "B", "C", "D"],
        "correct_answer": "string"
      }
    ]
  }
  ```

---

## Database Schema

### Snowflake Tables

#### USERS Table
```sql
- USERNAME (VARCHAR, PRIMARY KEY)
- PASSWORD (VARCHAR) - bcrypt hashed
- CREATED_AT (TIMESTAMP)
```

#### PLANS Table
```sql
- PLAN_ID (VARCHAR, PRIMARY KEY)
- USERNAME (VARCHAR, FOREIGN KEY)
- TITLE (VARCHAR)
- SUMMARY (TEXT)
- KEY_TOPICS (JSON) - Array of strings
- LEARNING_OUTCOMES (TEXT)
- CREATED_AT (TIMESTAMP)
```

#### MODULES Table
```sql
- MODULE_ID (VARCHAR, PRIMARY KEY)
- PLAN_ID (VARCHAR, FOREIGN KEY)
- MODULE (INTEGER) - Module number
- TITLE (VARCHAR)
- DESCRIPTION (TEXT)
- CREATED_AT (TIMESTAMP)
```

#### LINKS Table (for Airflow)
```sql
- ARTICLE_ID (VARCHAR, PRIMARY KEY)
- URL (VARCHAR)
- TITLE (VARCHAR)
- STATUS (VARCHAR) - 'NEW' or 'PROCESSED'
- CREATED_AT (TIMESTAMP)
```

### Pinecone Indexes

#### Text Index (e.g., "learning-content")
```python
Dimension: 1536 (Ada-002 embeddings)
Metadata:
  - article_id
  - chunk_id
  - title
  - type: "text"
  - text: actual content
  - url
```

#### YouTube Index
```python
Dimension: 1536
Metadata:
  - video_id
  - title
  - description
  - chunk_text
```

#### Image Index
```python
Dimension: 512 (CLIP embeddings)
Metadata:
  - article_id
  - image_id
  - title
  - type: "image"
  - url
```

---

## Setup and Deployment

### Prerequisites
- Python 3.10+ (FastAPI), 3.12+ (Streamlit)
- Docker & Docker Compose
- Poetry (dependency management)

### Environment Variables

Create a `.env` file with:
```bash
# FastAPI
DEPLOY_URL=http://localhost:8000

# Snowflake
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema

# JWT
SECRET_KEY=your_secret_key

# OpenAI
OPENAI_API_KEY=your_openai_key

# Pinecone
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment
CLOUD_PROVIDER=aws
REGION=us-east-1
INDEX_NAME=your_text_index
YOUTUBE_INDEX=your_youtube_index
IMG_INDEX_NAME=your_image_index
DIMENSION=1536
IMAGE_DIMENSIONS=512
METRIC=cosine

# YouTube
YOUTUBE_API_KEY=your_youtube_key
```

### Local Development

#### FastAPI:
```bash
cd fastapi
python -m venv venv_fastapi
source venv_fastapi/bin/activate  # Windows: venv_fastapi\Scripts\activate
poetry install
uvicorn main:app --reload
```
Access: http://localhost:8000/docs

#### Streamlit:
```bash
cd streamlit
python -m venv venv_streamlit
source venv_streamlit/bin/activate
poetry install
streamlit run app.py
```
Access: http://localhost:8501

#### Airflow:
```bash
cd Airflow
docker-compose up --build
```
Access: http://localhost:8080

### Docker Deployment

Run all services:
```bash
docker-compose up --build
```

Services:
- FastAPI: http://localhost:8000
- Streamlit: http://localhost:8501

---

## Development Workflow

### Adding a New Feature

1. **Backend (FastAPI)**:
   - Define endpoint in `main.py`
   - Add helper functions in appropriate module (`syllabus.py`, `lessons.py`, etc.)
   - Update Pydantic models in `utils.py` if needed
   - Test with FastAPI docs (/docs)

2. **Frontend (Streamlit)**:
   - Create or modify UI file in `streamlit/ui/`
   - Add API calls using `requests`
   - Manage state with `st.session_state`
   - Add navigation in `app.py` if new page

3. **Data Pipeline (Airflow)**:
   - Modify DAG in `Airflow/dags/`
   - Add processing logic in `extraction_files/`
   - Test DAG in Airflow UI

### Testing

**Manual Testing**:
- FastAPI: Use Swagger UI at `/docs`
- Streamlit: Navigate through UI flows
- Integration: Test end-to-end user journeys

**Postman Testing**:
The team has extensively tested APIs with Postman (mentioned in README).

### Logging

All components use Python's `logging` module:
- **FastAPI**: Logs requests, errors, and processing steps
- **Airflow**: DAG execution logs
- **Streamlit**: UI interactions and errors

View logs:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Your message")
```

---

## Key Features Explained

### 1. RAG (Retrieval-Augmented Generation)

**How it works**:
1. User query → Generate embedding
2. Query Pinecone for similar content (top_k=30, score > 0.85)
3. Retrieved chunks → Feed to GPT-3.5
4. GPT generates contextual response

**Code Location**: `syllabus.py` (`retrieve_information()`)

### 2. Learning Plan Generation

**Process**:
1. Analyze user query for topics
2. Retrieve relevant context from Pinecone
3. GPT-3.5 generates structured JSON plan
4. Validate and parse JSON
5. Generate summary

**Code Location**: `syllabus.py` (`generate_plan()`)

### 3. Dynamic Content Formatting

**Detailed Explanations**:
- Retrieves chunks from Pinecone (relevance > 0.85)
- GPT-3.5 formats as professional article
- Includes code snippets and formulas

**Code Location**: `main.py` (`get_module_details()`)

### 4. YouTube Video Integration

**Pipeline**:
1. Summarize module content
2. Search YouTube API
3. Fetch transcripts for videos
4. Chunk transcripts and embed
5. Query Pinecone for most relevant video

**Code Location**: `lessons.py` & `main.py` (`get_relevant_youtube_video()`)

### 5. Connection Pooling

**Snowflake Connection Management**:
- Pre-initialized pool of connections
- Connection validation before use
- Automatic recreation of failed connections
- Periodic keep-alive task

**Code Location**: `utils.py` (`SnowflakeConnectionPool`)

### 6. Caching Strategy

**LRU Cache**:
- Embeddings cached to reduce OpenAI API calls
- Semantic similarity check for cache hits
- Hash-based cache keys

**Code Location**: `syllabus.py` (`get_embedding()`, `retrieve_cached_chunks()`)

---

## Common Patterns

### Error Handling
```python
try:
    # Operation
except Exception as e:
    logging.error(f"Error: {e}")
    raise HTTPException(status_code=500, detail="Error message")
```

### Authentication Flow
```python
@app.get("/endpoint")
async def endpoint(username: str = Depends(get_current_username)):
    # Authenticated operation
```

### State Management (Streamlit)
```python
if "key" not in st.session_state:
    st.session_state["key"] = default_value
```

### Pinecone Querying
```python
results = index.query(
    vector=embedding,
    top_k=30,
    include_metadata=True
)
```

---

## Troubleshooting

### Common Issues

1. **Snowflake Connection Errors**:
   - Check `.env` credentials
   - Verify warehouse is running
   - Check connection pool logs

2. **Pinecone Index Not Found**:
   - Ensure indexes exist (`config.py` creates them)
   - Verify API key and environment

3. **OpenAI Rate Limits**:
   - Implement exponential backoff
   - Use caching to reduce calls

4. **Docker Build Failures**:
   - Clear Docker cache: `docker-compose build --no-cache`
   - Check dependency versions

---

## Future Enhancements

Based on commented code and structure:
- Image summarization with OpenAI (currently commented)
- Enhanced CLIP-based image search (CLIP = Contrastive Language-Image Pre-training)
- More sophisticated caching
- Advanced quiz types
- Progress tracking

---

## Team & Contributions

| Name    | Responsibilities                                           | Contribution |
|---------|-----------------------------------------------------------|--------------|
| Abhinav | UX, Frontend, Backend, Dockerization, CI/CD, DB Design    | 55%          |
| Nishita | Web scraping, Airflow, Pinecone, Documentation            | 25%          |
| Dhir    | DB config, Quiz/Flashcard/Lesson generation              | 20%          |

---

## Additional Resources

- **Documentation**: [Codelabs](https://codelabs-preview.appspot.com/?file_id=1qj_hNrPvLQEQt1r2RgSReLwTkpFyHrK22inGdcHLsY8#0)
- **Demo Video**: [Google Drive](https://drive.google.com/drive/folders/1sBLrejhuyzoQXzyt3lf0hlBzZ16HBY_4?usp=drive_link)
- **Deployed FastAPI**: http://3.14.131.176:8000/docs
- **Deployed Streamlit**: http://3.14.131.176:8501/

---

## Quick Reference

### Key Files to Start With:
1. `README.md` - Project overview
2. `fastapi/main.py` - All API endpoints
3. `streamlit/app.py` - Frontend entry point
4. `Airflow/dags/GFG_Data_Extraction_DAG.py` - Data pipeline

### Development Commands:
```bash
# FastAPI
uvicorn main:app --reload

# Streamlit
streamlit run app.py

# Airflow
docker-compose up

# Install dependencies
poetry install

# Docker rebuild
docker-compose up --build
```

---

*This guide is a living document. Update as the codebase evolves.*

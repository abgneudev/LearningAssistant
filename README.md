# AIVY - AI Powered Learning Assistant

## **Overview**


---

## **Key Features**
1. **Dynamic Lesson Plans**:
   - Lessons organized into modules with titles, descriptions, and detailed explanations.
2. **Quiz and Flashcard Generation**:
   - Auto-generated quizzes and flashcards based on lesson content.
3. **RAG Functionality**:
   - Retrieves relevant YouTube videos, images, and text for user queries.
4. **Logging and Testing**:
   - Comprehensive logging in Streamlit and Airflow for error tracking.
   - APIs tested extensively with **Postman**.
5. **Scalable Design**:
   - Separate environments for Streamlit and FastAPI to ensure modularity.

---

## **Project Structure**
The project is divided into three main components:

### **1. Airflow**
- Handles data extraction pipelines for lesson content, images, and links.
- Key files:
  - `extraction.py`: Extracts lesson content.
  - `image_extraction.py`: Processes images.
  - `links.py`: Manages and fetches additional learning resources.
  - `GFG_Data_Extraction_DAG.py`: Manages and schedules DAG workflows.

### **2. FastAPI**
- Backend API for:
  - Fetching lessons, images, and YouTube video URLs.
  - Generating quizzes and flashcards.
- Lightweight and scalable.

### **3. Streamlit**
- User-friendly frontend for:
  - Navigating and viewing lesson plans.
  - Interacting with quizzes and results.
- Files include:
  - `lesson.py`: Displays detailed lesson content.
  - `quiz.py`: Manages quiz interface and results.
  - `planner.py`: Allows navigation across modules and plans.

---

### **Folder Structure**
```
├── Airflow
│   ├── dags
│   │   ├── extraction_files
│   │   │   ├── extraction.py
│   │   │   ├── image_extraction.py
│   │   │   ├── links.py
│   │   │   ├── testfiles/
│   │   ├── GFG_Data_Extraction_DAG.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── entrypoint.sh
│   ├── requirements.txt
├── fastapi
│   ├── config.py
│   ├── lessons.py
│   ├── main.py
│   ├── Dockerfile
│   ├── poetry.lock
├── streamlit
│   ├── ui
│   │   ├── lesson.py
│   │   ├── planner.py
│   │   ├── plans.py
│   │   ├── quiz.py
│   ├── app.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── README.md
```

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone <repository-link>
cd <repository-folder>
```

### **2. Setup Environments**

#### **For Streamlit**:
```bash
cd streamlit
python -m venv venv_streamlit  # Create virtual environment
source venv_streamlit/bin/activate  # Activate the environment
poetry install  # Install dependencies
```

#### **For FastAPI**:
```bash
cd fastapi
python -m venv venv_fastapi  # Create virtual environment
source venv_fastapi/bin/activate  # Activate the environment
poetry install  # Install dependencies
```

### **3. Run the Components**

#### **Airflow**:
```bash
cd Airflow
docker-compose up --build
```

#### **FastAPI**:
```bash
cd fastapi
source venv_fastapi/bin/activate
uvicorn main:app --reload
```

#### **Streamlit**:
```bash
cd streamlit
source venv_streamlit/bin/activate
streamlit run app.py
```

---

## **Conclusion**
This modular project integrates Airflow, FastAPI, and Streamlit to provide a dynamic learning assistant. With separate environments and comprehensive testing, it is scalable, efficient, and user-friendly.

---

Let me know if you'd like any part of this adjusted!

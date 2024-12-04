import os
import streamlit as st
from dotenv import load_dotenv
import requests
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
import json

# Loading API Keys using environment variables (add a .env file in root folder) and initializing clients for our LLM and Knowledge Base)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Defining pinecone index name and creating an index (if already doesn't exsist) 
INDEX_NAME = "test"
if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index(INDEX_NAME)



# -----Utility Functions-----

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generates an embedding vector for the given text using OpenAI's embedding model.

    Args:
        text (str): The text to encode.
        model (str): The OpenAI model name for embeddings.

    Returns:
        list: Embedding vector for the text.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    MAX_TOKENS = 8192

    if len(tokens) > MAX_TOKENS:
        text = encoding.decode(tokens[:MAX_TOKENS])

    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def retrieve_information(user_query):
    """
    Handles embedding generation and querying Pinecone to retrieve information from our knowledge base.
    
    Args:
        user_query (str): The user query.
        
    Returns:
        str: The retrieved information in a concatenated format.
    """
    query_embedding = get_embedding(user_query)
    results = index.query(vector=query_embedding, top_k=4, include_metadata=True)
    
    context_info = " ".join([
        f"Chunk ID: {match['metadata'].get('chunk_id', 'Unknown')}. Text: {match['metadata'].get('text', '').strip()}"
        for match in results["matches"]
    ])
    return context_info
def generate_plan(user_query, context_info, current_plan=None):
    """
    Calls GPT to generate or refine a structured learning plan based on the query and retrieved information.

    Args:
        user_query (str): The user query.
        context_info (str): The retrieved information.
        current_plan (dict, optional): The current plan to refine. If None, a new plan will be created.

    Returns:
        str: The generated or refined learning plan in structured JSON format.
    """
    if not context_info:
        return "No relevant information found in the Knowledge Base"

    # Base prompt for the assistant
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

    # Add the current plan to the user query if it exists
    if current_plan:
        # Format the current plan as JSON for clarity
        current_plan_text = json.dumps(current_plan, indent=2)
        user_prompt = (
            f"Here is the current learning plan:\n{current_plan_text}\n\n"
            f"User Query: {user_query}\n"
            "Update the learning plan based on the query. Make sure to incorporate the user's new focus or requirements "
            "while retaining the structure and relevant parts of the existing plan."
        )
    else:
        # If no current plan exists, create a new one
        user_prompt = (
            f"User Query: {user_query}\nContext: {context_info}\n"
            "Create a new structured learning plan based on the query and context."
        )

    # Call the OpenAI API
    plan_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return plan_response.choices[0].message.content


def validate_and_clean_json(response_text):
    """
    Validates and extracts JSON from the LLM response.

    Args:
        response_text (str): The raw response text from the LLM.

    Returns:
        dict: Parsed JSON object if valid, or None if invalid.
    """
    try:
        # Locate the JSON part by finding the opening and closing braces
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx == -1 or end_idx == -1:
            return None  # No valid JSON found
        
        # Extract the JSON substring and parse it
        json_text = response_text[start_idx:end_idx]
        return json.loads(json_text)
    except Exception as e:
        return None


def summarize_plan(plan):
    """
    Summarizes the learning plan into a concise, practical, and straightforward description.
    
    Args:
        plan (str): The detailed learning plan.
        
    Returns:
        str: The summarized learning plan as a simple, logical description.
    """
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

def parse_learning_plan(plan_text):
    """
    Parses the unstructured LLM learning plan output into a structured format.
    """
    lines = plan_text.split('\n')
    structured_plan = {
        "Objective": "",
        "KeyTopics": [],
        "Weeks": [],
        "Timeline": "",
        "ExpectedOutcome": ""
    }
    current_week = None

    for line in lines:
        line = line.strip()
        if line.startswith("Objective:"):
            structured_plan["Objective"] = line.split("Objective:")[-1].strip()
        elif line.startswith("Key Topics:"):
            structured_plan["KeyTopics"] = [topic.strip() for topic in line.split("Key Topics:")[-1].split(",")]
        elif line.startswith("Week"):
            current_week = {"week": line, "title": "", "details": ""}
            structured_plan["Weeks"].append(current_week)
        elif current_week and line:
            if not current_week["title"]:
                current_week["title"] = line
            else:
                current_week["details"] += f" {line}"
        elif line.startswith("Timeline:"):
            structured_plan["Timeline"] = line.split("Timeline:")[-1].strip()
        elif line.startswith("Expected Outcome:"):
            structured_plan["ExpectedOutcome"] = line.split("Expected Outcome:")[-1].strip()

    return structured_plan

def main():
    st.title("Planner")

    # Initialize chat history and current plan in session state if not present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "current_plan" not in st.session_state:
        st.session_state["current_plan"] = None

    # Chat interface with chat history display (user inputs only)
    st.markdown("### Your Queries:")
    with st.container():
        if st.session_state["chat_history"]:
            for chat in st.session_state["chat_history"]:
                st.markdown(f"🗨️ {chat}")  # Display user inputs in a chat-like format
        else:
            st.markdown("No queries yet. Start by asking something!")

    # Chat input for user query
    user_query = st.chat_input("What do you want to learn today?")

    if user_query:
        # Add user query to chat history
        st.session_state["chat_history"].append(user_query)

        # Retrieve context and determine whether to refine or create a new plan
        if st.session_state["current_plan"]:
            # If a current plan exists, pass it as context for refinement
            context_info = retrieve_information(user_query)
            current_plan_text = json.dumps(st.session_state["current_plan"])
            combined_context = f"Current Plan: {current_plan_text}\nUser Query: {user_query}"
            learning_plan_json = generate_plan(user_query, combined_context)
        else:
            # If no current plan exists, generate a new one
            context_info = retrieve_information(user_query)
            learning_plan_json = generate_plan(user_query, context_info)

        # Parse JSON response
        try:
            plan = validate_and_clean_json(learning_plan_json)
            if not plan:
                raise ValueError("Invalid JSON format in LLM response.")
            st.session_state["current_plan"] = plan  # Save the new or updated plan
        except ValueError as ve:
            st.error("Failed to parse learning plan JSON. Please ensure the LLM returns valid JSON.")
            return
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return

        # Generate the summary
        summary = summarize_plan(learning_plan_json)

        # Display Metadata
        with st.expander("Outcomes"):
            st.markdown(f"**Timeline:** {plan.get('Timeline', 'N/A')}")
            st.markdown(f"**Expected Outcome:** {plan.get('ExpectedOutcome', 'N/A')}")

        # Display Summary as Plain Text
        st.markdown("### Summary")
        st.text(summary)

        # Display Weekly Tabs
        st.markdown("### Weekly Plan")
        week_tabs = [week["week"] for week in plan.get("Weeks", [])]
        tab_containers = st.tabs(week_tabs)

        for i, tab in enumerate(tab_containers):
            with tab:
                week = plan.get("Weeks", [])[i]
                st.markdown(f"#### {week['title']}")
                st.write(week['details'])

                if st.button(f"Go to Lesson - {week['week']}", key=f"button_{week['week']}"):
                    st.session_state["current_week"] = week
                    st.session_state["page"] = "lesson"
                    st.rerun()

        st.markdown("### Key Topics")
        for topic in plan.get('KeyTopics', []):
            st.write(f"- {topic}")

    # Check if a week is selected and navigate to lesson page
    if "page" in st.session_state and st.session_state["page"] == "lesson":
        lesson.main()

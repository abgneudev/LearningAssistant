import os
import streamlit as st
from dotenv import load_dotenv
import requests
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

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

def get_embedding(text, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    MAX_TOKENS = 8192

    if len(tokens) > MAX_TOKENS:
        text = encoding.decode(tokens[:MAX_TOKENS])

    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def retrieve_context(query):
    """
    Handles embedding generation and querying Pinecone for context.
    
    Args:
        query (str): The user query.
        
    Returns:
        str: The retrieved context in a concatenated format.
    """
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    
    context = " ".join([
        f"Chunk ID: {match['metadata'].get('chunk_id', 'Unknown')}. Text: {match['metadata'].get('text', '').strip()}"
        for match in results["matches"]
    ])
    return context

def generate_plan(query, context):
    """
    Calls GPT to generate a learning plan based on the query and context.
    
    Args:
        query (str): The user query.
        context (str): The retrieved context.
        
    Returns:
        str: The generated learning plan.
    """
    if not context:
        return "No relevant context found in the Knowledge Base"

    plan_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": (
                "You are an assistant that creates specific and actionable MVP-style learning plans based on the user's query. "
                "Focus on clear objectives, key topics, step-by-step actions, a realistic timeline, and clear expected outcomes. "
                "Adjust the number of lessons to suit the user's query and complexity of the topic."
            )},
            {"role": "user", "content": (
                f"User Query: {query}\n"
                f"Learning Context: {context}\n\n"
                "Generate a learning plan following these guidelines:\n"
                "1. Objective: Define the goal.\n"
                "2. Key Topics: Break the goal into essential learning modules.\n"
                "3. Step-by-Step Actions: Provide practical steps for each topic.\n"
                "4. Timeline: Suggest a realistic completion time.\n"
                "5. Expected Outcome: Clearly state what the user will achieve."
            )}
        ]
    )
    return plan_response.choices[0].message.content 

def summarize_plan(plan):
    """
    Summarizes the generated learning plan into a concise description.
    
    Args:
        plan (str): The detailed learning plan.
        
    Returns:
        str: The summarized learning plan.
    """
    # Summarize the learning plan
    summary_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that summarizes learning plans into concise descriptions."},
            {"role": "user", "content": f"Learning Plan: {plan}. Summarize it briefly."}
        ]
    )
    return summary_response.choices[0].message.content

def main():
    st.title("Planner")

    query = st.text_input("What do you want to learn today?")

    if query:
        context = retrieve_context(query)
        learning_plan = generate_plan(query, context)
        summary = summarize_plan(learning_plan)

        if learning_plan:
            st.markdown("### Summary")
            st.write(summary)

            st.markdown("### Learning Plan")
            st.write(learning_plan)
        else:
            st.write("No plan generated")

if __name__ == "__main__":
    main()

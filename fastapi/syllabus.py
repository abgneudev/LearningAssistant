import json
from typing import Optional, List
from pydantic import BaseModel, ValidationError
from datetime import datetime
from fastapi import HTTPException
from openai import OpenAI
from pinecone import Pinecone
import tiktoken
from utils import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_token,
    get_db_connection,
    get_current_username,
    get_user,
    create_user,
    inspect_index,
    YouTubeVideoResponse,
    Module,
    Plan,
    pool,
)  
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
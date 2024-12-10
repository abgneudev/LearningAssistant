import logging
from fastapi import HTTPException
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from pydantic import BaseModel
from typing import Optional, List
import logging
from syllabus import get_embedding
from config import (
    client,
    youtube,
    index,
    youtube_index,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
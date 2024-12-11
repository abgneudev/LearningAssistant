import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the FastAPI URL from environment variables
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")

def main():
    st.title("Lesson Details")

    # Ensure username, access_token, and selected_module_id are available in session state
    if (
        "username" not in st.session_state
        or "access_token" not in st.session_state
        or "selected_module_id" not in st.session_state
    ):
        st.error("You are not logged in or no module was selected. Please go back and select a module.")
        return

    access_token = st.session_state["access_token"]
    selected_module_id = st.session_state["selected_module_id"]

    # Fetch details for the selected module from the backend
    try:
        response = requests.get(
            f"{FASTAPI_URL}/get_module_details/{selected_module_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code == 200:
            module_details = response.json()
            if isinstance(module_details, dict) and "message" in module_details:
                st.warning(module_details["message"])
                return
        else:
            st.error("Failed to fetch module details.")
            return
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return

    # Display the module details
    st.markdown(f"## Module {module_details.get('module')}: {module_details.get('title')}")
    st.write(module_details.get("description", "No description available."))

    # Display detailed explanation as an article
    detailed_explanation = module_details.get("detailed_explanation", "No detailed explanation available.")
    st.markdown("### Detailed Explanation")
    paragraphs = detailed_explanation.split("\n\n")  # Split into paragraphs by double line breaks
    for paragraph in paragraphs:
        st.markdown(paragraph.strip())  # Render each paragraph as Markdown with proper spacing

    # Add a back button to navigate back to the plan page
    if st.button("Back to Plans"):
        del st.session_state["selected_module_id"]
        st.session_state["page"] = "planner"
        st.rerun()

    try:
        with st.spinner("Fetching the most relevant YouTube video..."):
            video_response = requests.get(
                f"{FASTAPI_URL}/get_relevant_youtube_video/{selected_module_id}"
            )

        if video_response.status_code == 200:
            video_data = video_response.json()
            if video_data["video_url"]:
                st.markdown("### Most Relevant YouTube Video")
                st.write(f"**Relevance Score:** {video_data['relevance_score']:.2f}")
                st.video(video_data["video_url"])
            else:
                st.warning("No relevant YouTube video found for this module.")
        else:
            st.error("Failed to fetch relevant YouTube video.")
    except Exception as e:
        st.error(f"An error occurred while fetching the YouTube video: {e}")

if __name__ == "__main__":
    main()

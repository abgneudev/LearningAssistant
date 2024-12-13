import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the FastAPI URL from environment variables
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

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

    # Add a back button to navigate back to the plan page
    if st.button("Back to Plans"):
        del st.session_state["selected_module_id"]
        st.session_state["page"] = "plans"
        st.rerun()

    # Display detailed explanation as an article
    detailed_explanation = module_details.get("detailed_explanation", "No detailed explanation available.")
    st.markdown("### Detailed Explanation")
    paragraphs = detailed_explanation.split("\n\n")  # Split into paragraphs by double line breaks
    for paragraph in paragraphs:
        st.markdown(paragraph.strip())  # Render each paragraph as Markdown with proper spacing

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

    if video_response.status_code == 200 and video_data.get("video_url"):
        try:
            with st.spinner("Generating flashcards..."):
                flashcards_response = requests.get(f"{FASTAPI_URL}/generate_flashcards/{selected_module_id}")
            if flashcards_response.status_code == 200:
                # Parse the JSON response
                flashcards_data = flashcards_response.json()  # Expecting a nested JSON structure
                flashcards = flashcards_data.get("flashcards", [])  # Extract the list of flashcards
                if flashcards:
                    # st.subheader(f"Flashcards for Module ID: {selected_module_id}")
                    st.subheader(f"Flashcards")
                    for i, flashcard in enumerate(flashcards):
                        question = flashcard.get("question", "Question not available.")
                        answer = flashcard.get("answer", "Answer not available.")
                        # Display each flashcard in a clean format
                        st.markdown(f"**Flashcard {i+1}**")
                        st.markdown(f"**Q: {question}**")
                        st.markdown(f"**A: {answer}**")
                        st.markdown("---")  # Separator for clarity
                else:
                    st.warning("No flashcards generated. Please try again.")
            else:
                st.error("Failed to generate flashcards. Please check the module ID or try again later.")
        except Exception as e:
            st.error(f"An error occurred while generating flashcards: {e}")

    # if video_response.status_code == 200 and video_data.get("video_url"):
    #     try:
    #         with st.spinner("Generating quiz..."):
    #             quiz_response = requests.get(f"{FASTAPI_URL}/generate_quiz/{selected_module_id}")
    #         if quiz_response.status_code == 200:
    #             # Parse the JSON response
    #             quiz_data = quiz_response.json()  # Expecting a nested JSON structure
    #             quiz_questions = quiz_data.get("quiz", [])  # Extract the list of quiz questions
    #             if quiz_questions:
    #                 st.subheader("Quiz")
    #                 user_answers = []  # List to store user-selected answers

    #                 for i, question_item in enumerate(quiz_questions):
    #                     question = question_item.get("question", "Question not available.")
    #                     options = question_item.get("options", [])

    #                     st.markdown(f"**Question {i + 1}: {question}**")
    #                     if options:
    #                         selected_option = st.radio(
    #                             label=f"Select an answer for Question {i + 1}",
    #                             options=options,
    #                             key=f"quiz_q{i}"
    #                         )
    #                         user_answers.append({
    #                             "question": question,
    #                             "selected_option": selected_option
    #                         })
    #                     else:
    #                         st.warning("No options available for this question.")

    #                     st.markdown("---")  # Separator for clarity

    #                 if st.button("Submit Quiz"):
    #                     st.subheader("Quiz Results")
    #                     correct_answers_count = 0

    #                     for i, answer in enumerate(user_answers):
    #                         correct_answer = quiz_questions[i].get("correct_answer", "")
    #                         selected_option = answer["selected_option"]

    #                         if selected_option == correct_answer:
    #                             correct_answers_count += 1
    #                             st.markdown(f"✅ **Question {i + 1}: Correct!**")
    #                         else:
    #                             st.markdown(f"❌ **Question {i + 1}: Incorrect.**")
    #                             st.markdown(f"**Correct Answer:** {correct_answer}")

    #                     total_questions = len(quiz_questions)
    #                     st.markdown(f"**You got {correct_answers_count} out of {total_questions} questions correct.**")
    #             else:
    #                 st.warning("No quiz generated. Please try again.")
    #         else:
    #             st.error("Failed to generate quiz. Please check the module ID or try again later.")
    #     except Exception as e:
    #         st.error(f"An error occurred while generating the quiz: {e}")

if __name__ == "__main__":
    main()

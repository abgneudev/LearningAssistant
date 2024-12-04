import streamlit as st
import requests
import json

def main():
    st.title("Planner")

    # Initialize chat history, current plan, and fallback response if not present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "current_plan" not in st.session_state:
        st.session_state["current_plan"] = None
    if "current_summary" not in st.session_state:
        st.session_state["current_summary"] = "No summary provided yet."  # To retain the summary for the current plan
    if "general_response" not in st.session_state:
        st.session_state["general_response"] = "No general response yet."

    # Chat input for user query
    user_query = st.chat_input("What do you want to learn today?")

    if user_query:
        # Add user query to chat history
        st.session_state["chat_history"].append({"role": "user", "content": user_query})

        # Make a request to the backend for processing
        response = requests.post(
            "http://127.0.0.1:8000/query",  # Using the `/query` endpoint
            json={
                "user_query": user_query,
                "current_plan": st.session_state.get("current_plan"),
                "current_summary": st.session_state.get("current_summary"),  # Pass current summary
            },
        )

        if response.status_code == 200:
            data = response.json()

            # Handle learning plan if it exists
            if data.get("plan"):
                st.session_state["current_plan"] = data["plan"]
                summary = data.get("summary", st.session_state["current_summary"])  # Retain the previous summary if not updated
                st.session_state["current_summary"] = summary  # Update or retain the summary
                st.session_state["chat_history"].append({"role": "assistant", "content": summary})

                # Update general response to indicate the plan has been updated
                st.session_state["general_response"] = data.get(
                    "response", "I've generated a plan for you based on your query."
                )

                # Display Metadata
                if st.session_state["current_plan"]:
                    with st.expander("Outcomes"):
                        timeline = st.session_state["current_plan"].get("Timeline", "N/A")
                        expected_outcome = st.session_state["current_plan"].get("ExpectedOutcome", "N/A")
                        st.markdown(f"**Timeline:** {timeline}")
                        st.markdown(f"**Expected Outcome:** {expected_outcome}")

                    # Display Summary
                    st.markdown("### Summary")
                    st.text(summary)

                    # Display Weekly Tabs
                    st.markdown("### Weekly Plan")
                    weeks = st.session_state["current_plan"].get("Weeks", [])
                    if weeks:
                        week_tabs = [week["week"] for week in weeks]
                        tab_containers = st.tabs(week_tabs)

                        for i, tab in enumerate(tab_containers):
                            with tab:
                                week = weeks[i]
                                st.markdown(f"#### {week['title']}")
                                st.write(week['details'])

                                if st.button(f"Go to Lesson - {week['week']}", key=f"button_{week['week']}"):
                                    st.session_state["current_week"] = week
                                    st.session_state["page"] = "lesson"
                                    st.rerun()

                    # Display Key Topics
                    st.markdown("### Key Topics")
                    for topic in st.session_state["current_plan"].get("KeyTopics", []):
                        st.write(f"- {topic}")

            # Handle fallback response if no plan is available
            elif data.get("response"):
                # Update the general response dynamically
                st.session_state["general_response"] = data["response"]
                st.session_state["chat_history"].append({"role": "assistant", "content": data["response"]})

            else:
                # Handle the case where no response or plan is provided
                st.session_state["general_response"] = "Failed to fetch the learning plan details."
                st.error("Failed to fetch the learning plan details.")
        else:
            # Handle error responses from the backend
            st.session_state["general_response"] = "Failed to process the query. Please try again later."
            st.error("Failed to process the query. Please try again later.")

    # Display the general response in another text area
    st.text_area(
        "General Response",
        value=st.session_state.get("general_response", "No general response yet."),
        height=200,
        key="general_response_text"
    )

if __name__ == "__main__":
    main()

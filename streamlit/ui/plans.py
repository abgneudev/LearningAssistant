import streamlit as st
import requests


def main():
    st.title("Saved Plans")

    # Button to navigate back to the Planner
    if st.button("Make a new Plan"):
        st.session_state["page"] = "planner"
        st.rerun()


    # Fetch all saved plans from the backend
    try:
        response = requests.get("http://127.0.0.1:8000/get_plans")
        if response.status_code == 200:
            plans = response.json()  # Assuming backend returns a list of plans
        else:
            st.error("Failed to fetch plans.")
            plans = []
    except Exception as e:
        st.error(f"An error occurred: {e}")
        plans = []

    if not plans:
        st.info("No saved plans available.")
        return

    # Dropdown to select a plan
    plan_options = {plan["plan_id"]: plan["summary"] for plan in plans}
    selected_plan_id = st.selectbox(
        "Select a Plan",
        options=plan_options.keys(),
        format_func=lambda x: plan_options[x] if x in plan_options else "Unknown Plan",
    )

    # Display selected plan details
    if selected_plan_id:
        selected_plan = next((plan for plan in plans if plan["plan_id"] == selected_plan_id), None)
        
        if not selected_plan:
            st.error("Selected plan could not be found.")
            return

        st.markdown(f"### Plan Summary")
        st.write(selected_plan["summary"])

        st.markdown(f"### Key Topics")
        key_topics = selected_plan.get("key_topics", [])
        if isinstance(key_topics, list):
            st.write(", ".join(key_topics) if key_topics else "No key topics available.")
        else:
            st.write("Invalid key topics format.")

        st.markdown(f"### Timeline")
        st.write(selected_plan.get("timeline", "No timeline provided."))

        st.markdown(f"### Outcomes")
        st.write(selected_plan.get("learning_outcomes", "No outcomes provided."))

        # Fetch weeks for the selected plan
        try:
            weeks_response = requests.get(f"http://127.0.0.1:8000/get_weeks/{selected_plan_id}")
            if weeks_response.status_code == 200:
                weeks = weeks_response.json()  # Assuming backend returns a list of weeks
                if isinstance(weeks, dict) and "message" in weeks:
                    st.warning(weeks["message"])  # Display message if no weeks are found
                    weeks = []
            else:
                st.error("Failed to fetch weeks for the selected plan.")
                weeks = []
        except Exception as e:
            st.error(f"An error occurred while fetching weeks: {e}")
            weeks = []

        if weeks:
            st.markdown("## Plan Timeline (Weeks)")
            for week in weeks:
                st.markdown(f"#### Week {week['week']}: {week['title']}")
                st.write(week.get("description", "No description available."))

                # Add a button to navigate to the "lesson" page for each week
                if st.button(f"Go to Lesson for Week {week['week']}", key=f"lesson_{week['week_id']}"):
                    st.session_state["selected_week_id"] = week["week_id"]
                    st.session_state["page"] = "lesson"
                    st.rerun()

if __name__ == "__main__":
    main()

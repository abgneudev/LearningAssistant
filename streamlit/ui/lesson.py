import streamlit as st

def main():
    st.title("Lesson Details")

    # Check if a week is selected
    if "current_week" in st.session_state:
        week = st.session_state["current_week"]

        st.markdown(f"### {week['week']}: {week['title']}")
        st.write(week['details'])

        # Add any additional content or actions for the lesson page
        st.markdown("#### Lesson Content")
        st.write("This is where lesson-specific content or actions would go.")

        # Add a back button to return to the planner
        if st.button("Back to Planner"):
            st.session_state["page"] = "planner"  # Update the page state to planner
            st.rerun()  # Reload the app to reflect the change
    else:
        st.write("No lesson selected. Go back to the planner.")
        if st.button("Back to Planner"):
            st.session_state["page"] = "planner"  # Update the page state to planner
            st.rerun()  # Reload the app to reflect the change

if __name__ == "__main__":
    main()

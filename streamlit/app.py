import os
import streamlit as st
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

st.set_page_config(layout="wide")

# FastAPI URL from environment variables
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

def user_signup(username, password):
    """
    Function to register a new user, sending a post request to the fastapi backend
    """
    response = requests.post(f"{FASTAPI_URL}/signup", params={"username": username, "password": password})
    return response.json()

def user_login(username, password):
    """
    Function to login an existing user, verifying the user user credentials through Snowflake
    """
    response = requests.post(f"{FASTAPI_URL}/login", params={"username": username, "password": password})
    return response.json()

def signup():
    st.subheader("Signup page!")

    username = st.text_input("Create a valid username")
    password = st.text_input("Create a Valid password", type="password")
    confirm_password = st.text_input("Confirm your password", type="password")

    if st.button("Signup"):
        if password == confirm_password:
            result = user_signup(username, password)
            if result.get("message"):
                st.success(result["message"])
            else:
                st.error(result.get("detail", "Signup failed"))
        else:
            st.error("Passwords do not match. Kindly retry and enter correct passwords.")

def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        result = user_login(username, password)
        if "access_token" in result:
            st.session_state["access_token"] = result["access_token"]
            st.session_state["username"] = username
            st.session_state["logged_in"] = True
            st.session_state["login_time"] = datetime.now(timezone.utc)
            st.session_state["page"] = "planner"
            st.rerun()
        else:
            st.error(result.get("detail", "Login failed"))

def user_logout():
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

def page_loader():
    """Loads planner page"""
    if st.session_state["page"]=="planner":
        import ui.planner as planner
        planner.main()
    else:
        st.error("404 - Page Not Found")

def main():
    """Main entry point for the Streamlit app."""
    st.title("Personalized AI Learning Assistant")

    # Ensure session state variables are initialized
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    # Display appropriate pages based on login state
    if st.session_state["logged_in"]:
        page_loader()
    else:
        choice = st.radio("Choose an option:", ("Login", "Signup"))
        if choice == "Signup":
            signup()
        else:
            login()

if __name__ == "__main__":
    main()


import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from agent import SimpleAgent, create_gemini_client

# --- Initialize agent ---
db_path = "jobs.db"  # same SQLite DB
client = create_gemini_client()
agent = SimpleAgent(db_path=db_path, gemini_client=client)

st.title("Job Openings Agent (Streamlit)")

# --- Tabs for different actions ---
tabs = ["Ingest jobs from web", "Search jobs (semantic)", "List stored jobs"]
choice = st.sidebar.radio("Select Action", tabs)

if choice == "Ingest jobs from web":
    st.subheader("Ingest jobs from the web (Tavily Search)")
    keywords = st.text_input("Enter job keywords (e.g., python backend)")
    location = st.text_input("Optional location")
    max_results = st.number_input("Number of results", min_value=1, max_value=20, value=5)

    if st.button("Fetch & Store Jobs"):
        if not keywords.strip():
            st.warning("Please enter keywords!")
        else:
            inserted = agent.ingest_from_web(keywords, location or None, max_results)
            st.success(f"Inserted {inserted} new jobs into DB")

elif choice == "Search jobs (semantic)":
    st.subheader("Search jobs using natural language")
    query = st.text_input("Enter your search query")

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a query!")
        else:
            answer, matches = agent.answer_query_with_rag(query)
            st.markdown("**RAG Answer:**")
            st.write(answer)
            st.markdown("---")
            st.markdown("**Top matching jobs:**")
            for job in matches:
                st.write(f"**{job['title']}** at {job['company']} ({job['location']})")
                st.write(f"URL: {job['url']}")
                st.write(job.get("description", "No description"))
                st.markdown("---")

elif choice == "List stored jobs":
    st.subheader("All jobs in the database")
    jobs = agent.semantic_search(" ")  # dummy query to fetch all jobs
    if jobs:
        for job in jobs:
            st.write(f"{job['id']}: {job['title']} at {job['company']} ({job['location']})")
    else:
        st.info("No jobs found in the database.")

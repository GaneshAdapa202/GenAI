"""
main.py
Small CLI to:
1) ingest jobs from web (tavily)
2) query the local DB using natural language (RAG)
3) list stored jobs
"""

import os
from dotenv import load_dotenv
from agent import SimpleAgent, create_gemini_client

# load .env
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "job_postings.db")

def main():
    print("== Job Openings Agent (simple demo) ==")
    # create gemini client and agent
    agent = SimpleAgent(db_path=DB_PATH)

    while True:
        print("\nChoose an action:")
        print("1) Ingest jobs from web (keywords)")
        print("2) Search DB (natural language)")
        print("3) List stored jobs (ids, titles)")
        print("4) Exit")
        choice = input("Enter choice [1-4]: ").strip()
        if choice == "1":
            kws = input("Enter job keywords (e.g., 'python backend'): ").strip()
            loc = input("Optional location (press Enter to skip): ").strip() or None
            maxr = input("How many results to fetch (default 5): ").strip()
            try:
                maxr = int(maxr) if maxr else 5
            except:
                maxr = 5
            agent.ingest_from_web(kws, location=loc, max_results=maxr)
        elif choice == "2":
            q = input("Enter a natural-language search (e.g., 'remote python jobs with AWS experience'): ").strip()
            if not q:
                print("Please enter a non-empty query.")
                continue
            answer, matches = agent.answer_query_with_rag(q, top_k=3)
            print("\n=== Agent Answer ===")
            print(answer)
            print("\n=== Matched Jobs (top) ===")
            for m in matches:
                print(f"- {m.get('title')} @ {m.get('company')} — {m.get('url')}")
        elif choice == "3":
            rows = agent_client = None
            from db import list_jobs
            rows = list_jobs(DB_PATH, limit=50)
            if not rows:
                print("No jobs in DB yet.")
            else:
                for r in rows:
                    print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]}")
        elif choice == "4":
            print("Bye.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()

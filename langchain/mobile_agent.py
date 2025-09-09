import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema import StrOutputParser

def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python mobile_agent.py 'Mobile Name'")
    #     return

    # mobile_name = sys.argv[1]
    # print(f"Fetching info for: {mobile_name}")

    if len(sys.argv) > 1:
        mobile_name = sys.argv[1]
    else:
        mobile_name = input("Enter mobile name: ")

    # API Key check
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY is missing! Run: export GOOGLE_API_KEY='your_key'")
        return

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
    )

    # Initialize DuckDuckGo tool
    search = DuckDuckGoSearchRun()

    # Search the web
    search_results = search.run(f"{mobile_name} specifications and details")
    print("Web search done")

    # Create prompt with search results
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a mobile expert. Use the given web data to provide accurate mobile details."),
        ("human", "Mobile: {mobile_name}\n\nWeb data:\n{web_data}\n\nNow give a detailed description and specifications.")
    ])

    # Create pipeline
    chain = prompt | llm | StrOutputParser()

    # Run pipeline
    response = chain.invoke({"mobile_name": mobile_name, "web_data": search_results})

    print("\n Final Response:\n")
    print(response)

if __name__ == "__main__":
    main()

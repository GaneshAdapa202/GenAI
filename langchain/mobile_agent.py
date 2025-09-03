import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

def main():
    print("✅ Script started")  # Debug

    if len(sys.argv) < 2:
        print("❌ No mobile name provided. Usage: python mobile_agent.py 'iPhone 15 Pro Max'")
        return

    mobile_name = sys.argv[1]
    print(f"🔍 Mobile requested: {mobile_name}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY is missing!")
        return
    print("🔑 API Key detected")

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=api_key,
        )
        print("✅ LLM initialized")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a mobile specification expert."),
            ("human", "Give me detailed specifications and description of {mobile_name}.")
        ])
        chain = prompt | llm
        print("⚙️ Chain created")

        response = chain.invoke({"mobile_name": mobile_name})
        print("✅ Response received")

        print("\n📱 Mobile Specifications:\n")
        print(response.content)

    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    main()

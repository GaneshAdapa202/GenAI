# from langchain_google_genai import ChatGoogleGenerativeAI
# import sys
# import os

# def main():
#     print("Script started")   # Debug line 1

#     api_key = os.environ.get("GOOGLE_API_KEY")
#     print(f"API Key Found: {bool(api_key)}")  # Debug line 2

#     if not api_key:
#         print("GOOGLE_API_KEY not set. Run: export GOOGLE_API_KEY='your_api_key_here'")
#         return

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         temperature=0.3
#     )
#     print("Model initialized")  # Debug line 3

#     if len(sys.argv) < 2:
#         print("Usage: python mobile_agent.py \"Mobile Name\"")
#         return

#     mobile_name = sys.argv[1]
#     print(f"Input mobile: {mobile_name}")  # Debug line 4

#     query = f"Give me detailed specifications and description of {mobile_name}."
#     print(f"Query: {query}")  # Debug line 5

#     try:
#         response = llm.invoke(query)
#         print("Model invoked")  # Debug line 6
#         print("Raw response object:", response)  # Debug line 7

#         if hasattr(response, "content"):
#             print("\n Response content:")
#             print(response.content)
#         else:
#             print("\n No 'content' attribute in response")
#     except Exception as e:
#         print("Error while invoking model:", str(e))

# if __name__ == "__main__":
#     main()

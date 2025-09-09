from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # instead of gemini-pro
    temperature=0.2
)

print("Starting Gemini test...")
response = llm.invoke("Give me details of iPhone 15 Pro Max")
print("Gemini Response:", response.content)

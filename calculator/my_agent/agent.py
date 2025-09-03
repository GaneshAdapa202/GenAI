
from google.adk.agents import Agent
from .tools import calculator
import os

def create_agent(api_key: str):
    os.environ["GOOGLE_API_KEY"] = api_key
    

    agent = Agent(
        name = "calculator_agent",
        model="gemini-2.0-flash",
        instruction="You can use the calculator tool to evaluate expressions.",
        tools=[calculator]
    )
    return agent

root_agent=create_agent("AIzaSyApJ_ILJfhjXxv5XpyZsi-pBHEmwr16jo4")
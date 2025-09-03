from my_agent import create_agent

API_KEY = "AIzaSyApJ_ILJfhjXxv5XpyZsi-pBHEmwr16jo4"

agent = create_agent(API_KEY)

print(agent.run("Hello! Can you tell me a fun fact about space?"))
print(agent.run("Use the calculator to solve (15 + 10) * 3"))
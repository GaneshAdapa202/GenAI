import yaml
from crewai import Agent, Task, Crew

class Demo:
    def __init__(self):
        # Load agent definitions
        with open("agents.yaml", "r") as f:
            self.agent_data = yaml.safe_load(f)

        # Load task definitions
        with open("tasks.yaml", "r") as f:
            self.task_data = yaml.safe_load(f)

    def crew(self, topic="AI LLMs", current_year="2025"):
        # Define agents
        researcher = Agent(
            role=self.agent_data["researcher"]["role"].format(topic=topic),
            goal=self.agent_data["researcher"]["goal"].format(topic=topic),
            backstory=self.agent_data["researcher"]["backstory"].format(topic=topic),
        )

        analyst = Agent(
            role=self.agent_data["analyst"]["role"].format(topic=topic),
            goal=self.agent_data["analyst"]["goal"].format(topic=topic),
            backstory=self.agent_data["analyst"]["backstory"].format(topic=topic),
        )

        # Define tasks
        research_task = Task(
            description=self.task_data["research_task"]["description"].format(
                topic=topic, current_year=current_year
            ),
            expected_output=self.task_data["research_task"]["expected_output"].format(topic=topic),
            agent=researcher,
        )

        # Create crew with 2 agents, but 1 task
        crew = Crew(
            agents=[researcher, analyst],
            tasks=[research_task],
            verbose=True,
        )
        return crew

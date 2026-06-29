from crewai import Agent, Task, Crew, Process, LLM
from crewai import tool
from duckduckgo_search import DDGS
import os

# Custom DuckDuckGo search tool using CrewAI's native @tool decorator
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web using DuckDuckGo and return a summary of results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No results found."
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
        for r in results
    )

class ResearchCrew:
    """
    Encapsulates the Multi-Agent Research Logic.
    This structure is easy to explain in an interview as a 'Pipeline pattern'.
    """
    def __init__(self, topic: str, instructions: str = None, api_key: str = None):
        self.topic = topic
        self.instructions = instructions

        # Set the API key in the environment so CrewAI's LLM wrapper picks it up
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        # CrewAI's native LLM class — pass model as a LiteLLM-format string
        self.llm = LLM(model="gemini/gemini-2.0-flash")

    def run(self):
        # 1. Define Agents
        planner = Agent(
            role="Research Planner",
            goal=f"Plan research objectives for: {self.topic}. Instructions: {self.instructions}",
            backstory="Expert in breaking down complex topics into actionable research plans.",
            llm=self.llm,
            allow_delegation=False
        )

        researcher = Agent(
            role="Lead Researcher",
            goal=f"Gather data on: {self.topic}",
            backstory="Specialist in web research and identifying key facts.",
            tools=[duckduckgo_search],
            llm=self.llm
        )

        writer = Agent(
            role="Technical Writer",
            goal=f"Draft a report on: {self.topic}",
            backstory="Expert in creating clear, structured Markdown reports.",
            llm=self.llm
        )

        # 2. Define Tasks
        plan_task = Task(
            description=f"Create a research plan for '{self.topic}'.",
            expected_output="A list of 3-5 key research objectives.",
            agent=planner
        )

        research_task = Task(
            description=f"Gather detailed findings based on the plan for '{self.topic}'.",
            expected_output="A comprehensive collection of facts and data points.",
            agent=researcher
        )

        write_task = Task(
            description=f"Write a final Markdown report on '{self.topic}'.",
            expected_output="A polished Markdown report with Summary, Findings, and Outlook.",
            agent=writer
        )

        # 3. Assemble the Crew
        crew = Crew(
            agents=[planner, researcher, writer],
            tasks=[plan_task, research_task, write_task],
            process=Process.sequential,
            verbose=True
        )

        return crew.kickoff()

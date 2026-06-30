from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
import os


class DuckDuckGoSearchTool(BaseTool):
    name: str = "duckduckgo_search"
    description: str = "Search the web using DuckDuckGo. Input should be a search query string."

    def _run(self, query: str) -> str:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
            for r in results
        )


class ResearchCrew:
    def __init__(self, topic: str, instructions: str = None, api_key: str = None):
        self.topic = topic
        self.instructions = instructions

        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

        # Lower temperature reduces malformed tool-call syntax from the model
        self.llm = LLM(model="groq/llama-3.3-70b-versatile", temperature=0.2)
        self.search_tool = DuckDuckGoSearchTool()

    def run(self):
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
            backstory="Specialist in web research and identifying key facts. Always uses the duckduckgo_search tool exactly as instructed, passing only a plain text query string.",
            tools=[self.search_tool],
            llm=self.llm,
            max_iter=5
        )

        writer = Agent(
            role="Technical Writer",
            goal=f"Draft a report on: {self.topic}",
            backstory="Expert in creating clear, structured Markdown reports.",
            llm=self.llm
        )

        plan_task = Task(
            description=f"Create a research plan for '{self.topic}'.",
            expected_output="A list of 3-5 key research objectives.",
            agent=planner
        )

        research_task = Task(
            description=f"Gather detailed findings based on the plan for '{self.topic}'. Use the duckduckgo_search tool with simple, short text queries (2-6 words) one at a time.",
            expected_output="A comprehensive collection of facts and data points.",
            agent=researcher
        )

        write_task = Task(
            description=f"Write a final Markdown report on '{self.topic}'.",
            expected_output="A polished Markdown report with Summary, Findings, and Outlook.",
            agent=writer
        )

        crew = Crew(
            agents=[planner, researcher, writer],
            tasks=[plan_task, research_task, write_task],
            process=Process.sequential,
            verbose=True
        )

        return crew.kickoff()

from crewai import Agent, Task, Crew, Process, LLM
import crewai.llms.cache as _crewai_cache
import litellm

# Workaround for Groq cache_breakpoint bug: https://github.com/crewAIInc/crewAI/issues/5886
# 1. Patch CrewAI's cache marker to do nothing
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

# 2. Patch LiteLLM to strip cache_breakpoint from messages as a safety net
def _strip_cache_breakpoints(messages):
    if not messages:
        return messages
    for msg in messages:
        if isinstance(msg, dict):
            msg.pop("cache_breakpoint", None)
    return messages

original_completion = litellm.completion
def patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_cache_breakpoints(kwargs["messages"])
    return original_completion(*args, **kwargs)
litellm.completion = patched_completion

original_acompletion = litellm.acompletion
async def patched_acompletion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_cache_breakpoints(kwargs["messages"])
    return await original_acompletion(*args, **kwargs)
litellm.acompletion = patched_acompletion

from crewai.tools import BaseTool
from duckduckgo_search import DDGS
import os


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
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

        # Groq is free, fast, and reliable — llama-3.3-70b is their best free model
        self.llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=api_key
        )
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
            backstory="Specialist in web research and identifying key facts.",
            tools=[self.search_tool],
            llm=self.llm
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
            description=f"Gather detailed findings based on the plan for '{self.topic}'.",
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

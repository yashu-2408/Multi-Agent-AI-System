from crewai import Agent, Task, Crew, Process, LLM
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


# Errors that indicate "this provider is unavailable right now, try the next one"
# rather than "something is actually broken in the code/config".
FALLOVER_KEYWORDS = [
    "429", "rate limit", "ratelimit", "resource_exhausted", "quota",
    "503", "unavailable", "overloaded", "high demand",
    "500", "internal server error", "timeout",
]


def _is_failover_worthy(error: Exception) -> bool:
    msg = str(error).lower()
    return any(keyword in msg for keyword in FALLOVER_KEYWORDS)


class ResearchCrew:
    """
    Encapsulates the Multi-Agent Research Logic.
    Supports automatic failover across multiple LLM providers — if one
    provider is rate-limited or down, it automatically retries the whole
    crew run with the next available provider/key.
    """

    def __init__(self, topic: str, instructions: str = None,
                 groq_api_key: str = None, gemini_api_key: str = None):
        self.topic = topic
        self.instructions = instructions
        self.search_tool = DuckDuckGoSearchTool()

        # Build an ordered list of (provider_label, LLM instance) to try.
        # Order = preference. Add/remove/reorder entries here freely.
        self.providers = []

        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
            self.providers.append((
                "Groq (llama-3.3-70b)",
                LLM(model="groq/llama-3.3-70b-versatile")
            ))
            self.providers.append((
                "Groq (llama-3.1-8b)",
                LLM(model="groq/llama-3.1-8b-instant")
            ))

        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
            self.providers.append((
                "Gemini 2.5 Flash",
                LLM(model="gemini/gemini-2.5-flash")
            ))
            self.providers.append((
                "Gemini 2.5 Flash-Lite",
                LLM(model="gemini/gemini-2.5-flash-lite")
            ))

        if not self.providers:
            raise ValueError("No API keys provided. Add a Groq and/or Gemini API key in the sidebar.")

    def _build_crew(self, llm):
        planner = Agent(
            role="Research Planner",
            goal=f"Plan research objectives for: {self.topic}. Instructions: {self.instructions}",
            backstory="Expert in breaking down complex topics into actionable research plans.",
            llm=llm,
            allow_delegation=False
        )

        researcher = Agent(
            role="Lead Researcher",
            goal=f"Gather data on: {self.topic}",
            backstory="Specialist in web research and identifying key facts.",
            tools=[self.search_tool],
            llm=llm
        )

        writer = Agent(
            role="Technical Writer",
            goal=f"Draft a report on: {self.topic}",
            backstory="Expert in creating clear, structured Markdown reports.",
            llm=llm
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

        return Crew(
            agents=[planner, researcher, writer],
            tasks=[plan_task, research_task, write_task],
            process=Process.sequential,
            verbose=True
        )

    def run(self, on_provider_switch=None):
        """
        Tries each configured provider in order. Falls through to the next
        one on rate-limit / server-availability errors. Raises the last
        error if every provider has been exhausted.

        on_provider_switch: optional callback(label: str) so the UI can
        show "Trying Gemini 2.5 Flash..." etc.
        """
        last_error = None

        for label, llm in self.providers:
            if on_provider_switch:
                on_provider_switch(label)
            try:
                crew = self._build_crew(llm)
                return crew.kickoff()
            except Exception as e:
                last_error = e
                if _is_failover_worthy(e):
                    # Try the next provider in the list
                    continue
                # Not a rate-limit/availability issue — don't keep retrying
                # with other providers, just surface it immediately.
                raise

        # Every provider failed (mostly likely all rate-limited)
        raise last_error

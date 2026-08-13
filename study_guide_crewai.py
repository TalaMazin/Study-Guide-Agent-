"""
Module 2 Phase 1 Mini-Project — CrewAI.

Build one Agent with three sequential Tasks.

Run it with your own topic:
    python study_guide_crewai.py "temperature in language models"
"""

from __future__ import annotations

import os
import sys

from dotenv import find_dotenv, load_dotenv
from crewai import Agent, Crew, LLM, Process, Task

# Load environment variables
load_dotenv()
load_dotenv(find_dotenv(usecwd=True))

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found.\n"
        "Create a .env file containing:\n"
        "    OPENROUTER_API_KEY=sk-or-...\n"
    )


# Same model/provider configuration as the LangGraph version.
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    temperature=0,
    api_key=api_key,
)


def build_crew() -> Crew:
    # One agent handles all three sequential tasks.
    teacher = Agent(
        role="Patient Study Guide Teacher",
        goal=(
            "Create accurate, understandable study material that helps "
            "beginners learn a topic clearly."
        ),
        backstory=(
            "You are a patient teacher who specializes in explaining "
            "technical topics in plain language. You must be accurate, "
            "must not invent facts or statistics, and must clearly "
            "distinguish misconceptions from facts."
        ),
        llm=llm,
        verbose=True,
    )

    # Task 1: explanation
    explain_task = Task(
        description=(
            "Explain {topic} for a beginner in 2-3 plain-language sentences. "
            "Do not invent statistics or unsupported facts. Do not create "
            "a quiz yet."
        ),
        expected_output=(
            "A clear and accurate 2-3 sentence plain-language explanation "
            "of the topic."
        ),
        agent=teacher,
    )

    # Task 2: example + misconception
    example_task = Task(
        description=(
            "Using the explanation from the previous task, create one "
            "practical example for {topic} and one common misconception. "
            "Clearly distinguish the example from the misconception, and "
            "do not present the misconception as a fact."
        ),
        expected_output=(
            "One practical example related to the topic and one clearly "
            "identified common misconception."
        ),
        agent=teacher,
        context=[explain_task],
    )

    # Task 3: final assembly + quiz
    quiz_task = Task(
        description=(
            "Assemble the complete study guide for {topic} using the "
            "earlier explanation and example. Include these sections:\n"
            "1. Explanation\n"
            "2. Example and misconception\n"
            "3. Quiz\n\n"
            "The Quiz section must contain exactly three questions followed "
            "by a matching answer key. Do not add a fourth question. "
            "Preserve the useful work from the earlier tasks."
        ),
        expected_output=(
            "A complete study guide containing an Explanation section, "
            "an Example and misconception section, and a Quiz section "
            "with exactly three questions followed by a matching answer key."
        ),
        agent=teacher,
        context=[explain_task, example_task],
    )

    # One agent, three sequential tasks.
    return Crew(
        agents=[teacher],
        tasks=[explain_task, example_task, quiz_task],
        process=Process.sequential,
        verbose=True,
        tracing=False,
    )


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]).strip() or "temperature in language models"

    crew = build_crew()
    result = crew.kickoff(inputs={"topic": topic})

    print(f"\n# Study Guide: {topic}\n")
    print(result)
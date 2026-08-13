"""
Module 2 Phase 1 Mini-Project — LangGraph.

Build the same three-task Study Guide Agent in LangGraph and CrewAI.

Run it with your own topic:
    python study_guide_langgraph.py "temperature in language models"
"""

from __future__ import annotations

import os
import sys
from typing import TypedDict

from dotenv import find_dotenv, load_dotenv
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI

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

# Same model/provider configuration used for the CrewAI version.
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


class StudyGuideState(TypedDict):
    topic: str
    explanation: str
    example: str
    quiz: str


def explain_topic(state: StudyGuideState) -> StudyGuideState:
    """Task 1: explain the topic in plain language."""

    response = llm.invoke(
        f"""
        Explain the topic "{state['topic']}" for a beginner.

        Requirements:
        - Use 2-3 plain-language sentences.
        - Be accurate and understandable.
        - Do not invent statistics or unsupported facts.
        - Do not create a quiz yet.
        """
    )

    return {
        **state,
        "explanation": response.content,
    }


def create_example(state: StudyGuideState) -> StudyGuideState:
    """Task 2: use the explanation to create an example and misconception."""

    response = llm.invoke(
        f"""
        Topic: {state['topic']}

        Explanation:
        {state['explanation']}

        Create the next section of a study guide.

        Requirements:
        - Give one practical example related to the topic.
        - Give one common misconception about the topic.
        - Clearly distinguish the example from the misconception.
        - Do not present the misconception as a fact.
        - Keep the explanation above in mind when creating the example.
        """
    )

    return {
        **state,
        "example": response.content,
    }


def create_quiz(state: StudyGuideState) -> StudyGuideState:
    """Task 3: use earlier state to create three questions and answers."""

    response = llm.invoke(
        f"""
        Create the final quiz section for a study guide about:

        Topic:
        {state['topic']}

        Explanation:
        {state['explanation']}

        Example and misconception:
        {state['example']}

        Requirements:
        - Create exactly three questions.
        - Follow the three questions with an answer key.
        - Make the answers match the questions.
        - Base the questions on the material above.
        - Do not add a fourth question.
        """
    )

    return {
        **state,
        "quiz": response.content,
    }


def build_graph():
    graph = StateGraph(StudyGuideState)

    # Register the three tasks/nodes.
    graph.add_node("explain_topic", explain_topic)
    graph.add_node("create_example", create_example)
    graph.add_node("create_quiz", create_quiz)

    # Explicit control flow.
    graph.add_edge(START, "explain_topic")
    graph.add_edge("explain_topic", "create_example")
    graph.add_edge("create_example", "create_quiz")
    graph.add_edge("create_quiz", END)

    return graph.compile()


def run_study_guide(topic: str) -> StudyGuideState:
    app = build_graph()

    initial_state: StudyGuideState = {
        "topic": topic,
        "explanation": "",
        "example": "",
        "quiz": "",
    }

    return app.invoke(initial_state)


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]).strip() or "temperature in language models"

    result = run_study_guide(topic)

    print(f"# Study Guide: {result['topic']}\n")
    print("## Explanation\n", result["explanation"])
    print("\n## Example and misconception\n", result["example"])
    print("\n## Quiz\n", result["quiz"])
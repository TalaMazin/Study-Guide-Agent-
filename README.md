# Study Guide Agent — LangGraph & CrewAI

## Module 2 — Phase 1 Mini-Project

This project builds the same three-task Study Guide Agent using **LangGraph** and **CrewAI**.

Given a topic, the agent generates:

1. A short plain-language explanation
2. A practical example and common misconception
3. A three-question quiz with an answer key

## Files

- `study_guide_langgraph.py` — LangGraph implementation
- `study_guide_crewai.py` — CrewAI implementation
- `studyguide_langgraph_output.txt` — LangGraph generated study guide
- `studyguide_crewai_output.txt` — CrewAI generated study guide

## Setup

Create a virtual environment and install the required packages:

```bash
pip install langgraph langchain-openai crewai python-dotenv

Create a .env file containing:

OPENROUTER_API_KEY=your_api_key_here

The .env file is excluded from Git.

Run

LangGraph:

python study_guide_langgraph.py "Model Context Protocol"

CrewAI:

python study_guide_crewai.py "Model Context Protocol"

A second test topic used was:

temperature in language models
Implementation
LangGraph

LangGraph uses three nodes connected explicitly:

START → explain_topic → create_example → create_quiz → END

The typed state contains the topic, explanation, example, and quiz. Each task uses the results from the previous task.

CrewAI

CrewAI uses one Agent with three sequential Tasks:

Explain → Example & Misconception → Quiz

The later tasks receive the earlier tasks as context using context=[...].

Observations

What did LangGraph make explicit?

LangGraph made the state and workflow explicit. I could clearly see how data moves between the three nodes and control the exact order of execution.

What did CrewAI automate or hide?

CrewAI automated more of the orchestration. Using sequential Tasks and task context made the workflow simpler without manually defining graph edges.

Which would I choose for this pipeline?

For this simple three-task pipeline, I would choose CrewAI because the workflow is straightforward and sequential. I would choose LangGraph for a more complex workflow where explicit state and control flow are important.

Requirements
One Agent and three Tasks in CrewAI
Three nodes and explicit edges in LangGraph
Later tasks use earlier results
Exactly three quiz questions with answers
No hardcoded study-guide answers
.env, .venv, and __pycache__ excluded from Git

**This is the version I'd submit.** It directly answers the README requirements without overd

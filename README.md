# Deep Research Agent

A comprehensive research agent built with LangChain and LangGraph that performs multi-step research on complex topics using large language models and web search capabilities.

## Overview

This system implements an autonomous research workflow that:

1. Breaks down a complex query into targeted search queries
2. Gathers information from the web using Tavily search
3. Analyzes and structures the collected research data
4. Synthesizes a draft answer based on the analysis
5. Reviews and refines the draft into a polished final answer

The agent uses Groq's LLaMA-3 70B model with varying temperature settings for different tasks in the research pipeline.

## Architecture

The research workflow is implemented as a directed graph with the following components:

- **State Management**: Tracks the research process with a `AgentState` Pydantic model
- **LLM Integration**: Uses Groq's LLaMA-3 models with tailored parameters for each task
- **Web Search**: Integrates with Tavily API for comprehensive web searches
- **Workflow Graph**: Implements a sequential process flow using LangGraph's `StateGraph`

## Components

### Language Models

- **Research LLM**: Used for generating diverse search queries (temp=0.2)
- **Analysis LLM**: Used for structured analysis of research data (temp=0.1)
- **Synthesis LLM**: Used for drafting and refining answers (temp=0.5)

### Research Pipeline

1. **Query Generation**: Breaks down the main question into 3-5 specific search queries
2. **Research Execution**: Performs web searches and collects information
3. **Data Analysis**: Structures and analyzes the collected information
4. **Answer Drafting**: Creates a comprehensive initial answer
5. **Final Review**: Refines the draft into a polished final response

### Helper Functions

- `truncate_research_results()`: Manages token limits by truncating search results
- `web_search()`: Tool function that interfaces with the Tavily API

## Usage

```python
from research_agent import run_deep_research

query = "What are the latest developments in nuclear fusion energy and what challenges remain before commercial viability?"
answer = run_deep_research(query)
print(answer)
```

## Requirements

- Python 3.8+
- LangChain and LangGraph libraries
- Pydantic
- Groq API key (for LLM access)
- Tavily API key (for web search)

## Setup

1. Install required packages:
   ```
   pip install langchain langchain-groq langchain-core langgraph pydantic tavily-python
   ```

2. Set up API keys as environment variables or pass them directly in the code:
   ```
   export GROQ_API_KEY="your_groq_api_key"
   export TAVILY_API_KEY="your_tavily_api_key"
   ```

## Future Improvements

- Add parallel research paths for increased efficiency
- Implement feedback loops to refine search queries based on initial findings
- Add memory capabilities for context across multiple research sessions
- Implement citation tracking and formatting

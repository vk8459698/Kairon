from typing import List, Dict, Any, Optional
# Updated imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
# Use pydantic directly instead of from langchain_core
from pydantic import BaseModel, Field
# Fixed Tavily imports
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain.tools import tool
from langgraph.graph import StateGraph, END
import json

# Define state schema
class AgentState(BaseModel):
    """State for the research agent system."""
    query: str = Field(..., description="The original research query")
    research_queries: List[str] = Field(default_factory=list, description="List of search queries for the research agent")
    research_results: List[Dict[str, Any]] = Field(default_factory=list, description="Research results collected")
    analyzed_data: Dict[str, Any] = Field(default_factory=dict, description="Analyzed and structured research data")
    draft_answer: Optional[str] = Field(None, description="Draft answer based on research")
    final_answer: Optional[str] = Field(None, description="Final synthesized answer")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Message history")


# Initialize language models with different temperatures for different tasks
# Using Groq models instead of OpenAI
research_llm = ChatGroq(model="llama3-70b-8192", temperature=0.2, api_key="your_grok_API")
analysis_llm = ChatGroq(model="llama3-70b-8192", temperature=0.1, api_key="your_grok_API")
synthesis_llm = ChatGroq(model="llama3-70b-8192", temperature=0.5, api_key="your_grok_API")

# Initialize Tavily client with your API key
tavily_client = TavilyClient(api_key="your_tavily_API")

# Create search tool using the decorator approach
@tool
def web_search(query: str) -> List[Dict]:
    """Searches the web for information about a specific topic."""
    search_results = tavily_client.search(query=query, search_depth="advanced", max_results=5)  # Reduced from 10 to 5
    return search_results.get("results", [])


# Define parser for structured research query generation
class SearchQueries(BaseModel):
    """Output schema for search queries generation."""
    search_queries: List[str] = Field(..., description="List of 3-5 search queries to investigate the topic thoroughly")


search_query_parser = JsonOutputParser(pydantic_object=SearchQueries)

# Research Agent: Generate search queries based on the original question
def generate_search_queries(state):
    """Generate multiple search queries to explore the topic."""
    print("Generating search queries...")
    query_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research query formulation expert.
        Your task is to break down a complex research question into 3-5 specific search queries
        that will help gather comprehensive information on the topic.
        Generate diverse queries that explore different aspects of the question.
        Format your response as a JSON with a single 'search_queries' field containing a list of strings."""),
        ("user", "{query}")
    ])
    
    query_chain = query_prompt | research_llm | search_query_parser
    result = query_chain.invoke({"query": state.query})
    
    # Fixed: Access the dictionary key, not an attribute
    return {"research_queries": result["search_queries"]}


# Research Agent: Execute search queries and gather information
def execute_research(state):
    """Execute search queries and gather information."""
    print("Executing research queries...")
    results = []
    
    for query in state.research_queries:
        print(f"  Searching: {query}")
        # Use invoke instead of calling directly to avoid deprecation warning
        search_results = web_search.invoke(query)
        results.append({
            "query": query,
            "results": search_results
        })
    
    return {"research_results": results}


# Helper function to truncate research results to avoid token limit issues
def truncate_research_results(research_results, max_results_per_query=3, max_content_length=300):
    """Truncate research results to avoid exceeding token limits."""
    truncated_results = []
    
    for query_result in research_results:
        # Make a copy of the query result
        truncated_query_result = {
            "query": query_result["query"],
            "results": []
        }
        
        # Take only the top few results for each query
        for i, result in enumerate(query_result["results"]):
            if i >= max_results_per_query:
                break
                
            # Truncate content if it's too long
            content = result.get("content", "")
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
                
            # Create a truncated result
            truncated_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": content,
                "score": result.get("score", 0)
            }
            
            truncated_query_result["results"].append(truncated_result)
            
        truncated_results.append(truncated_query_result)
        
    return truncated_results


# Analysis Agent: Analyze and structure the research data
def analyze_research_data(state):
    """Analyze and structure the collected research data."""
    print("Analyzing research data...")
    # Truncate research results to avoid token limit issues
    truncated_results = truncate_research_results(state.research_results)
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data analysis expert.
        Review the collected research information and:
        1. Identify key facts, statistics, and insights relevant to the original query
        2. Note areas of consensus and disagreement among sources
        3. Evaluate the credibility and relevance of each source
        4. Structure the information into coherent themes and categories
        5. Identify gaps in the research that may need further investigation
        
        Keep your analysis concise and focused on the most important findings.
        Provide your analysis in a structured format that will be easy for the synthesis agent to use."""),
        ("user", "Original query: {query}\n\nResearch results: {truncated_results}")
    ])
    
    analysis_chain = analysis_prompt | analysis_llm
    analysis_result = analysis_chain.invoke({
        "query": state.query,
        "truncated_results": truncated_results
    })
    
    return {"analyzed_data": {"analysis": analysis_result.content}}


# Synthesis Agent: Draft comprehensive answer - Renamed function to avoid collision with state key
def create_draft_answer(state):
    """Draft a comprehensive answer based on the analyzed research data."""
    print("Creating draft answer...")
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert synthesis writer.
        Your task is to create a comprehensive, well-structured answer to the original query
        based on the analyzed research data.
        
        Your response should:
        1. Directly answer the original question with clarity and precision
        2. Include relevant facts, data, and insights from the research
        3. Present multiple perspectives when appropriate
        4. Cite sources to support key claims
        5. Be well-organized with clear sections and logical flow
        6. Use an authoritative yet accessible writing style
        
        Focus on creating a draft that is informative, balanced, and thorough."""),
        ("user", "Original query: {query}\n\nAnalyzed research data: {analyzed_data}")
    ])
    
    synthesis_chain = synthesis_prompt | synthesis_llm
    draft_result = synthesis_chain.invoke({
        "query": state.query,
        "analyzed_data": state.analyzed_data
    })
    
    return {"draft_answer": draft_result.content}


# Final Review Agent: Refine and finalize the answer
def finalize_answer(state):
    """Review and refine the draft answer to produce the final output."""
    print("Finalizing answer...")
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert editor and quality assurance specialist.
        Review the draft answer and refine it to ensure:
        1. Complete and accurate response to the original query
        2. Clear structure and logical flow
        3. Proper citation of sources
        4. Balanced presentation of information
        5. No redundancy or unnecessary information
        6. Appropriate level of detail
        7. Professional and accessible language
        
        Produce a polished final answer that represents the best possible response to the query."""),
        ("user", "Original query: {query}\n\nDraft answer: {draft_answer}")
    ])
    
    review_chain = review_prompt | synthesis_llm
    final_result = review_chain.invoke({
        "query": state.query,
        "draft_answer": state.draft_answer
    })
    
    return {"final_answer": final_result.content, "messages": [{"role": "assistant", "content": final_result.content}]}


# Define the workflow graph
def build_research_graph():
    """Build the research workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes - renamed the node to avoid collision with state key
    workflow.add_node("generate_search_queries", generate_search_queries)
    workflow.add_node("execute_research", execute_research)
    workflow.add_node("analyze_research_data", analyze_research_data)
    workflow.add_node("create_draft_answer", create_draft_answer)  # Changed node name
    workflow.add_node("finalize_answer", finalize_answer)
    
    # Define the graph edges - the workflow sequence
    workflow.add_edge("generate_search_queries", "execute_research")
    workflow.add_edge("execute_research", "analyze_research_data")
    workflow.add_edge("analyze_research_data", "create_draft_answer")  # Updated edge
    workflow.add_edge("create_draft_answer", "finalize_answer")  # Updated edge
    workflow.add_edge("finalize_answer", END)
    
    # Set the entry point
    workflow.set_entry_point("generate_search_queries")
    
    # Compile the graph
    return workflow.compile()


# Main function to execute the research process
def run_deep_research(query: str) -> str:
    """Execute the deep research process for a given query."""
    print("Building research graph...")
    graph = build_research_graph()
    
    # Initialize the state with the query
    print("Initializing research with query:", query)
    initial_state = AgentState(query=query)
    
    # Run the graph
    print("Starting research workflow...")
    result = graph.invoke(initial_state)
    
    # Fix: Extract final_answer from the result dictionary
    if isinstance(result, dict) and "final_answer" in result:
        return result["final_answer"]
    else:
        # Print the result structure for debugging
        print("Result type:", type(result))
        print("Result keys:", result.keys() if hasattr(result, "keys") else "No keys attribute")
        # Try to find the final answer in the result
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"Key: {key}, Value type: {type(value)}")
                if isinstance(value, dict) and "final_answer" in value:
                    return value["final_answer"]
        
        # Return a fallback message if we can't find the final answer
        return "Could not extract final answer from the research results. Please check the implementation."


# Example usage
if __name__ == "__main__":
    research_question = "What are the latest developments in nuclear fusion energy and what challenges remain before commercial viability?"
    print(f"\n--- STARTING RESEARCH: '{research_question}' ---\n")
    answer = run_deep_research(research_question)
    print("\n--- FINAL RESEARCH ANSWER ---\n")
    print(answer)

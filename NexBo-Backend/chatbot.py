import os
from dotenv import load_dotenv
from typing import Sequence, List, Dict
from typing_extensions import TypedDict

# Import LangChain messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Import your finance tools and helpers
from tools.stock import stock_lookup
from tools.research import tavily_search
from tools.recommend import tools as recommendation_tools
from utils.finance_parser import parse_finance_input
from utils.jsonparser import parse_ai_response
from prompts.finance_prompt import get_finance_prompt

# Import LangGraph components and persistent memory
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Import the generative LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB-CXqCqmdcxv-WiaoNKa5mQpHw0n_A_aE")

# Initialize ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# Define a state schema for the chatbot including both messages and persistent memory.
class ChatState(TypedDict, total=False):
    messages: Sequence[BaseMessage]
    memory: List[Dict]

# Create the workflow graph with the defined state schema.
workflow = StateGraph(state_schema=ChatState)

# Define your tool list.
tools = [tavily_search, stock_lookup, *recommendation_tools]

# Bind the tools to the LLM.
model_with_tools = llm.bind_tools(tools=tools)

# Define the model call function with persistent memory and tool integration.
def call_model(state: ChatState):
    # Retrieve conversation messages and persistent memory; initialize if missing.
    messages = state.get("messages", [])
    persistent_memory = state.get("memory", [])
    if not persistent_memory:
        # Initialize memory with the finance prompt.
        persistent_memory = [{"role": "system", "content": get_finance_prompt()}]
    
    # Add a system message that clearly lists only the available tools.
    system_message = SystemMessage(
        content=(
            "You are a helpful financial advisor. "
            "Remember previous conversations and maintain context. "
            "You have access only to the following tools: "
            "stock_lookup (for real-time stock data), "
            "tavily_search (for financial research), and "
            "recommendation_tools (for recommending stocks based on user preferences and budget). "
            "Do not reference any other tools."
        )
    )
    
    # Process the latest human message.
    if messages and getattr(messages[-1], "role", None) == "human":
        user_input = messages[-1].content.lower()
        parsed_params = parse_finance_input(user_input)
        print("Parsed Parameters:", parsed_params)
        
        # If user asks which tools are available, override with our own list.
        if "what tools" in user_input or "which tools" in user_input:
            tool_list_message = {"role": "system", "content": "Available tools: stock_lookup, tavily_search, recommendation_tools."}
            persistent_memory.append(tool_list_message)
        # If recommendation criteria exist, call the recommendation tool.
        elif parsed_params.get("criteria", {}).get("sector"):
            # Determine market based on currency (INR for India, otherwise default to US).
            market = "IN" if (parsed_params.get("currency") or "INR").upper() == "INR" else "US"
            recommendations = None
            # Loop through recommendation tools to call the one that provides recommendations.
            for tool in recommendation_tools:
                if hasattr(tool, "get_recommendations_tool"):
                    recommendations = tool.get_recommendations_tool(
                        budget=parsed_params.get("budget"),
                        market=market,
                        currency=parsed_params.get("currency") or "INR",
                        criteria=parsed_params.get("criteria"),
                        limit=5
                    )
                    break
            if recommendations is not None:
                tool_message = {"role": "system", "content": f"Recommendations: {recommendations}"}
                persistent_memory.append(tool_message)
        else:
            # Otherwise, add structured parameters for context.
            structured_message = {"role": "system", "content": f"Structured Parameters: {parsed_params}"}
            persistent_memory.append(structured_message)
    
    # Combine system message, persistent memory, and current conversation messages.
    all_messages = [system_message] + persistent_memory + list(messages)
    
    # Invoke the bound model with tools.
    response = model_with_tools.invoke(all_messages)
    
    # Update persistent memory with the conversation history and the latest response.
    persistent_memory.extend([m.__dict__ if hasattr(m, '__dict__') else m for m in messages])
    persistent_memory.append(response.__dict__ if hasattr(response, '__dict__') else response)
    
    return {"messages": messages + [response], "memory": persistent_memory}

# Create a tool node using only the imported tools.
tool_node = ToolNode(tools=tools)

# Define a function to conditionally decide if the tool node should be invoked.
def should_continue(state: ChatState):
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "tools"
    return END

# Add nodes and edges to the workflow.
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

# Initialize persistent memory saver.
memory_saver = MemorySaver()
app = workflow.compile(checkpointer=memory_saver)

# Chat function with persistent memory.
def chat():
    chat_history: List[BaseMessage] = []
    # Global memory state for persistence across turns.
    memory_state: List[Dict] = []
    config = {"configurable": {"thread_id": "finance_chat_1"}}
    
    while True:
        user_input = input("\n💬 You: ")
        if user_input.lower() == "exit":
            break
        
        # Append the new user message to existing chat history.
        current_messages = chat_history + [HumanMessage(content=user_input)]
        config["configurable"]["thread_id"] = "finance_chat_1"
        
        # Pass both current conversation and persistent memory.
        output = app.invoke({"messages": current_messages, "memory": memory_state}, config=config)
        
        # Update both chat history and persistent memory.
        chat_history = output["messages"]
        memory_state = output["memory"]
        try:
            response = parse_ai_response(chat_history[-1].content)
        except:
            response = chat_history[-1].content
        # Print the assistant's latest response.
        print("\n🤖Assistant:", response)


if __name__ == "__main__":
    chat()

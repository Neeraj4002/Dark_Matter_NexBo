import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from tools.stock import stock_lookup
from tools.research import tavily_search
from tools.recommend import tools as recommendation_tools
from utils.finance_parser import parse_finance_input
from utils.jsonparser import extract_json_from_response
from prompts.finance_prompt import get_finance_prompt

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB-CXqCqmdcxv-WiaoNKa5mQpHw0n_A_aE")

# Initialize ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# Define the state schema for the conversation
from typing import Sequence, List, Dict
from typing_extensions import TypedDict

class ChatState(TypedDict, total=False):
    messages: Sequence[BaseMessage]
    memory: List[Dict]

# Create the workflow graph and bind tools
workflow = StateGraph(state_schema=ChatState)
tools = [tavily_search, stock_lookup, *recommendation_tools]
model_with_tools = llm.bind_tools(tools=tools)

def call_model(state: ChatState):
    messages = state.get("messages", [])
    persistent_memory = state.get("memory", [])
    if not persistent_memory:
        persistent_memory = [{"role": "system", "content": get_finance_prompt()}]
    
    # Add a system message listing available tools
    from langchain_core.messages import SystemMessage
    system_message = SystemMessage(
        content=(
            "You are a helpful financial advisor. "
            "Remember previous conversations and maintain context. "
            "You have access only to the following tools: "
            "stock_lookup, tavily_search, and recommendation_tools. "
            "Do not reference any other tools."
        )
    )
    
    # Process the latest human message
    if messages and getattr(messages[-1], "role", None) == "human":
        user_input = messages[-1].content.lower()
        parsed_params = parse_finance_input(user_input)
        print("Parsed Parameters:", parsed_params)
        
        # If user asks which tools are available, provide the list
        if "what tools" in user_input or "which tools" in user_input:
            tool_list_message = {"role": "system", "content": "Available tools: stock_lookup, tavily_search, recommendation_tools."}
            persistent_memory.append(tool_list_message)
        # If recommendation criteria exist, call the recommendation tool
        elif parsed_params.get("criteria", {}).get("sector"):
            market = "IN" if (parsed_params.get("currency") or "INR").upper() == "INR" else "US"
            recommendations = None
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
            structured_message = {"role": "system", "content": f"Structured Parameters: {parsed_params}"}
            persistent_memory.append(structured_message)
    
    # Combine system message, persistent memory, and conversation messages
    all_messages = [system_message] + persistent_memory + list(messages)
    
    # Invoke the LLM with tools
    response = model_with_tools.invoke(all_messages)
    
    # Update persistent memory with conversation history and the latest response
    persistent_memory.extend([m.__dict__ if hasattr(m, '__dict__') else m for m in messages])
    persistent_memory.append(response.__dict__ if hasattr(response, '__dict__') else response)
    
    return {"messages": messages + [response], "memory": persistent_memory}

# Create a tool node for additional processing if needed
tool_node = ToolNode(tools=tools)

def should_continue(state: ChatState):
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "tools"
    return END

# Build the workflow graph
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

memory_saver = MemorySaver()
workflow_app = workflow.compile(checkpointer=memory_saver)

# Global conversation state
chat_history = []
memory_state = []

# Initialize FastAPI and configure CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat_endpoint(request: Request):
    global chat_history, memory_state
    try:
        data = await request.json()
        user_message = data.get("message", "")
        is_recommend = data.get("recommend", False)

        # Append recommendation trigger if needed
        if is_recommend:
            user_message += " recommend stocks for my budget "
        
        # Add the new human message to the conversation
        chat_history.append(HumanMessage(content=user_message))
        config = {"configurable": {"thread_id": "finance_chat_1"}}
        
        # Call the workflow to process the conversation
        output = workflow_app.invoke({"messages": chat_history, "memory": memory_state}, config=config)
        chat_history = output["messages"]
        memory_state = output["memory"]

        # Extract the assistant's response
        assistant_response = chat_history[-1].content

        # If recommendations were requested, try to parse and return structured JSON data
        if is_recommend:
            try:
                print("Assistant Response:", assistant_response)
                parsed_json = extract_json_from_response(assistant_response)
                print("Parsed JSON:", parsed_json)
                
                if parsed_json and parsed_json.get("recommendations"):
                    return JSONResponse(content={
                        "response": "Here are your recommendations.",
                        "recommendations": parsed_json.get("recommendations", []),
                        "status": "success"
                    })
                else:
                    # If no recommendations found in parsed JSON, return the AI response
                    return JSONResponse(content={
                        "response": assistant_response,
                        "recommendations": [],
                        "status": "partial",
                        "message": "Could not parse recommendations, but showing AI response"
                    })
                    
            except Exception as e:
                print(f"Parsing error: {str(e)}")
                # Return both the error and the AI response
                return JSONResponse(content={
                    "response": assistant_response,
                    "recommendations": [],
                    "status": "error",
                    "message": "Failed to parse recommendations, showing AI response instead",
                    "error_details": str(e)
                })
        else:
            return JSONResponse(content={
                "response": assistant_response,
                "status": "success"
            })
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(e),
                "status": "error"
            }
        )

from BookAgent import initialize_system_for_book, retrieve_relevant_chunks, generate_response_with_gemini, BOOK_INFO, BOOK_SYSTEMS
import json
from pydantic import BaseModel
from fastapi import HTTPException


class ChatRequest(BaseModel):
    message: str
    bookName: str

@app.post("/bookchat")
async def book_chat(request: ChatRequest):
    """
    Endpoint to handle chat queries for a given book.
    Expected JSON payload: {"message": "Hi", "bookName": "The Psychology of Money"}
    """
    book_name = request.bookName
    message = request.message

    if book_name not in BOOK_INFO:
        raise HTTPException(status_code=400, detail="Invalid bookName provided.")

    # Get the file info for the requested book
    book_files = BOOK_INFO[book_name]
    pdf_path = book_files["pdf"]
    embeddings_file = book_files["embeddings"]
    faiss_index_file = book_files["faiss_index"]

    # Initialize system for the book if not already done
    if book_name not in BOOK_SYSTEMS:
        try:
            BOOK_SYSTEMS[book_name] = initialize_system_for_book(book_name, pdf_path, embeddings_file, faiss_index_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error initializing system for the book.")

    book_data = BOOK_SYSTEMS[book_name]
    try:
        retrieved_chunks = retrieve_relevant_chunks(message, book_data["chunks"], book_data["faiss_index"], top_k=3)
        answer = generate_response_with_gemini(message, retrieved_chunks)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing chat query.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5500))
    uvicorn.run(app, host="0.0.0.0", port=port)

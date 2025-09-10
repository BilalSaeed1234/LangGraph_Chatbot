from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import os
import requests
from langsmith import Client
import time

# Load environment variables
load_dotenv()

# Initialize LangSmith client
langsmith_client = Client()

# ---- Load API Key from .env ----
gemini_api_key = os.getenv("GEMINI_API_KEY")

# ---- Define State ----
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---- Gemini LLM ----
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key
)

# -------------------
# Custom Search Tools with better error handling
# -------------------

@tool
def custom_web_search(query: str, num_results: int = 3) -> str:
    """
    Perform a custom web search using Google search API.
    Returns summarized results from top websites.
    """
    try:
        # Try to import googlesearch
        try:
            import googlesearch
        except ImportError:
            return "Error: Please install googlesearch package with: pip install googlesearch-python"
        
        # Perform Google search
        search_results = []
        for result in googlesearch.search(query, num_results=num_results, advanced=True):
            search_results.append({
                "title": result.title,
                "url": result.url,
                "description": result.description
            })
        
        # Summarize the results
        summary = f"Web search results for '{query}':\n\n"
        for i, result in enumerate(search_results, 1):
            summary += f"{i}. {result['title']}\n"
            summary += f"   URL: {result['url']}\n"
            if result['description']:
                summary += f"   Description: {result['description']}\n\n"
            else:
                summary += "\n"
        
        return summary
        
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for information on a specific topic.
    Returns a summary from Wikipedia.
    """
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code == 200:
            return f"Wikipedia results for '{query}':\n\nTitle: {data.get('title', 'N/A')}\n\nSummary: {data.get('extract', 'No summary available')}\n\nURL: {data.get('content_urls', {}).get('desktop', {}).get('page', 'N/A')}"
        else:
            return f"Wikipedia page for '{query}' not found."
            
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"

@tool
def news_search(topic: str, num_articles: int = 3) -> str:
    """
    Search for recent news articles on a specific topic.
    """
    try:
        # Try to import googlesearch
        try:
            import googlesearch
        except ImportError:
            return "Error: Please install googlesearch package with: pip install googlesearch-python"
        
        # Using simple Google search for news
        news_results = []
        search_query = f"{topic} news today 2024"
        
        for result in googlesearch.search(search_query, num_results=num_articles, advanced=True):
            news_results.append({
                "title": result.title,
                "url": result.url,
                "source": result.url.split('/')[2] if '/' in result.url else result.url
            })
        
        summary = f"Recent news about '{topic}':\n\n"
        for i, result in enumerate(news_results, 1):
            summary += f"{i}. {result['title']}\n"
            summary += f"   Source: {result['source']}\n"
            summary += f"   URL: {result['url']}\n\n"
        
        return summary if news_results else f"No recent news found about '{topic}'"
        
    except Exception as e:
        return f"News search failed: {str(e)}"

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=RD42O9QHRMW054U8"
        r = requests.get(url, timeout=10)
        data = r.json()
        
        if 'Global Quote' in data and data['Global Quote']:
            quote = data['Global Quote']
            return {
                "symbol": symbol,
                "price": quote.get('05. price', 'N/A'),
                "change": quote.get('09. change', 'N/A'),
                "change_percent": quote.get('10. change percent', 'N/A')
            }
        else:
            return {"error": f"Stock symbol '{symbol}' not found or API limit reached"}
            
    except Exception as e:
        return {"error": f"Stock price lookup failed: {str(e)}"}

# Bind tools to LLM
tools = [custom_web_search, wikipedia_search, news_search, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

# ---- Define Chat Node ----
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

# ---- Tool Node ----
tool_node = ToolNode(tools)

# ---- Build Graph ----
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

# ---- SQLite Checkpointer ----
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ---- Export chatbot ----
chatbot = graph.compile(checkpointer=checkpointer)

# ---- Utility functions ----
def retrieve_all_threads():
    """Retrieve all conversation thread IDs from the database"""
    try:
        all_threads = set()
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config['configurable']['thread_id']
            all_threads.add(thread_id)
        return list(all_threads)
    except Exception as e:
        print(f"Error retrieving threads: {e}")
        return []

def get_thread_conversations(thread_id):
    """Get all messages from a specific thread"""
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        return state.values['messages']
    except:
        return []
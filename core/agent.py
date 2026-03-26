import os
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, Annotated, Sequence, Optional
import operator
from datetime import datetime

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END

from memory.models import MemoryItem
from memory.router import classifier
from memory.longterm import LongTermMemory
from memory.session import SessionMemory
from memory.scratch import ScratchMemory
from memory.crypto import load_keypair, sign_item, sign_session_item


llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)


ddg_search = DuckDuckGoSearchRun()


# agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str
    retrieved_context: str


# web search tool
@tool
def web_search(query: str) -> str:
    """
    Search the web for current cybersecurity threats, vulnerabilities, research papers, 
    or technical documentation. Use this tool when you need up-to-date threat intelligence,
    security advisories, or to research specific cybersecurity topics, attack techniques,
    or mitigation strategies. Returns relevant search results with URLs and technical snippets.
    """
    try:
        return ddg_search.run(query)
    except Exception as e:
        return f"Error performing web search: {e}"


# web scraper tool
@tool
def web_scraper(url: str) -> str:
    """
    Extract technical content from cybersecurity websites, security blogs, vulnerability 
    reports, or research papers. Use this tool when you have a specific URL containing 
    security advisories, technical documentation, malware analysis reports, or threat 
    intelligence that needs detailed analysis. Returns the full technical content for 
    security research and analysis.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"Error: received HTTP {response.status_code} for URL: {url}"
        soup = BeautifulSoup(response.content, "html.parser", from_encoding="utf-8")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except requests.exceptions.MissingSchema:
        return f"Error: invalid URL format — '{url}'. Include the scheme (https://)."
    except requests.exceptions.ConnectionError:
        return f"Error: could not connect to '{url}'."
    except Exception as e:
        return f"Error scraping website: {e}"


# document parser tool
@tool
def document_parser(file_path: str) -> str:
    """
    Parse and extract content from security reports, technical documentation, or 
    research papers in PDF format. Use this tool when you need to analyze security 
    advisories, vulnerability reports, malware analysis documents, or cybersecurity 
    research papers. Returns the full technical content extracted from the PDF for 
    detailed security analysis.
    """
    if not os.path.exists(file_path):
        return f"Error: file not found at path '{file_path}'."
    if not file_path.lower().endswith(".pdf"):
        return f"Error: '{file_path}' is not a PDF file."
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        if not pages:
            return "Error: PDF loaded but contained no extractable text."
        full_text = " ".join([page.page_content for page in pages])
        return full_text
    except Exception as e:
        return f"Error parsing document: {e}"


# long-term memory initialization with key loading
try:
    private_key, public_key = load_keypair()
    longterm_memory = LongTermMemory(public_key)
except ValueError as e:
    print(f"Error: {e}")
    raise SystemExit("Error: Memory signing disabled due to invalid keys.")


# session and scratch memory initialization
session_memory = SessionMemory()
scratch_memory = ScratchMemory()


tools = [web_search, web_scraper, document_parser]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}


def retrieve_memory_node(state: AgentState) -> dict:
    """Pull context from all memory tiers."""
    latest_message = state["messages"][-1].content
    user_id = state["user_id"]

    context_pieces = []

    # scratch memory
    try:
        scratch_items = scratch_memory.get_all()
        user_scratch = [item for item in scratch_items if item.user_id == user_id]
        relevant_scratch = [
            item.content
            for item in user_scratch
            if latest_message.lower() in item.content.lower()
        ]
        if relevant_scratch:
            context_pieces.append(
                "--- SCRATCH MEMORY ---\n" + "\n".join(relevant_scratch[:3])
            )
    except Exception as e:
        print(f"Error searching scratch memory: {e}")

    # session memory
    try:
        session_items = session_memory.search(latest_message, n_results=3)
        user_session = [item for item in session_items if item.user_id == user_id]
        if user_session:
            context_pieces.append(
                "--- SESSION MEMORY ---\n"
                + "\n".join([item.content for item in user_session])
            )
    except Exception as e:
        print(f"Error searching session memory: {e}")

    # long-term memory
    if longterm_memory:
        try:
            longterm_items = longterm_memory.search(latest_message, n_results=3)
            user_longterm = [item for item in longterm_items if item.user_id == user_id]
            if user_longterm:
                context_pieces.append(
                    "--- LONGTERM MEMORY ---\n"
                    + "\n".join([item.content for item in user_longterm])
                )
        except Exception as e:
            print(f"Error searching long-term memory: {e}")

    return {"retrieved_context": "\n\n".join(context_pieces)}


def orchestrator_node(state: AgentState) -> dict:
    """LLM evaluates context and decides to answer directly or invoke a tool."""
    context = state.get("retrieved_context", "")

    system_prompt = (
        f"You are a professional cybersecurity research agent with broad expertise across "
        f"multiple domains including network security, application security, cryptography, "
        f"threat intelligence, malware analysis, and security compliance. Your role is to "
        f"provide comprehensive research and analysis on cybersecurity topics.\n\n"
        f"Here is relevant context from your secure memory hierarchy:\n\n{context}\n\n"
        f"Use this contextual information to inform your research. When analyzing security topics, "
        f"provide detailed technical insights, current threat landscape information, and "
        f"practical recommendations. If you need additional information, use your available "
        f"tools to gather current data, technical documentation, or threat intelligence. "
        f"Maintain a professional, research-oriented approach suitable for cybersecurity "
        f"analysis and reporting."
    )

    messages_for_llm = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm_with_tools.invoke(messages_for_llm)
    return {"messages": [response]}


def execute_tools_node(state: AgentState) -> dict:
    """Execute whichever tools the LLM requested."""
    last_message = state["messages"][-1]
    tool_outputs = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        selected_tool = tool_map.get(tool_name)

        if selected_tool is None:
            result = f"Error: unknown tool '{tool_name}'."
        else:
            result = selected_tool.invoke(tool_args)

        tool_outputs.append(
            ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": tool_outputs}


def store_to_memory_node(state: AgentState) -> dict:
    """Store tool outputs and conversation to appropriate memory tiers using router."""
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            memory_item = MemoryItem.create(
                content=message.content,
                tier="SCRATCH",  # dummy tier
                trust_score=0.7,  # dummy trust score
                user_id=state["user_id"],
            )

            # dummy router
            classifier(memory_item)

            if memory_item.tier == "SESSION":
                from datetime import timedelta

                # 168 hours = 1 week
                memory_item.expires_at = datetime.utcnow() + timedelta(hours=168)

            # sign if keys
            if private_key:
                if memory_item.tier == "LONGTERM":
                    sign_item(memory_item, private_key)
                elif memory_item.tier == "SESSION":
                    sign_session_item(memory_item)

            # store based on tier
            try:
                if memory_item.tier == "LONGTERM" and longterm_memory:
                    longterm_memory.add(memory_item)
                    print(f"Stored to long-term memory: {memory_item.id[:8]}...")
                elif memory_item.tier == "SESSION":
                    session_memory.add(memory_item)
                    print(f"Stored to session memory: {memory_item.id[:8]}...")
                elif memory_item.tier == "SCRATCH":
                    scratch_memory.add(memory_item)
                    print(f"Stored to scratch memory: {memory_item.id[:8]}...")
            except Exception as e:
                print(f"Failed to store to {memory_item.tier.lower()} memory: {e}")

            break

    return {}


def should_continue(state: AgentState) -> str:
    """Route to tool execution if the LLM issued tool calls; otherwise end."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "Execute_Tools_Node"
    return END


# build graph
workflow = StateGraph(AgentState)

workflow.add_node("Retrieve_Memory_Node", retrieve_memory_node)
workflow.add_node("Orchestrator_Node", orchestrator_node)
workflow.add_node("Execute_Tools_Node", execute_tools_node)
workflow.add_node("Store_To_Memory_Node", store_to_memory_node)

workflow.set_entry_point("Retrieve_Memory_Node")

workflow.add_edge("Retrieve_Memory_Node", "Orchestrator_Node")

workflow.add_conditional_edges(
    "Orchestrator_Node",
    should_continue,
    {
        "Execute_Tools_Node": "Execute_Tools_Node",
        END: END,
    },
)

workflow.add_edge("Execute_Tools_Node", "Store_To_Memory_Node")
workflow.add_edge("Store_To_Memory_Node", "Orchestrator_Node")

research_agent = workflow.compile()


def process_discord_message(user_id: str, message_content: str) -> str:
    human_message = HumanMessage(content=message_content)

    initial_state: AgentState = {
        "messages": [human_message],
        "user_id": user_id,
        "retrieved_context": "",
    }

    try:
        result = research_agent.invoke(initial_state)

        final_messages = result["messages"]
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                return msg.content

        return "I'm sorry, I couldn't generate a response."

    except Exception as e:
        return f"Error processing message: {str(e)}"
    finally:
        scratch_memory.clear()
        session_memory.purge_expired()


def get_user_memory_summary(user_id: str) -> str:
    summary = f"Memory summary for user {user_id}:\n"

    # session memory
    try:
        session_items = session_memory.get_active()
        user_session = [item for item in session_items if item.user_id == user_id]
        summary += f"Session items: {len(user_session)}\n"
        if user_session:
            summary += "--- SESSION MEMORY CONTENT ---\n"
            for i, item in enumerate(user_session[:5]):
                summary += f"{i+1}. {item.content[:100]}{'...' if len(item.content) > 100 else ''}\n"
            if len(user_session) > 5:
                summary += f"... and {len(user_session) - 5} more items\n"
        summary += "\n"
    except Exception as e:
        summary += f"Session error: {e}\n"

    # long-term memory
    try:
        longterm_items = longterm_memory.get_all_verified()
        user_longterm = [item for item in longterm_items if item.user_id == user_id]
        summary += f"Long-term items: {len(user_longterm)}\n"
        if user_longterm:
            summary += "--- LONGTERM MEMORY CONTENT ---\n"
            for i, item in enumerate(user_longterm[:5]):
                summary += f"{i+1}. {item.content[:100]}{'...' if len(item.content) > 100 else ''}\n"
            if len(user_longterm) > 5:
                summary += f"... and {len(user_longterm) - 5} more items\n"
    except Exception as e:
        summary += f"Long-term error: {e}\n"

    return summary

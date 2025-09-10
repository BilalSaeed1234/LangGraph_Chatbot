import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from langsmith import Client

# Initialize LangSmith client
langsmith_client = Client()

# ----------------- Set page config FIRST -----------------
st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖")

# ----------------- Utility functions -----------------
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    st.rerun()

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        return state.values['messages']
    except:
        return []  # Return empty list if no conversation exists

# ----------------- Session Setup -----------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

# ----------------- Sidebar (New/Resume Chat) -----------------
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# Display all available threads
for thread_id in st.session_state['chat_threads'][::-1]:
    display_text = f"💬 {thread_id[:8]}..."
    if st.sidebar.button(display_text, key=f"thread_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            else:
                role = "system"
            temp_messages.append({'role': role, 'content': str(msg.content)})

        st.session_state['message_history'] = temp_messages
        st.rerun()

# Clear conversations button
if st.sidebar.button("🗑️ Clear All Conversations"):
    st.session_state['chat_threads'] = []
    st.session_state['message_history'] = []
    st.session_state['thread_id'] = generate_thread_id()
    st.rerun()

# ----------------- Main UI -----------------
st.title("🤖 Bilal Chatbot")
st.caption(f"Conversation ID: {st.session_state['thread_id']}")

# Load conversation history
for message in st.session_state['message_history']:
    if message['role'] == 'tool':
        with st.chat_message("assistant"):
            st.info(f"🔧 Tool used: {message['content']}")
    else:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    
    # Display user message
    with st.chat_message('user'):
        st.markdown(user_input)

    # Prepare all messages for the chatbot
    all_messages = []
    for msg in st.session_state['message_history']:
        if msg['role'] == 'user':
            all_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            all_messages.append(AIMessage(content=msg['content']))
        elif msg['role'] == 'tool':
            all_messages.append(ToolMessage(content=msg['content'], tool_call_id="auto"))

    # Configuration
    config = {
        'configurable': {
            'thread_id': st.session_state['thread_id']
        }
    }

    # Stream assistant response with tool handling
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        tool_status = None

        try:
            for message_chunk, metadata in chatbot.stream(
                {'messages': all_messages},
                config=config,
                stream_mode="messages"
            ):
                # Handle tool messages
                if isinstance(message_chunk, ToolMessage):
                    if tool_status is None:
                        tool_status = st.status("🔧 Using tool...", expanded=False)
                    tool_status.update(label=f"🔧 Tool: {getattr(message_chunk, 'name', 'unknown')}", state="running")
                
                # Handle AI messages
                elif isinstance(message_chunk, AIMessage) and hasattr(message_chunk, 'content'):
                    text_chunk = message_chunk.content
                    full_response += text_chunk
                    placeholder.markdown(full_response + "▌")
            
            # Finalize tool status if any tool was used
            if tool_status is not None:
                tool_status.update(label="✅ Tool execution completed", state="complete", expanded=False)
            
            placeholder.markdown(full_response)
            
            # Save assistant message to history
            st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            placeholder.markdown(error_msg)
            st.session_state['message_history'].append({'role': 'assistant', 'content': error_msg})

# Display current thread count
st.sidebar.info(f"Total conversations: {len(st.session_state['chat_threads'])}")
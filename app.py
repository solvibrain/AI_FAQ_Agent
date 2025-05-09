import gradio as gr
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage # Added for history conversion

from langchain_utils import (
    get_google_generativeai_llm,
    create_rag_chain,
    DEFAULT_RAG_PROMPT_TEMPLATE,
    FAQResponse # Import the Pydantic model
)
from utils import (
    process_faqs_and_setup_vector_store,
    ESCALATION_MESSAGE # Using the predefined escalation message
)

# Load environment variables
load_dotenv()

# --- Global Variables --- #
# Initialize components once to be reused across Gradio calls
RAG_CHAIN = None

def initialize_agent():
    """Initializes the RAG chain and other necessary components."""
    global RAG_CHAIN
    if RAG_CHAIN is None:
        print("Initializing RAG agent...")
        # 1. Process FAQs and set up vector store
        db = process_faqs_and_setup_vector_store("data/faq_data.csv") 
        if not db:
            print("ERROR: Failed to initialize vector store in initialize_agent.") 
            return False
        retriever = db.as_retriever()

        # 2. Initialize LLM
        llm = get_google_generativeai_llm()
        if not llm:
            print("ERROR: Failed to initialize LLM. Check GOOGLE_API_KEY.")
            return False

        # 3. Create RAG chain
        RAG_CHAIN = create_rag_chain(retriever, llm, DEFAULT_RAG_PROMPT_TEMPLATE)
        if not RAG_CHAIN: 
            print("ERROR: Failed to create RAG chain in initialize_agent.")
            return False
        print("RAG agent initialized successfully.")
        return True
    print("INFO: Agent was already initialized.")
    return True 

def answer_question(user_input, history):
    """
    Processes the user's query using the RAG chain, considering chat history, and checks for escalation.
    Args:
        user_input (str): The latest message from the user.
        history (list[dict]): Gradio chat history with 'role' and 'content' keys
    Returns:
        str: The AI agent's response.
    """
    print(f"\n--- Received user_input: '{user_input}' ---")
    print(f"--- Current history: {history} ---")

    # Convert Gradio history (now list of dicts with 'role' and 'content') to Langchain message history
    langchain_history = []
    if history: # Ensure history is not None or empty
        for message_dict in history:
            role = message_dict.get("role")
            content = message_dict.get("content")
            if role == "user":
                langchain_history.append(HumanMessage(content=content))
            elif role == "assistant": # Gradio uses 'assistant' for bot messages with type='messages'
                langchain_history.append(AIMessage(content=content))
            # else: # Handle other roles or malformed entries if necessary
            #     print(f"WARN: Unknown role or malformed message in history: {message_dict}")

    print(f"--- Converted Langchain history: {langchain_history} ---")

    if not RAG_CHAIN:
        print("INFO: RAG_CHAIN is None. Attempting to initialize agent on-the-fly...") 
        if not initialize_agent():
            print("ERROR: Agent not initialized after on-the-fly attempt in answer_question.") 
            return "Error: Agent not initialized. Please check API key and configurations."
        if not RAG_CHAIN: 
             print("CRITICAL ERROR: RAG_CHAIN still None after successful on-the-fly initialization call.") 
             return "Error: Agent could not be initialized. Critical component missing."
        print("INFO: Agent initialized successfully on-the-fly.") 

    try:
        print("INFO: Invoking RAG_CHAIN with question and history...") 
        rag_output = RAG_CHAIN.invoke({"question": user_input, "chat_history": langchain_history})
        print(f"INFO: RAG_CHAIN output received: {rag_output}") 
        llm_response_data = rag_output.get('answer') 

        if not llm_response_data:
            print(f"WARNING: RAG chain returned unexpected output or no 'answer': {rag_output}")
            return "Sorry, I encountered an issue processing your request. The RAG chain output was not as expected."
        
        print(f"INFO: Extracted llm_response_data: {llm_response_data}")

        human_handoff_needed = False
        answer_content = "Could not determine an answer."
        is_in_scope = True # Assume in scope unless LLM says otherwise

        if isinstance(llm_response_data, FAQResponse):
            human_handoff_needed = llm_response_data.human_handoff_needed
            answer_content = llm_response_data.answer_content
            is_in_scope = llm_response_data.is_in_scope
        elif isinstance(llm_response_data, dict):
            human_handoff_needed = llm_response_data.get('human_handoff_needed', False)
            answer_content = llm_response_data.get('answer_content', "Could not determine an answer.")
            is_in_scope = llm_response_data.get('is_in_scope', True)
        else:
            # Fallback if the structure isn't as expected
            print(f"Unexpected llm_response_data type: {type(llm_response_data)}. Content: {llm_response_data}")
            # Try to parse as string if it's just raw text output (older version behavior)
            answer_content = str(llm_response_data) 
            # We can't reliably check human_handoff_needed or is_in_scope here
            # For now, we'll just return the answer content directly if this happens
            # and not attempt escalation based on structured output.
            return f"Unexpected response format from AI: {answer_content}"

        print(f"INFO: Parsed - human_handoff: {human_handoff_needed}, is_in_scope: {is_in_scope}, answer: '{answer_content[:100]}...' ")

        if human_handoff_needed or not is_in_scope:
            if human_handoff_needed:
                print("INFO: Escalation triggered due to human_handoff_needed=True.")
                return ESCALATION_MESSAGE
            else: 
                print(f"INFO: Out of scope or conversational. Returning LLM's direct answer_content: '{answer_content[:100]}...' ")
                return answer_content
        else: # In scope and no human handoff needed
            print(f"INFO: In scope. Returning LLM's answer_content: '{answer_content[:100]}...' ")
            return answer_content

    except Exception as e:
        print(f"ERROR: Error during RAG chain invocation or processing: {e}")
        import traceback
        traceback.print_exc()
        return f"An error occurred while processing your request: {e}. Please check the logs."

# --- Gradio Interface --- #
# Use gr.ChatInterface for a conversational UI
iface = gr.ChatInterface(
    fn=answer_question,
    title="AI FAQ Agent (Conversational)",
    description="Ask questions about our services. The agent will try to answer or escalate if needed. Supports multi-turn conversation.",
    chatbot=gr.Chatbot(height=400, type='messages'), # Set type='messages'
    textbox=gr.Textbox(placeholder="Ask your question here...", container=False, scale=7),
)

if __name__ == "__main__":
    print("INFO: Attempting to initialize agent at startup...") 
    if not initialize_agent():
        print("ERROR: Failed to initialize the agent at startup. Gradio app backend might be unresponsive.")
    else:
        print("INFO: Agent initialized at startup. Starting Gradio app...")
        iface.launch(share=True)

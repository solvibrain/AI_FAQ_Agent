# rag_chain_tester.py
import os
import json # Keep for potential direct LLM string debugging if needed, but not for main flow
import time # Import time module
from dotenv import load_dotenv

from utils import (
    process_faqs_and_setup_vector_store, 
    ESCALATION_MESSAGE
)
from langchain_utils import (
    get_google_generativeai_llm, 
    create_rag_chain,
    FAQResponse # Import the Pydantic model for type checking if desired
)

# --- Configuration ---
FAQ_DATA_PATH = "faq_data.csv"
VECTOR_STORE_PERSIST_DIR = "chroma_db_faq"
SIMILARITY_SEARCH_K = 3 

def main():
    print("--- Starting RAG Chain Tester ---")
    load_dotenv()

    vector_store = process_faqs_and_setup_vector_store(
        csv_file_path=FAQ_DATA_PATH,
        persist_directory=VECTOR_STORE_PERSIST_DIR
    )
    if not vector_store:
        print("Failed to setup vector store. Exiting.")
        return

    retriever = vector_store.as_retriever(search_kwargs={"k": SIMILARITY_SEARCH_K})
    print(f"Retriever created with k={SIMILARITY_SEARCH_K}")

    llm = get_google_generativeai_llm()
    if not llm:
        print("Failed to initialize LLM. Exiting.")
        return

    rag_chain = create_rag_chain(retriever=retriever, llm=llm)
    if not rag_chain:
        print("Failed to create RAG chain. Exiting.")
        return

    test_questions = [
        "What are your business hours?",
        "How can I reset my password?",
        "I want to speak to a human agent immediately!",
        "This is unacceptable, I demand to talk to your manager!",
        "What is the meaning of life?",
        "Tell me about your refund policy.",
        "My account is locked and I'm really angry!",
        "Can you explain quantum physics?",
        "I need help with a very sensitive legal matter regarding my account.",
        "Thanks, that was helpful!"
    ]

    for i, question in enumerate(test_questions):
        print(f"\n--- Test {i+1}: Querying RAG chain with: '{question}' ---")
        
        try:
            # rag_chain.invoke(question) returns a dictionary like: 
            # {'question': '...', 'context': [...], 
            #  'answer': FAQResponse(human_handoff_needed=..., is_in_scope=..., answer_content=...) }
            # The 'answer' key now contains an instance of our FAQResponse Pydantic model.
            chain_output = rag_chain.invoke(question)
            
            if 'answer' not in chain_output:
                print(f"  Error: 'answer' key missing in RAG chain output.")
                print(f"  Raw RAG chain output: {chain_output}")
                continue

            # The 'answer' should be our FAQResponse object (or a dict that Pydantic parsed into)
            llm_response_data = chain_output['answer']
            print(f"  Raw LLM Pydantic Object (from chain's 'answer' key):\n  {llm_response_data}")

            # Check if it's an instance of our Pydantic model or a dict
            if isinstance(llm_response_data, FAQResponse):
                human_handoff_needed = llm_response_data.human_handoff_needed
                answer_content = llm_response_data.answer_content
            elif isinstance(llm_response_data, dict): # If it's a dict, access keys
                human_handoff_needed = llm_response_data.get("human_handoff_needed", False)
                answer_content = llm_response_data.get("answer_content", "Error: 'answer_content' not found.")
            else:
                print(f"  Error: Unexpected type for 'answer' in RAG chain output: {type(llm_response_data)}")
                print(f"  Content: {llm_response_data}")
                # Attempt to print a string representation if it's not a model or dict
                # This might happen if the Pydantic parser fails and the LLM returns a string despite instructions.
                # In this case, it's likely an unparsable string, but we print it for debug.
                try:
                    raw_llm_string_output = str(llm_response_data)
                    print(f"  Attempting to decode as a fallback JSON string: {raw_llm_string_output}")
                    fallback_data = json.loads(raw_llm_string_output) # Try to parse as JSON as a last resort
                    human_handoff_needed = fallback_data.get("human_handoff_needed", False)
                    answer_content = fallback_data.get("answer_content", "Error: Fallback JSON parsing failed to find content.")
                except Exception as fallback_e:
                    print(f"  Error: Fallback JSON parsing also failed: {fallback_e}. Output was not the expected Pydantic model or parsable JSON.")
                    print("  Problematic output: " + str(llm_response_data))
                    continue

            print("\n  Interpreted LLM Output:")

            if human_handoff_needed:
                print(f"  Escalation Triggered: {ESCALATION_MESSAGE}")
            else:
                print(f"  Answer: {answer_content}")

        except Exception as e:
            print(f"  An unexpected error occurred while processing the question '{question}': {e}")
            import traceback
            print(traceback.format_exc()) 
        time.sleep(0.1) # Add a small delay

    print("\n--- RAG Chain Tester Finished ---")

if __name__ == "__main__":
    main()

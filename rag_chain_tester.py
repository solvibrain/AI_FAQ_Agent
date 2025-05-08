# rag_chain_tester.py
import os
from dotenv import load_dotenv

from utils import process_faqs_and_setup_vector_store
from langchain_utils import (
    get_google_generativeai_llm,
    create_rag_chain,
    perform_similarity_search # Though RAG chain handles retrieval, direct search can be useful for context inspection
)

# Load environment variables from .env file (especially GOOGLE_API_KEY)
load_dotenv()

PROMPT_TEMPLATE = """You are a helpful FAQ assistant. Answer the user's question based ONLY on the following context. \nIf the context doesn't contain the answer, say you don't know. Be friendly and professional.\n\nContext: \n{context}\n\nQuestion: {question}\n\nAnswer:"""

def main():
    print("--- Starting RAG Chain Tester ---")

    # 1. Process FAQs and set up the vector store
    vector_store = process_faqs_and_setup_vector_store()
    if not vector_store:
        print("Failed to initialize vector store. Exiting.")
        return

    # 2. Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 docs
    print("Retriever created from vector store.")

    # 3. Initialize the LLM
    llm = get_google_generativeai_llm()
    if not llm:
        print("Failed to initialize LLM. Exiting.")
        return

    # 4. Create the RAG chain
    rag_chain = create_rag_chain(retriever, llm, PROMPT_TEMPLATE)
    if not rag_chain:
        print("Failed to create RAG chain. Exiting.")
        return

    print("\n--- Testing RAG Chain with Sample Questions ---")
    test_questions = [
        "What are your business hours?",
        "How can I reset my password?",
        "What payment methods do you accept?",
        "Where is your office located?",
        "How do I contact customer support?",
        "What is the return policy?" # This question is likely not in FAQs
    ]

    for question in test_questions:
        print(f"\nProcessing question: '{question}'")
        try:
            # Invoke the RAG chain
            # The input to the chain is a dictionary with the key "question"
            result = rag_chain.invoke({"question": question})
            
            print(f"  Question: {result.get('question')}")
            print(f"  Retrieved Context Snippets (first 100 chars each):")
            for i, doc in enumerate(result.get('context', [])):
                print(f"    Doc {i+1}: {doc.page_content[:100]}...")
            print(f"  LLM Answer: {result.get('answer')}")
            
        except Exception as e:
            print(f"  Error invoking RAG chain for question '{question}': {e}")

    print("\n--- RAG Chain Tester Finished ---")

if __name__ == "__main__":
    main()

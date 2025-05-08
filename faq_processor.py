from utils import process_faqs_and_setup_vector_store
from langchain_utils import perform_similarity_search

def main():
    print("Starting FAQ Processor...")

    # 1. Process FAQs and set up the vector store (in-memory Chroma by default)
    #    This function now handles loading, splitting, embedding, and vector store creation.
    vector_store = process_faqs_and_setup_vector_store()

    if not vector_store:
        print("Failed to initialize the vector store. Exiting.")
        return

    print("\n--- Testing Vector Store with Sample Queries ---")

    queries = [
        "What are the business hours?",
        "How to reset password?",
        "payment methods accepted",
        "How to contact support?",
        "office location"
    ]

    for query in queries:
        print(f"\nSearching for: '{query}'")
        search_results = perform_similarity_search(vector_store, query, k=2) # Get top 2 results
        
        if search_results:
            for i, doc in enumerate(search_results):
                print(f"  Result {i+1}:")
                print(f"    Content: {doc.page_content[:200]}...") # Print a snippet
                print(f"    Source: {doc.metadata.get('source')}, Row: {doc.metadata.get('row')}")
                print(f"    Original Question: {doc.metadata.get('original_question')}")
                # Note: ChromaDB's similarity_search might not directly return scores in the Document object by default.
                # If scores are needed, similarity_search_with_score can be used, returning (Document, score) tuples.
        else:
            print("  No relevant documents found.")

    print("\nFAQ Processor finished.")

if __name__ == "__main__":
    main()

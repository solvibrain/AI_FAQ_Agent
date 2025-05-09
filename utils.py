# utils.py
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_utils import (
    load_faqs_from_csv,
    get_text_splitter,
    get_huggingface_embeddings,
    create_chroma_vector_store
)

FAQ_DATA_PATH = "data/faq_data.csv"

# --- Component 3: Complexity Detection & Escalation Logic (LLM-driven approach) ---

# Static keyword-based escalation logic has been removed in favor of an LLM-based approach.

ESCALATION_MESSAGE = "I understand. This query seems to require more specialized assistance. I'll escalate this to a human agent who can help you further."

# --- End of Component 3 --- 

def process_faqs_and_setup_vector_store(csv_file_path: str = FAQ_DATA_PATH,
                                        chunk_size: int = 500,
                                        chunk_overlap: int = 50,
                                        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                                        persist_directory: str | None = None) -> Chroma | None:
    """
    Orchestrates the loading of FAQs, text splitting, embedding creation,
    and vector store setup.

    Returns:
        Chroma: The initialized vector store, or None if an error occurs.
    """
    print("--- Starting FAQ Processing and Vector Store Setup ---")

    # 1. Load FAQs
    docs = load_faqs_from_csv(csv_file_path)
    if not docs:
        print("Failed to load FAQs. Aborting vector store setup.")
        return None

    # 2. Get text splitter and split documents
    text_splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    if not split_docs:
        print("Failed to split documents. Aborting vector store setup.")
        return None
    print(f"Split {len(docs)} documents into {len(split_docs)} chunks.")

    # 3. Get embeddings
    embeddings = get_huggingface_embeddings(model_name=embedding_model_name)
    if not embeddings:
        print("Failed to initialize embeddings. Aborting vector store setup.")
        return None

    # 4. Create Chroma vector store
    vector_store = create_chroma_vector_store(
        documents=split_docs, 
        embeddings=embeddings,
        persist_directory=persist_directory
    )
    if not vector_store:
        print("Failed to create vector store.")
        return None

    print("--- FAQ Processing and Vector Store Setup Complete ---")
    return vector_store

if __name__ == '__main__':
    # Example usage: This will typically be called from your main application script
    print("Running utils.py directly for demonstration...")
    db = process_faqs_and_setup_vector_store()
    if db:
        print("Successfully created vector store from utils.py example.")
        # You could add a sample query here if needed for direct testing of utils.py
        # from langchain_utils import perform_similarity_search # This import is for the test block only
        # test_query = "business hours"
        # results = perform_similarity_search(db, test_query)
        # if results:
        #     print(f"\nTest query: '{test_query}'")
        #     for doc in results:
        #         print(f"  - {doc.page_content[:100]}... (Score: {doc.metadata.get('score', 'N/A')})") # Chroma might not add score by default
        # else:
        #     print(f"Test query '{test_query}' returned no results.")
    else:
        print("Failed to create vector store from utils.py example.")

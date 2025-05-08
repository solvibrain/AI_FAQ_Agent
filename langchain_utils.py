# langchain_utils.py
import os
import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_faqs_from_csv(csv_file_path: str) -> list[Document]:
    """
    Loads FAQs from a CSV file and converts them into a list of Langchain Document objects.
    Each document combines the 'Question' and 'Answer'.
    """
    try:
        df = pd.read_csv(csv_file_path)
        if 'Question' not in df.columns or 'Answer' not in df.columns:
            raise ValueError("CSV must contain 'Question' and 'Answer' columns.")
        
        df['text'] = "Question: " + df['Question'] + "\nAnswer: " + df['Answer']
        
        docs = [
            Document(
                page_content=row['text'],
                metadata={"source": csv_file_path, "row": index, "original_question": row['Question']}
            ) for index, row in df.iterrows()
        ]
        print(f"Successfully loaded {len(docs)} documents from {csv_file_path}")
        return docs
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return []
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file_path}' is empty.")
        return []
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading CSV: {e}")
        return []

def get_text_splitter(chunk_size: int = 500, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
    """
    Initializes and returns a RecursiveCharacterTextSplitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

def get_huggingface_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                               model_kwargs: dict = {'device': 'cpu'}, 
                               encode_kwargs: dict = {'normalize_embeddings': False}) -> HuggingFaceEmbeddings | None:
    """
    Initializes and returns HuggingFaceEmbeddings.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print(f"Successfully loaded HuggingFace embeddings model: {model_name}")
        return embeddings
    except Exception as e:
        print(f"Error initializing HuggingFace embeddings: {e}")
        return None

def create_chroma_vector_store(documents: list[Document], 
                               embeddings: HuggingFaceEmbeddings,
                               persist_directory: str | None = None) -> Chroma | None:
    """
    Creates or loads a Chroma vector store from documents and embeddings.
    If persist_directory is None, it creates an in-memory store.
    """
    if not documents or not embeddings:
        print("Error: Documents or embeddings not provided for Chroma vector store creation.")
        return None
    try:
        if persist_directory and os.path.exists(persist_directory):
            print(f"Loading existing Chroma vector store from: {persist_directory}")
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        elif persist_directory:
            print(f"Creating new Chroma vector store and persisting to: {persist_directory}")
            vector_store = Chroma.from_documents(
                documents=documents, 
                embedding=embeddings, 
                persist_directory=persist_directory
            )
        else:
            print("Creating new in-memory Chroma vector store.")
            vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
        
        print("Chroma vector store created/loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating/loading Chroma vector store: {e}")
        return None

def perform_similarity_search(vector_store: Chroma, query: str, k: int = 3) -> list[Document]:
    """
    Performs a similarity search on the vector store.
    """
    if not vector_store:
        print("Error: Vector store not provided for similarity search.")
        return []
    try:
        results = vector_store.similarity_search(query, k=k)
        print(f"Similarity search for '{query}' returned {len(results)} results.")
        return results
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []

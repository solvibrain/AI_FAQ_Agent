import os
import pandas as pd
import json 
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from operator import itemgetter

# Load environment variables from .env file
load_dotenv()

# --- Pydantic Model for Structured Output ---
class FAQResponse(BaseModel):
    human_handoff_needed: bool = Field(description="Set to true if the user explicitly asks for a human, expresses significant frustration, makes a formal complaint, or if their query is highly sensitive and clearly beyond standard FAQ resolution. Otherwise, set to false.")
    is_in_scope: bool = Field(description="Set to true if the retrieved context contains sufficient information to DIRECTLY and ADEQUATELY answer the user's question. Otherwise, set to false.")
    answer_content: str = Field(description="Behavior depends on other fields. If human_handoff_needed is true, provide a polite handoff. If is_in_scope is true (and no handoff), answer from context. If is_in_scope is false (and no handoff): for simple conversational phrases, give a natural reply (e.g., 'You're welcome!'); for genuine but out-of-scope queries, state information is not in FAQs.")

# --- Constants for Prompts ---
# The prompt will now incorporate format instructions from the Pydantic parser
DEFAULT_RAG_PROMPT_TEMPLATE = """You are an advanced FAQ assistant for an e-commerce store. Your primary goal is to answer customer questions based STRICTLY on the provided FAQ context.

User's question: {question}
Retrieved context: {context}

{format_instructions}

Follow these rules carefully when determining the fields:
1.  **is_in_scope**: Set to `true` ONLY if the retrieved context DIRECTLY and ADEQUATELY answers the user's question. Otherwise, set to `false`.
2.  **human_handoff_needed**: Set to `true` if the user explicitly asks for a human, expresses significant frustration, makes a formal complaint, or if their query is highly sensitive and clearly beyond standard FAQ resolution (e.g., complex legal issues, security breaches not covered by FAQs). Otherwise, set to `false`.
3.  **answer_content**:
    *   If `human_handoff_needed` is `true`, `answer_content` should be a polite handoff message (e.g., 'I understand you need further assistance. I'll connect you with a team member.').
    *   Else if `is_in_scope` is `true`, `answer_content` should be the answer derived ONLY from the retrieved context. Be direct, concise, friendly, and professional.
    *   Else (`is_in_scope` is `false` and `human_handoff_needed` is `false`):
        *   If the user's question is a simple conversational phrase (e.g., 'thank you', 'hello', 'okay', 'bye') and not a request for specific information, `answer_content` should be a polite, brief, and natural conversational reply (e.g., 'You're welcome!', 'Hello! How can I help you with our FAQs today?', 'Okay!', 'Goodbye!'). For these, `is_in_scope` should still be `false` as they are not FAQ-based.
        *   Otherwise (the user's question is a genuine query for information but not covered by the context), `answer_content` should politely state that the information isn't available in the FAQs (e.g., 'I can only answer questions based on our FAQs, and I don't have information on that topic.').

Provide ONLY the structured data as specified by the format instructions.
"""

# --- Function Definitions ---

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

def get_google_generativeai_llm(model_name: str = "gemini-2.0-flash", 
                                temperature: float = 0.0,
                                top_p: float = 0.85) -> ChatGoogleGenerativeAI | None:
    """
    Initializes and returns a Google Generative AI LLM.
    Requires GOOGLE_API_KEY to be set in the environment.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables. ")
        print("Please ensure it's set in your .env file and load_dotenv() has been called.")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            top_p=top_p
        )
        print(f"Successfully initialized Google Gemini LLM: {model_name}")
        return llm
    except Exception as e:
        print(f"Error initializing Google Gemini LLM: {e}")
        return None

def create_rag_chain(retriever, llm: ChatGoogleGenerativeAI, prompt_template_str: str = DEFAULT_RAG_PROMPT_TEMPLATE):
    """
    Creates a RAG chain using Langchain Expression Language (LCEL).
    This version uses PydanticOutputParser for structured JSON output.
    Args:
        retriever: The retriever object (e.g., from a vector store).
        llm: The language model to use.
        prompt_template_str: The string for the prompt template.
    Returns:
        A runnable RAG chain that outputs a parsed Pydantic object, or None if an error occurs.
    """
    try:
        # Instantiate the Pydantic parser
        parser = PydanticOutputParser(pydantic_object=FAQResponse)

        prompt = ChatPromptTemplate.from_template(
            template=prompt_template_str,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def ensure_llm_output_is_string_for_parser(message_or_dict):
            """Ensures the content fed to Pydantic parser is a string."""
            if hasattr(message_or_dict, 'content'): # AIMessage
                content = message_or_dict.content
                if isinstance(content, dict):
                    # Convert dict from LLM (if pre-parsed) back to JSON string
                    return json.dumps(content) 
                return content # Already a string or other, pass as is
            elif isinstance(message_or_dict, dict):
                return json.dumps(message_or_dict) # Should not happen if LLM outputs AIMessage
            return str(message_or_dict) # Fallback to string conversion

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | RunnableLambda(ensure_llm_output_is_string_for_parser) 
            | parser 
        )

        # The retriever needs the 'question' string.
        # The final chain (rag_chain_from_docs) might need 'question', 'context', and 'chat_history'.
        chain_setup = RunnableParallel(
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history") # Pass chat_history through
            }
        )

        rag_chain = chain_setup.assign(answer=rag_chain_from_docs)
        
        print("RAG chain with PydanticOutputParser created successfully.")
        return rag_chain
    except Exception as e:
        print(f"Error creating RAG chain with PydanticOutputParser: {e}")
        return None

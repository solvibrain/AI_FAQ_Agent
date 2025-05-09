# AI FAQ Agent with Multi-Turn Conversation

## Overview

This project implements an AI-powered FAQ agent designed to answer user queries based on a provided set of frequently asked questions. It features a conversational Gradio web interface, allowing for multi-turn interactions. The agent utilizes a Retrieval Augmented Generation (RAG) pipeline with Langchain, HuggingFace sentence transformers for embeddings, Google Gemini for language generation, and ChromaDB as a vector store. It also includes logic for query complexity assessment and escalation to a human agent if needed.

## Features

*   **Conversational UI**: Interactive chat interface built with Gradio (`gr.ChatInterface`) supporting multi-turn dialogues.
*   **Retrieval Augmented Generation (RAG)**: Leverages Langchain to retrieve relevant FAQ snippets and generate context-aware answers.
*   **FAQ-Based**: Answers are grounded in the content of `data/faq_data.csv`.
*   **Semantic Search**: Uses HuggingFace sentence transformers (`all-MiniLM-L6-v2`) for creating text embeddings and ChromaDB for efficient similarity search.
*   **Powerful LLM**: Utilizes Google's Gemini model for understanding queries and generating responses.
*   **Structured Output**: Employs Pydantic models to ensure the LLM provides responses in a consistent format, including escalation flags.
*   **Escalation Logic**: Can identify queries that are out-of-scope or require human intervention, providing appropriate escalation messages.
*   **Environment Configuration**: Securely manages API keys using a `.env` file.
*   **Modular Design**: Code is organized into logical modules (`app.py`, `langchain_utils.py`, `utils.py`).

## Technology Stack

*   **Python 3.x**
*   **Gradio**: For the web interface.
*   **Langchain**: Core framework for building the RAG pipeline and managing LLM interactions.
*   **HuggingFace Sentence Transformers**: For generating text embeddings.
*   **Google Generative AI (Gemini)**: Language model for generation and understanding.
*   **ChromaDB**: Vector database for storing and retrieving embeddings.
*   **Pandas**: For loading and processing the FAQ data from CSV.
*   **Dotenv**: For managing environment variables.

## Setup Instructions

1.  **Clone the Repository (if applicable)**:
    ```bash
    # If this were a git repository:
    # git clone <repository_url>
    # cd <repository_directory>
    ```
    For now, ensure you have all project files in a single directory.

2.  **Create and Activate a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Ensure your `requirements.txt` file is up-to-date with all necessary packages. Example packages include:
    ```
    gradio
    langchain
    langchain-community
    langchain-google-genai
    sentence-transformers
    chromadb
    pandas
    python-dotenv
    # Add other dependencies as needed
    ```
    Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**:
    Create a file named `.env` in the root directory of the project and add your Google API key:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```
    Replace `"YOUR_GOOGLE_API_KEY_HERE"` with your actual API key.

## Running the Application

1.  Navigate to the project's root directory in your terminal.
2.  Ensure your virtual environment is activated.
3.  Run the Gradio application:
    ```bash
    python app.py
    ```
4.  The terminal will display a local URL (e.g., `http://127.0.0.1:7860`). Open this URL in your web browser to interact with the AI FAQ Agent.

## Project Structure

```
. (root directory)
├── .env                 # Environment variables (e.g., API keys) - Create this manually
├── app.py               # Main application script with Gradio interface and core logic
├── langchain_utils.py   # Utilities for Langchain RAG chain, prompts, and LLM setup
├── utils.py             # General utilities (e.g., document loading, embedding creation)
├── requirements.txt     # Python package dependencies
├── README.md            # This file
└── data/
    └── faq_data.csv     # CSV file containing questions and answers for the FAQ agent
└── rag_chain_tester.py  # Optional script for testing the RAG chain directly (CLI)
```

## Testing

*   **Gradio UI**: The primary way to test is by interacting with the chat interface launched by `app.py`. Try various types of questions:
    *   Direct FAQ questions.
    *   Questions requiring some interpretation.
    *   Out-of-scope or irrelevant questions.
    *   Conversational phrases (greetings, thanks).
    *   Queries that might warrant escalation.
*   **Command Line Tester**: You can use `rag_chain_tester.py` to test the RAG chain's responses directly in the terminal without the Gradio UI. This can be useful for debugging the core LLM and retrieval logic.
    ```bash
    python rag_chain_tester.py
    ```

## Further Development Ideas

*   **Persistent Chat History**: Implement a database to store and retrieve chat histories for returning users.
*   **Contextual Carry-over**: Enhance the prompt or chain to better utilize conversation history for more contextually aware follow-up answers.
*   **Advanced Query Condensation**: Implement techniques to condense conversation history and the current question into a more effective search query for the retriever.
*   **Feedback Mechanism**: Add a way for users to rate the usefulness of answers.
*   **Deployment**: Package and deploy the application (e.g., using Docker, cloud platforms).

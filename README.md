# RAG-based Web Context Retriever

This project demonstrates a **Retrieval-Augmented Generation (RAG)**
pipeline built with **LangChain**, **ChromaDB**, **HuggingFace
embeddings**, and **Groq LLM**.\
It retrieves contextual information from a given webpage and answers
user queries based on the extracted content.

------------------------------------------------------------------------

## Features

-   Loads webpage content dynamically using `WebBaseLoader`.
-   Splits documents into chunks for efficient retrieval with
    `RecursiveCharacterTextSplitter`.
-   Creates vector embeddings using
    `sentence-transformers/all-MiniLM-L6-v2`.
-   Stores embeddings in **ChromaDB** for fast similarity search.
-   Uses **Groq LLM** (`deepseek-r1-distill-llama-70b`) for contextual
    question answering.
-   Ensures answers are restricted to the retrieved context only.

------------------------------------------------------------------------

## Installation

``` bash
# Clone this repository
git clone https://github.com/MohitAryal/Website-RAG
cd Website-RAG

# Create and activate virtual environment
python -m venv venv
source venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Environment Setup

Create a `.env` file in the project root and add your API keys:

``` env
GROQ_API_KEY=your_groq_api_key
```

------------------------------------------------------------------------

## Example Output

    Query: What are the major achievements?

    Output: The Nepal national cricket team has achieved T20I status in 2014, qualified for the 2014 ICC World Twenty20, gained ODI status in 2018, and participated in multiple ICC tournaments.

------------------------------------------------------------------------

## Requirements

-   Python 3.10+
-   LangChain
-   LangChain Chroma
-   HuggingFace Transformers
-   Langchain-groq
-   dotenv

Install all dependencies via:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

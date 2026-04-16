import os
import glob
import logging
from dotenv import load_dotenv

# Document loading and splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Azure and OpenAI integrations
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch

# Load environment variables from .env file
load_dotenv(override=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("indexer")

def index_docs():
    """
    Reads PDFs from the data folder, chunks them into smaller segments, 
    and uploads them to the Azure AI Search vector database.
    """
    
    # 1. Define Paths relative to the script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "..", "backend", "data")

    # 2. Validate required Environment Variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT", 
        "AZURE_OPENAI_API_KEY", 
        "AZURE_SEARCH_ENDPOINT", 
        "AZURE_SEARCH_API_KEY", 
        "AZURE_SEARCH_INDEX_NAME"
    ]
    
    missing_vars = [v for v in required_vars if not os.getenv(v)]
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        return

    # 3. Initialize Azure OpenAI Embeddings
    logger.info("Initializing Azure OpenAI Embeddings...")
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01"
        )

        # 4. Initialize Azure AI Search Vector Store
        logger.info("Initializing Azure AI Search vector store...")
        vector_store = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            embedding_function=embeddings.embed_query
        )

        # 5. Find and Process PDF Files in the data folder
        pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
        if not pdf_files:
            logger.warning("No PDFs found in the data folder.")
            return

        logger.info(f"Found {len(pdf_files)} PDFs to process.")
        all_splits = []

        for pdf_path in pdf_files:
            logger.info(f"Loading: {os.path.basename(pdf_path)}")
            loader = PyPDFLoader(pdf_path)
            raw_docs = loader.load()

            # Chunking strategy: 1000 characters with 200 overlap to maintain context
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            
            splits = text_splitter.split_documents(raw_docs)
            
            # Metadata tagging to track which PDF provided each rule
            for split in splits:
                split.metadata["source"] = os.path.basename(pdf_path)
            
            all_splits.extend(splits)
            logger.info(f"Split {os.path.basename(pdf_path)} into {len(splits)} chunks.")

        # 6. Upload Chunks to Azure AI Search
        if all_splits:
            logger.info(f"Uploading {len(all_splits)} total chunks to Azure Search...")
            vector_store.add_documents(all_splits)
            logger.info("Indexing complete. Knowledge base is ready for audits.")

    except Exception as e:
        logger.error(f"An error occurred during indexing: {str(e)}")

if __name__ == "__main__":
    index_docs()
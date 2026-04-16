import json
import os
import logging
import re
from typing import Dict, Any, List

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Internal imports from the project structure
from backend.src.graph.state import VideoAuditState, ComplianceIssue
from backend.src.services.video_indexer import VideoIndexerService

# Configure Logger
logger = logging.getLogger("brand_guardian")
logging.basicConfig(level=logging.INFO)

# --- NODE 1: INDEXER ---
def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    """Downloads video, uploads to Azure Video Indexer, and extracts text/transcript."""
    video_url = state.get("video_url")
    video_id_input = state.get("video_id")
    
    logger.info(f"Node Indexer: Processing {video_url}")
    local_path = "temp_audit_video.mp4"
    
    try:
        vi_service = VideoIndexerService()
        
        # 1. Download from YouTube
        if "youtube.com" in video_url or "youtu.be" in video_url:
            vi_service.download_youtube_video(video_url, local_path)
        else:
            raise Exception("Please provide a valid YouTube URL.")

        # 2. Upload to Azure
        azure_video_id = vi_service.upload_video(local_path, video_id_input)
        logger.info(f"Upload Success. Azure Video ID: {azure_video_id}")

        # 3. Cleanup local file
        if os.path.exists(local_path):
            os.remove(local_path)

        # 4. Wait for Azure to finish processing and extract data
        raw_insights = vi_service.wait_for_processing(azure_video_id)
        clean_data = vi_service.extract_data(raw_insights)
        
        logger.info("Node Indexer: Extraction complete.")
        return clean_data

    except Exception as e:
        logger.error(f"Video Indexer failed: {e}")
        return {
            "errors": [str(e)],
            "final_status": "fail",
            "transcript": "",
            "ocr_text": []
        }

# --- NODE 2: COMPLIANCE AUDITOR ---
def audio_content_node(state: VideoAuditState) -> Dict[str, Any]:
    """Uses LLM and RAG to audit the extracted video data against legal PDFs."""
    logger.info("Node Auditor: Querying knowledge base and LLM.")
    
    transcript = state.get("transcript", "")
    ocr_text = state.get("ocr_text", [])

    if not transcript and not ocr_text:
        return {
            "final_status": "fail",
            "final_report": "Audit skipped: No video data extracted."
        }

    # Initialize Azure Clients
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )
    
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    
    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function=embeddings.embed_query
    )

    # RAG: Retrieve relevant legal rules
    query_text = f"{transcript} {' '.join(ocr_text)}"
    docs = vector_store.similarity_search(query_text, k=3)
    retrieved_rules = "\n".join([doc.page_content for doc in docs])

    # AI Reasoning
    system_prompt = f"""
    You are a senior brand compliance auditor. 
    Rules: {retrieved_rules}
    Instructions: Analyze the transcript and OCR. Identify violations.
    Return strictly JSON with: compliance_results (list), status (pass/fail), and final_report (markdown).
    """
    
    user_message = f"""
    Metadata: {state.get('video_metadata')}
    Transcript: {transcript}
    OCR: {ocr_text}
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        
        # Clean output of any markdown backticks using Regex
        content = response.content
        audit_data = json.loads(re.sub(r"```json\n|\n```", "", content))

        return {
            "compliance_results": audit_data.get("compliance_results", []),
            "final_status": audit_data.get("status", "fail"),
            "final_report": audit_data.get("final_report", "No report generated.")
        }

    except Exception as e:
        logger.error(f"System Error in Auditor Node: {e}")
        return {"errors": [str(e)], "final_status": "failed"}
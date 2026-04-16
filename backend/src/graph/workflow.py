"""in this module , i will create stateGraph and compile it """

from langgraph.graph import StateGraph, END
from backend.src.graph.state import VideoAuditState
from backend.src.graph.nodes import (
    index_video_node, audio_content_node
)

def create_graph():

    workflow = StateGraph(VideoAuditState)

    workflow.add_node("indexer", index_video_node)
    workflow.add_node("auditor", audio_content_node)
    
    workflow.set_entry_point("indexer")
    workflow.add_edge("indexer","auditor")
    workflow.add_edge("auditor", END)
    compiler=workflow.compile()
    return compiler

app=create_graph()


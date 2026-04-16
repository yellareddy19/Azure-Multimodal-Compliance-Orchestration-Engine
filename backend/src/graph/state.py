import operator
from typing import Annotated, List, Dict, Optional, Any, TypedDict

# defining schema for a single compliance result
#error report

class ComplianceIssue(TypedDict):
    category : str
    description : str # desc of violation
    severity : str
    timestamp : Optional[str]


# defining the global graph state
class VideoAuditState(TypedDict):
    """
    Defines the data schema for langgraph execution content, holds all the info about
    audit right from start to end to  final report
    """

    # input parameters
    video_url : str
    video_id : str

    # ingestion and extraction data
    local_file_path : Optional[str]
    video_metadata : Dict[str,Any]
    transcript : Optional[str]
    ocr_text : List[str]

    # analysis output
    compliance_results : Annotated[List[ComplianceIssue],operator.add]

    #final delivarables
    final_status : str #pass or fail
    final_report : str #report 


    #system observability like api timeouts, any system level errors

    errors : Annotated[List[str], operator.add]



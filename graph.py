from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
from agents import create_speaker_agent, create_judge_agent
from langchain_core.messages import AIMessage
from loguru import logger

class ProductInfo(TypedDict):
    product_name: str
    product_type: str
    features: str
    target_audience: str
    usp: str

class GraphState(TypedDict):
    product_info: ProductInfo
    speaker1_strategy: AIMessage
    speaker2_strategy: AIMessage
    judge_decision: AIMessage

def create_marketing_campaign_graph(llm):
    workflow = StateGraph(GraphState)

    speaker1 = create_speaker_agent(llm, "Marketing Expert 1")
    speaker2 = create_speaker_agent(llm, "Marketing Expert 2")
    judge = create_judge_agent(llm)

    def speaker1_node(state: GraphState) -> Dict[str, Any]:
        product_info = state["product_info"]
        strategy = speaker1(product_info)
        return {"speaker1_strategy": strategy}

    def speaker2_node(state: GraphState) -> Dict[str, Any]:
        product_info = state["product_info"]
        strategy = speaker2(product_info)
        return {"speaker2_strategy": strategy}

    def judge_node(state: GraphState) -> Dict[str, Any]:
        product_info = state["product_info"]
        strategy1 = state["speaker1_strategy"].content
        strategy2 = state["speaker2_strategy"].content
        decision = judge(product_info, strategy1, strategy2)
        return {"judge_decision": decision}

    workflow.add_node("speaker1", speaker1_node)
    workflow.add_node("speaker2", speaker2_node)
    workflow.add_node("judge", judge_node)

    workflow.set_entry_point("speaker1")
    workflow.add_edge("speaker1", "speaker2")
    workflow.add_edge("speaker2", "judge")
    workflow.add_edge("judge", END)

    return workflow.compile()
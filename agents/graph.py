from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.nodes import (
    planner_node,
    analyzer_node,
    cartographer_node,
    synthesizer_node,
    verify_planner_node,
    verify_analyzer_node,
    verify_cartographer_node,
    verify_synthesizer_node,
    verify_planner_edge,
    verify_analyzer_edge,
    verify_cartographer_edge,
    verify_synthesizer_edge,
)

# Pipeline order (fully sequential):
#   planner -> analyzer -> cartographer -> synthesizer
# Each producer is followed by a verifier NODE (writes verdict/feedback to state) and then a
# conditional edge (pure routing: "retry" loops back to the producer, "approved" moves on).
# Routing functions cannot update state, which is why verification lives in its own node.

workflow = StateGraph(AgentState)

# 1. Register nodes
workflow.add_node("planner", planner_node)
workflow.add_node("verify_planner", verify_planner_node)

workflow.add_node("analyzer", analyzer_node)
workflow.add_node("verify_analyzer", verify_analyzer_node)

workflow.add_node("cartographer", cartographer_node)
workflow.add_node("verify_cartographer", verify_cartographer_node)

workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("verify_synthesizer", verify_synthesizer_node)

# 2. Planner (+ verification gate) -> Analyzer
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "verify_planner")
workflow.add_conditional_edges(
    "verify_planner", verify_planner_edge,
    {"retry": "planner", "approved": "analyzer"},
)

# 3. Analyzer (+ verification gate) -> Cartographer
workflow.add_edge("analyzer", "verify_analyzer")
workflow.add_conditional_edges(
    "verify_analyzer", verify_analyzer_edge,
    {"retry": "analyzer", "approved": "cartographer"},
)

# 4. Cartographer (+ verification gate) -> Synthesizer
workflow.add_edge("cartographer", "verify_cartographer")
workflow.add_conditional_edges(
    "verify_cartographer", verify_cartographer_edge,
    {"retry": "cartographer", "approved": "synthesizer"},
)

# 5. Synthesizer (+ verification gate) -> END
workflow.add_edge("synthesizer", "verify_synthesizer")
workflow.add_conditional_edges(
    "verify_synthesizer", verify_synthesizer_edge,
    {"retry": "synthesizer", "approved": END},
)

app = workflow.compile()

# Retries are capped per agent (MAX_RETRIES in agents/nodes.py): after that the pipeline moves on
# to the next agent. Worst case = 4 agents x (MAX_RETRIES + 1) runs x 2 steps (agent + verifier),
# which exceeds LangGraph's default limit of 25 with the default settings, so invoke with:
#   app.invoke({"chapter_text": text}, config={"recursion_limit": 8 * (MAX_RETRIES + 1) + 10})
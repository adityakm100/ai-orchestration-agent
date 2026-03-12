from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ----------------------------
# 1) Define shared state
# ----------------------------
class AgentState(TypedDict, total=False):
    task: str
    plan: str
    draft: str
    critique: str
    verdict: str
    revision_count: int
    max_revisions: int


# ----------------------------
# 2) Initialize Gemini
# ----------------------------
llm = ChatGoogleGenerativeAI(        
    model="gemini-2.5-flash",
    temperature=0,
)


# ----------------------------
# 3) Define AI Nodes
# ----------------------------
def planner_node(state: AgentState):

    prompt = f"""
You are a planner AI.
Create a concise step-by-step plan.

Task:
{state['task']}

Return only bullet points.
"""
    plan = llm.invoke(prompt).content
    return {"plan": plan}


def writer_node(state: AgentState):
    prompt = f"""
You are a writer AI.

Task:
{state['task']}

Plan:
{state.get('plan', '')}

Critique (if any):
{state.get('critique', '')}

Write a high-quality final answer.
"""
    draft = llm.invoke(prompt).content
    return {"draft": draft}


def critic_node(state: AgentState):
    prompt = f"""
You are a strict critic.

Evaluate the draft.

If good:
VERDICT: PASS

If needs improvement:
VERDICT: REVISE

Also give short feedback.

Task:
{state['task']}

Draft:
{state.get('draft', '')}
"""
    critique = llm.invoke(prompt).content

    verdict = "REVISE"
    if "VERDICT: PASS" in critique:
        verdict = "PASS"

    return {
        "critique": critique,
        "verdict": verdict
    }


def increment_revision(state: AgentState):
    return {
        "revision_count": state.get("revision_count", 0) + 1
    }


# ----------------------------
# 4) Routing Logic
# ----------------------------
def should_continue(state: AgentState):
    if state.get("verdict") == "PASS":
        return "end"

    if state.get("revision_count", 0) >= state.get("max_revisions", 2):
        return "end"

    return "revise"


# ----------------------------
# 5) Build Graph
# ----------------------------
builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)
builder.add_node("increment", increment_revision)

builder.set_entry_point("planner")

builder.add_edge("planner", "writer")
builder.add_edge("writer", "critic")

builder.add_conditional_edges(
    "critic",
    should_continue,
    {
        "revise": "increment",
        "end": END
    }
)

builder.add_edge("increment", "writer")

graph = builder.compile()


# ----------------------------
# 6) Run It
# ----------------------------
if __name__ == "__main__":
    result = graph.invoke({
        "task": "I have a spring festival (Chinese New Year) coming up, and my life has been really busy recently. What is the best way I can relax, and I can give a gift to my mom considering she is in China, and I am at the University of Michigan?",
        "revision_count": 0,
        "max_revisions": 10
    })

    print("\n===== PLAN =====\n")
    print(result.get("plan"))

    print("\n===== FINAL DRAFT =====\n")
    print(result.get("draft"))

    print("\n===== CRITIQUE =====\n")
    print(result.get("critique"))

    print("\nVerdict:", result.get("verdict"))
    print("Revisions:", result.get("revision_count"))

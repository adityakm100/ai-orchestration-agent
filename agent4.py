from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. Define the shared state
class MyState(TypedDict):
    text: str

# 2. Define nodes (functions that modify state)
def step_a(state: MyState):
    return {"text": state["text"] + " -> A"}

def step_b(state: MyState):
    return {"text": state["text"] + " -> B"}

def step_c(state: MyState):
    return {"text": state["text"] + " -> C"}

# 3. Create the graph
builder = StateGraph(MyState)

builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)
builder.add_node("step_c", step_c)

# 4. Define edges
builder.set_entry_point("step_a")
builder.add_edge("step_a", "step_b")
builder.add_edge("step_a", "step_c")
builder.add_edge("step_c", "step_b")
builder.add_edge("step_b", END)

# 5. Compile graph
graph = builder.compile()

# 6. Run graph
result = graph.invoke({"text": "Start"})
print(result)

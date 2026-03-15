from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from dotenv import load_dotenv
from os import environ

load_dotenv()

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )["embedding"]
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )["embedding"]

API_KEY = environ.get("API_KEY")

SYSTEM_TEMPLATE = """"You are a high-precision AI Email Architect. Your goal is to convert loosely structured key-value data (header-content, header-content) into professional emails using pre-defined templates. Parse input using the : separator to map headers to values. Identify: Recipient_Name, Context, Key_Points, Call_To_Action, and Desired_Tone. You must perform a self-correction loop internally before providing the final response. Step A: Generate an initial draft based on the most relevant template. Step B: Score the draft (1-10) on clarity, tone, and structure. Step C: If the score is <8.5, rewrite the email to address the specific weaknesses. Step D: Repeat until the score is ≥8.5. You must return your response in the following JSON schema: \"draft_1\": \"...\", \"critique\": \"...\", \"final_email\": \"...\" " """

class AgentState(TypedDict, total=False):
    task: str
    plan: str
    draft: str
    critique: str
    verdict: str
    revision_count: int
    max_revisions: int
    selected_template: str
    selected_template_name: str

def build_template_store():
    TEMPLATES = {
        "sponsorship": {
            "text": """Dear {Recipient_Name},

I hope this message finds you well. I'm reaching out on behalf of {org_name} regarding {Context}.

{Key_Points}

{Call_To_Action}

Best regards,
{sender_name}""",
            "description": "For sponsorship requests"
        },
        "collaboration": {
            "text": """Hi {Recipient_Name},

{Context}

We believe this partnership could {Key_Points}.

{Call_To_Action}

Looking forward to connecting,
{sender_name}""",
            "description": "For collaboration proposals"
        },
        "cold_outreach": {
            "text": """Dear {Recipient_Name},

My name is {sender_name} from {org_name}.

{Context}

{Key_Points}

{Call_To_Action}

Warm regards,
{sender_name}""",
            "description": "For cold outreach"
        }
    }
    documents = [
        Document(
            page_content=template["text"],
            metadata={"name": name, "description": template["description"]}
        )
        for name, template in TEMPLATES.items()
    ]

    embeddings = GeminiEmbeddings(api_key=API_KEY)
    
    Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
    print("Success!")



llm = ChatGoogleGenerativeAI(        
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=API_KEY
)

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

def template_selector_node(state: AgentState):
    vectordb = Chroma(
        persist_directory = "./chroma_db",
        embedding_function = GeminiEmbeddings(api_key=API_KEY),

    )
    results = vectordb.similarity_search(state['task'], k=1)
    if results:
        selected_doc = results[0]
        return {
            "selected_template": selected_doc.page_content,
            "selected_template_name": selected_doc.metadata.get("name", "unknown")
        }
    else:
        return {
            "selected_template": "",
            "selected_template_name": "none"
        }


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

def should_continue(state: AgentState):
    if state.get("verdict") == "PASS":
        return "end"

    if state.get("revision_count", 0) >= state.get("max_revisions", 2):
        return "end"

    return "revise"


builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)
builder.add_node("increment", increment_revision)
builder.add_node("template_selector", template_selector_node)

builder.set_entry_point("planner")

builder.add_edge("planner", "template_selector")
builder.add_edge("template_selector", "writer")
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

if __name__ == "__main__":
    result = graph.invoke({
        "task": "You are a high-precision AI Email Architect. Your goal is to convert loosely structured key-value data (header-content, header-content) into professional emails using pre-defined templates. Parse input using the : separator to map headers to values. Identify: Recipient_Name, Context, Key_Points, Call_To_Action, and Desired_Tone. You must perform a self-correction loop internally before providing the final response. Step A: Generate an initial draft based on the most relevant template. Step B: Score the draft (1-10) on clarity, tone, and structure. Step C: If the score is <8.5, rewrite the email to address the specific weaknesses. Step D: Repeat until the score is ≥8.5. You must return your response in the following JSON schema: \"draft_1\": \"...\", \"critique\": \"...\", \"final_email\": \"...\" ",
        "revision_count": 0,
        "max_revisions": 3
    })

    print("\n===== PLAN =====\n")
    print(result.get("plan"))

    print("\n===== FINAL DRAFT =====\n")
    print(result.get("draft"))

    print("\n===== CRITIQUE =====\n")
    print(result.get("critique"))

    print("\nVerdict:", result.get("verdict"))
    print("Revisions:", result.get("revision_count"))
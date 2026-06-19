from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from os import environ

load_dotenv()

API_KEY = environ.get("API_KEY")


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


class AgentState(TypedDict, total=False):
    task: str
    org_name: str
    org_info: dict[str, Any]
    contact_name: str
    contact_uniqname: str
    contact_phone: str
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
        "nonprofit_partnership": {
            "text": """Subject: Partnership with U-M Student Software Engineers

To Whom It May Concern,

My name is {contact_name}, and I am reaching out on behalf of the Google Developer Student Club at the University of Michigan. Our organization is dedicated to using technology to create positive social impact. We have had the privilege of supporting charities and nonprofits globally, including our recent work with the Nevada Homeless Alliance, where we helped develop technological solutions to improve community outreach and engagement.

We are reaching out because we admire the important work that {org_name} does in [what they do], which is a valuable contribution to the community, and we believe technology can help extend that impact even further.

As student software engineers, we would love to collaborate with you on creating a digital solution tailored to your needs. Our team would provide this development pro bono as part of our commitment to using our skills for good.

If you're interested, I'd be happy to arrange a brief meeting to discuss how we can support your work and enhance your mission with technology.

Thank you for considering this opportunity. I look forward to hearing from you.

Warm regards,
{contact_name}
Outreach & Partnerships, Google Developer Student Club
University of Michigan
{contact_uniqname}@umich.edu
{contact_phone}""",
            "description": "General nonprofit partnership outreach from GDSC University of Michigan"
        },
        "youth_services_partnership": {
            "text": """Subject: Tech Partnership for Youth Empowerment

Dear Team,

My name is {contact_name}, a student at the University of Michigan and member of the Google Developer Student Club (GDSC). We specialize in building pro bono software solutions for nonprofits making a difference in young people's lives.

We've been following {org_name}'s incredible work and believe that a custom digital tool—whether for program registration, volunteer coordination, or impact tracking—could meaningfully amplify your reach.

Our team offers full-cycle development at no cost, driven entirely by our mission to apply technology for social good.

Could we schedule a short call to explore possibilities?

Best,
{contact_name}
Google Developer Student Club | University of Michigan
{contact_uniqname}@umich.edu
{contact_phone}""",
            "description": "Outreach to youth-focused or education nonprofits"
        },
        "community_services_partnership": {
            "text": """Subject: Student Engineers Offering Pro Bono Tech Support

Hello,

I'm {contact_name}, representing the Google Developer Student Club at the University of Michigan. We partner with community-focused nonprofits to build free, custom software that solves real operational challenges.

{org_name}'s commitment to the community caught our attention, and we'd love to explore how a tailored digital solution could support your programs—from outreach tools to data management systems.

This would be entirely pro bono, backed by our team's passion for community-driven tech.

Would you be open to a brief conversation?

Sincerely,
{contact_name}
GDSC @ University of Michigan
{contact_uniqname}@umich.edu
{contact_phone}""",
            "description": "Outreach to community service or social welfare nonprofits"
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
    print("Template store built successfully.")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=API_KEY
)


def template_selector_node(state: AgentState):
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=GeminiEmbeddings(api_key=API_KEY),
    )
    query = f"nonprofit outreach email for {state.get('org_name', '')} - {state.get('task', '')}"
    results = vectordb.similarity_search(query, k=1)
    if results:
        doc = results[0]
        return {
            "selected_template": doc.page_content,
            "selected_template_name": doc.metadata.get("name", "unknown")
        }
    return {
        "selected_template": state.get("task", ""),
        "selected_template_name": "fallback"
    }


def planner_node(state: AgentState):
    prompt = f"""
You are a planner for a nonprofit outreach email.
Create a concise step-by-step plan to tailor the provided template to the organization.

Template selected: {state.get('selected_template_name', '')}
Organization name: {state.get('org_name', '')}
Organization info (JSON): {json.dumps(state.get('org_info', {}), ensure_ascii=False)}
Sender name: {state.get('contact_name', '')}
Sender uniqname: {state.get('contact_uniqname', '')}
Sender phone (optional): {state.get('contact_phone', '')}

Return only bullet points.
"""
    plan = llm.invoke(prompt).content
    return {"plan": plan}


def writer_node(state: AgentState):
    prompt = f"""
You are a professional outreach email writer.
Write a polished partnership email to the organization using the provided template as a base.

Requirements:
- Output ONLY the final email text (no analysis).
- Keep the Subject line.
- Do not leave any placeholders like [name], [uniqname], [phone number- optional], [what they do].
- Use org JSON to describe what they do accurately but concisely.
- Keep tone warm, respectful, and specific to the org.

Plan:
{state.get('plan', '')}

Critique (if any):
{state.get('critique', '')}

Organization name: {state.get('org_name', '')}
Organization info (JSON): {json.dumps(state.get('org_info', {}), ensure_ascii=False)}

Sender:
- name: {state.get('contact_name', '')}
- uniqname: {state.get('contact_uniqname', '')}
- phone: {state.get('contact_phone', '')}

Template to adapt:
{state.get('selected_template', state.get('task', ''))}
"""
    draft = llm.invoke(prompt).content
    return {"draft": draft}


def critic_node(state: AgentState):
    prompt = f"""
You are a strict critic for an outreach email.

Evaluate the draft.

If good:
VERDICT: PASS

If needs improvement:
VERDICT: REVISE

Also give short feedback.

Checklist:
- Includes a Subject line.
- No unreplaced placeholders like [name], [uniqname], [phone number- optional], [what they do].
- References the organization by name and accurately describes what they do based on the JSON.
- Professional, concise, and respectful tone.
- Signature has sender name and uniqname@umich.edu; phone line included only if provided.

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

builder.add_node("template_selector", template_selector_node)
builder.add_node("planner", planner_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)
builder.add_node("increment", increment_revision)

builder.set_entry_point("template_selector")

builder.add_edge("template_selector", "planner")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a tailored outreach letter to a nonprofit from saved_nonprofits JSON."
    )
    parser.add_argument(
        "organization",
        help="JSON filename (with or without .json) inside saved_nonprofits/, usually the organization name.",
    )
    parser.add_argument("name", help="Your full name (used in the letter).")
    parser.add_argument("uniqname", help="Your U-M uniqname (used for email: uniqname@umich.edu).")
    parser.add_argument(
        "phonenumber",
        nargs="?",
        default="",
        help="Optional phone number to include at the end of the letter.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=3,
        help="Maximum number of rewrite cycles (default: 3).",
    )
    parser.add_argument(
        "--rebuild-templates",
        action="store_true",
        help="Force rebuild the Chroma template store even if it already exists.",
    )
    args = parser.parse_args()

    if args.rebuild_templates or not os.path.exists("./chroma_db"):
        build_template_store()

    org_file = args.organization
    if not org_file.lower().endswith(".json"):
        org_file += ".json"

    org_path = Path(__file__).resolve().parent / "saved_nonprofits" / org_file
    if not org_path.exists():
        raise FileNotFoundError(f"Organization JSON not found: {org_path}")

    org_info = json.loads(org_path.read_text(encoding="utf-8"))
    org_name = org_path.stem

    phone = (args.phonenumber or "").strip()

    result = graph.invoke(
        {
            "task": "nonprofit partnership outreach email",
            "org_name": org_name,
            "org_info": org_info,
            "contact_name": args.name,
            "contact_uniqname": args.uniqname,
            "contact_phone": phone,
            "revision_count": 0,
            "max_revisions": args.max_revisions,
        }
    )

    print("\n===== SELECTED TEMPLATE =====\n")
    print(result.get("selected_template_name"))

    print("\n===== PLAN =====\n")
    print(result.get("plan"))

    print("\n===== FINAL DRAFT =====\n")
    print(result.get("draft"))

    print("\n===== CRITIQUE =====\n")
    print(result.get("critique"))

    print("\nVerdict:", result.get("verdict"))
    print("Revisions:", result.get("revision_count"))

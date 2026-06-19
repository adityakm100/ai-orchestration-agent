[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange)](https://python.langchain.com/docs/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple)](https://www.trychroma.com/)

An intelligent orchestration agent built with **LangGraph** and **ChromaDB** that utilizes Retrieval-Augmented Generation (RAG) to automate and personalize email writing based on dynamic templates and context.

Repository: [https://github.com/adityakm100/ai-orchestration-agent](https://github.com/adityakm100/ai-orchestration-agent)

---

## Project Overview

Writing contextual, well-formatted professional emails takes time. This project solves that by deploying an **AI Orchestration Agent** that doesn't just generate text, but actively *reasons* about the user's intent, searches a local knowledge base for the right template, and synthesizes a final draft.

Using **LangGraph** for cyclic, stateful agent routing and **ChromaDB** as the vector database for RAG, this agent:
1. Understands the email request (e.g., "Follow up with a client about a missed meeting").
2. Retrieves the most relevant email templates and company context from ChromaDB.
3. Injects this context into the prompt.
4. Drafts a highly tailored, ready-to-send email.

---

## Key Features

* **Stateful Orchestration:** Utilizes LangGraph to manage the agent's workflow (routing between retrieval, drafting, and reviewing states).
* **RAG-Powered Templates:** Uses ChromaDB to store, embed, and semantically retrieve email templates based on user intent.
* **Context-Aware Drafting:** The LLM isn't just writing blindly; it relies on retrieved ground-truth documents to ensure tone and formatting remain consistent.
* **Modular Design:** `agent5.py` acts as the main execution graph, making it easy to swap out LLMs or vector stores.

---

## Tech Stack

* **Language:** Python
* **Orchestration:** LangGraph / LangChain
* **Vector Database:** ChromaDB
* **LLM Provider:** Gemini 2.5 Flash (or interchangeable via LangChain)
* **Embeddings:** Gemini Embeddings / HuggingFace

---

## Getting Started

Follow these steps to set up and run the agent locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/adityakm100/ai-orchestration-agent.git](https://github.com/adityakm100/ai-orchestration-agent.git)
cd ai-orchestration-agent
```
2. Set Up Virtual Environment (Recommended)
Bash

# Windows
```
python -m venv venv
venv\Scripts\activate
```
# macOS/Linux
```
python3 -m venv venv
source venv/bin/activate
```
3. Install Dependencies
Bash
```
pip install -r requirements.txt
```
4. Configure Environment Variables

Create a .env file in the root directory and add your API keys:

Usage

The main entry point for the orchestrator is agent5.py.

To run the agent and generate an email:
Bash
```
python agent5.py
```
How it works under the hood (agent5.py):

    Initialization: The script initializes the ChromaDB client and loads/embeds your email templates into the vector store.

    Graph Compilation: LangGraph compiles the nodes (retrieve_template, draft_email, review_email) into a stateful workflow.

    Execution: You provide a prompt/intent to the script. The graph traverses the nodes, retrieves the optimal template using semantic search, and streams the final output to your console.

🏗️ Architecture / Workflow

    User Input: "I need to send a welcome email to a new SaaS subscriber."

    Agent State (LangGraph): Agent enters the Retrieval node.

    Vector Search (ChromaDB): Queries the DB for "SaaS welcome email template".

    Agent State (LangGraph): Agent passes the retrieved template and user prompt to the Generation node.

    Output: A customized, professional email draft is produced.

📈 Future Improvements

    [ ] Connect agent output directly to the Gmail API for one-click sending.

    [ ] Add a Streamlit UI for easier non-terminal interaction.

    [ ] Implement a human-in-the-loop (HITL) node in LangGraph for draft approval before finalization.

👤 Author

adityakm100

    GitHub: @adityakm100

If you find this project helpful, please consider giving it a ⭐!

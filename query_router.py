from typing import Literal

from langchain_classic.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a routing controller that decides whether a question should be
answered with a SQL database or a document retrieval system.

If the question requests metrics, counts, aggregations, filters, or
references specific columns/rows that likely live inside structured tables,
respond with SQL.

If the question is better answered from unstructured knowledge or the SQL
database is unlikely to have the information, respond with RAG.

When unsure, prefer RAG.

Available SQL schema (tables, columns, keys):
{schema}

Question: {question}

Respond with ONE WORD: SQL or RAG.
""".strip()
)


def determine_route(question: str, llm, schema_snapshot: str, sql_available: bool) -> Literal["sql", "rag"]:
    """Determine whether to route the query to SQL or RAG."""
    if not sql_available:
        return "rag"

    prompt = ROUTER_PROMPT.format(question=question, schema=schema_snapshot or "No schema available.")
    decision = llm.invoke(prompt).strip().upper()

    if decision.startswith("SQL"):
        return "sql"
    return "rag"


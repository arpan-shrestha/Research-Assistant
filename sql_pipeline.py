import json
from dataclasses import dataclass
from typing import Any, Optional

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

@dataclass
class SQLPipelineState:
    chain: Any
    schema_snapshot: str

def init_sql_pipeline(llm) -> Optional[SQLPipelineState]:
    """Initialize the SQL NL2SQL pipeline."""
    database_url = "sqlite:////Users/arpanshrestha/Desktop/Research-Assistant/db.sqlite3"

    try:
        sql_db = SQLDatabase.from_uri(database_url)
        schema_snapshot = sql_db.get_table_info()

        chain = create_sql_agent(
            llm=llm,
            db=sql_db,
            verbose=True,
        )

        print("[SQL] SQL pipeline initialized successfully")
        return SQLPipelineState(chain=chain, schema_snapshot=schema_snapshot)

    except Exception as exc:
        print(f"[SQL] Failed to initialize SQL pipeline: {exc}")
        return None


def run_sql_query(chain: Any, question: str) -> dict:
    """Run NL2SQL query using the agent."""
    output = chain.invoke({"input": question})

    return {
        "answer": str(output.get("output", output)),
        "metadata": {}
    }

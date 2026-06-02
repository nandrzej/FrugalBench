"""Task definitions for the 8B Deterministic Benchmark."""

from tasks.task1_extraction import task1_extraction
from tasks.task2_bash_sandbox import task2_bash_sandbox
from tasks.task3_email_reply import task3_email_reply
from tasks.task4_summarization import task4_summarization
from tasks.task5_agentic import task5_agentic
from tasks.task6_hallucination import task6_hallucination
from tasks.task7_routing import task7_routing
from tasks.task8_rag_abstention import task8_rag_abstention
from tasks.task9_tabular_math import task9_tabular_math
from tasks.task10_code_debug import task10_code_debug
from tasks.task11_logic_puzzle import task11_logic_puzzle
from tasks.task12_safety_refusal import task12_safety_refusal
from tasks.task13_schema_extraction import task13_schema_extraction
from tasks.task14_pii_redaction import task14_pii_redaction
from tasks.task16_sql_execution import task16_sql_execution

__all__ = [
    "task1_extraction",
    "task2_bash_sandbox",
    "task3_email_reply",
    "task4_summarization",
    "task5_agentic",
    "task6_hallucination",
    "task7_routing",
    "task8_rag_abstention",
    "task9_tabular_math",
    "task10_code_debug",
    "task11_logic_puzzle",
    "task12_safety_refusal",
    "task13_schema_extraction",
    "task14_pii_redaction",
    "task16_sql_execution",
]

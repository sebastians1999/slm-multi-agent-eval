from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import uuid, textwrap
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from scr.utilities.helper_functions import extract_final_answer

load_dotenv()

# LangGraph / LangChain
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchResults


# ============================================================
#  IMMUTABLE FIELDS
# ============================================================

model="llama3.1:8b"

IMMUTABLE_FIELDS = {
    "session_id",
    "problem",
    "findings_ready",
    "summaries_ready",
    "solution_ready",
    "meta",
}


def filter_immutable(fn: Callable):
    """Strip immutable keys from node outputs to avoid update errors."""
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = fn(state)
        if not isinstance(new_state, dict):
            return new_state
        return {k: v for k, v in new_state.items() if k not in IMMUTABLE_FIELDS}
    return wrapper


# ============================================================
#  MEMORY CLASSES
# ============================================================

@dataclass
class FindingsRecord:
    query: str
    snippets: List[str]
    sources: List[str]

@dataclass
class SolutionRecord:
    solution: str
    meta: Dict[str, Any]

class FindingsMemory:
    def __init__(self):
        self._store: Dict[str, FindingsRecord] = {}

    def add(self, sid: str, record: FindingsRecord):
        self._store[sid] = record

    def get(self, sid: str) -> Optional[FindingsRecord]:
        return self._store.get(sid)

    def clear(self, sid: str):
        self._store.pop(sid, None)

class SolutionsMemory:
    def __init__(self):
        self._store: Dict[str, SolutionRecord] = {}

    def add(self, sid: str, record: SolutionRecord):
        self._store[sid] = record

    def get(self, sid: str) -> Optional[SolutionRecord]:
        return self._store.get(sid)

    def clear(self, sid: str):
        self._store.pop(sid, None)


FINDINGS_MEMORY = FindingsMemory()
SOLUTIONS_MEMORY = SolutionsMemory()


# ============================================================
#  LOCAL LLM + WEB TOOL
# ============================================================

local_llm = OllamaLLM(model=model, temperature=0.3)
duckduckgo = DuckDuckGoSearchResults(max_results=5)


# ============================================================
#  UTILITIES
# ============================================================

def divider(title: str):
    print("\n" + "═" * 80)
    print(f"🧠 {title}")
    print("═" * 80 + "\n")


def ddg_collect_raw(query: str) -> Dict[str, Any]:
    """Collect raw snippets from DuckDuckGo."""
    print(f"   🔎 [DuckDuckGo] Searching for: {query!r}")
    results = duckduckgo.invoke(query, output_format="list")
    urls, snippets = [], []

    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                url = r.get("link") or r.get("url")
                snippet = r.get("snippet") or r.get("text") or ""
                if url:
                    urls.append(url)
                if snippet:
                    snippets.append(snippet)
            else:
                snippets.append(str(r))
    else:
        snippets.append(str(results))

    print(f"   ✅ Collected {len(snippets)} snippets and {len(urls)} URLs.\n")
    return {"snippets": snippets, "urls": urls[:10]}


# ============================================================
#  GRAPH STATE
# ============================================================



class StructuredOutput(BaseModel):
    model_answer: str | None = None
    reasoning_trace: str | None = None
    
    
class GraphState(TypedDict, total=False):
    session_id: str
    problem: str
    findings_ready: bool
    summaries_ready: bool
    solution_ready: bool
    needs_revision: bool
    iteration_count: int      # 🆕 Track number of feedback iterations
    final_solution: Optional[str]
    structured_output: StructuredOutput
    meta: Dict[str, Any]


# ============================================================
#  AGENT NODES
# ============================================================

@filter_immutable
def researcher_node(state: GraphState) -> GraphState:
    divider("Researcher Agent")
    sid = state["session_id"]
    problem = state["problem"]

    # Include evaluation feedback if present
    feedback_record = SOLUTIONS_MEMORY.get(sid)
    feedback = ""
    if feedback_record and feedback_record.meta.get("stage") == "feedback":
        feedback = feedback_record.solution

    prompt = f"""
You are the Researcher Agent.
Your task is to decide whether a web search is needed to answer the user's problem, and
if yes, generate the best possible concise search query.

Guidelines:
- Look carefully at the user's problem below.
- If it is purely mathematical, logical, or reasoning-based → respond ONLY "NO_SEARCH". No other text
- If it may depend on real-world facts, data, events, or statistics → respond ONLY "SEARCH: <optimal query>".
- The query should be short, focused, and maximize the likelihood of retrieving relevant answers. It must a google query style search.
- Do NOT just repeat the problem text verbatim — rephrase it into an effective search query.

User Problem:
{problem}

Previous feedback from evaluation (if any):
{feedback}
"""
    decision = local_llm.invoke(prompt).strip()
    print(f"[Researcher] Decision: {decision}")

    if decision.upper().startswith("NO_SEARCH"):
        FINDINGS_MEMORY.add(
            sid,
            FindingsRecord(
                query=problem,
                snippets=["(No web search performed — reasoning sufficient.)"],
                sources=[],
            ),
        )
        print("   ⚙️ No search required (problem can be reasoned internally).")
        state["findings_ready"] = True
        return state

    if "SEARCH:" in decision.upper():
        query = decision.split("SEARCH:", 1)[1].strip()
    else:
        query = problem.strip()

    print(f"   🔎 Using generated query: {query}")
    raw = ddg_collect_raw(query)

    FINDINGS_MEMORY.add(
        sid,
        FindingsRecord(
            query=query,
            snippets=raw["snippets"],
            sources=raw["urls"],
        ),
    )

    print(f"[Researcher] Stored {len(raw['snippets'])} snippets and {len(raw['urls'])} URLs.")
    state["findings_ready"] = True
    return state


@filter_immutable
def summarizer_node(state: GraphState) -> GraphState:
    divider("Summarizer Agent")
    sid = state["session_id"]
    record = FINDINGS_MEMORY.get(sid)

    all_snippets = "\n\n".join(record.snippets) if record else "(no findings)"
    prompt = f"""You are the Summarizer.
Summarize the following evidence into a concise factual summary (5–7 sentences). Do not add new info.
Evidence:
{all_snippets}
"""
    summary = local_llm.invoke(prompt)
    FINDINGS_MEMORY.add(
        sid,
        FindingsRecord(
            query=f"SUMMARY",
            snippets=[summary],
            sources=record.sources if record else [],
        ),
    )
    print(f"[Summarizer] Summary snippet:\n{textwrap.indent(summary[:400], '   ')}\n...")
    state["summaries_ready"] = True
    return state


@filter_immutable
def solver_node(state: GraphState) -> GraphState:
    divider("Solver Agent")
    sid = state["session_id"]
    problem = state["problem"]
    findings = FINDINGS_MEMORY.get(sid)
    summary = "\n".join(findings.snippets) if findings else "(no summary available)"

    feedback_record = SOLUTIONS_MEMORY.get(sid)
    feedback = ""
    if feedback_record and feedback_record.meta.get("stage") == "feedback":
        feedback = feedback_record.solution

    prompt = f"""You are the Solver Agent.
Use reasoning and the summarized evidence to solve the problem below.

Problem:
{problem}

Summary context:
{summary}

Previous evaluation feedback (if any):
{feedback}

Provide an improved, clear, step-by-step solution and a final answer.
"""
    solution = local_llm.invoke(prompt)
    SOLUTIONS_MEMORY.add(sid, SolutionRecord(solution, {"stage": "iteration"}))
    print(f"[Solver] Solution snippet:\n{textwrap.indent(solution[:400], '   ')}\n...")
    state["solution_ready"] = True
    state["final_solution"] = solution
    return state


@filter_immutable
def evaluator_node(state: GraphState) -> GraphState:
    divider("Evaluator Agent")
    sid = state["session_id"]
    problem = state["problem"]
    solution_record = SOLUTIONS_MEMORY.get(sid)
    solution_text = solution_record.solution if solution_record else "(no solution)"

    prompt = f"""You are the Evaluator Agent.
Evaluate the following solution to the problem. 

Problem:
{problem}

Proposed Solution:
{solution_text}

Tasks:
1. Determine if the final answer is correct and logically consistent.
2. If correct, respond with "CORRECT: <brief confirmation>" No other text or characters before CORRECT. Finish you answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
3. If not, respond with "INCORRECT: <specific feedback and what to adjust for a new iteration>".
"""
    evaluation = local_llm.invoke(prompt)
    print(f"[Evaluator] Evaluation:\n{textwrap.indent(evaluation[:400], '   ')}\n...")

    if evaluation.strip().upper().startswith("CORRECT"):
        print("✅ Solution validated as correct.")
        model_answer = extract_final_answer(evaluation)
        state["structured_output"] = StructuredOutput(reasoning_trace=solution_text,model_answer=model_answer)
        state["final_solution"] = solution_text
        state["needs_revision"] = False
    else:
        print("⚠️ Evaluation suggests improvement; storing feedback.")
        SOLUTIONS_MEMORY.add(sid, SolutionRecord(evaluation, {"stage": "feedback"}))
        state["final_solution"] = evaluation
        model_answer = extract_final_answer(evaluation)
        state["structured_output"] = StructuredOutput(reasoning_trace=solution_text,model_answer=model_answer)
        state["final_solution"] = solution_text
        state["needs_revision"] = True

    # 🆕 Count feedback iterations
    if state.get("needs_revision"):
        state["iteration_count"] = state.get("iteration_count", 0) + 1
    else:
        state["iteration_count"] = 0

    return state


@filter_immutable
def reset_node(state: GraphState) -> GraphState:
    divider("Cycle Reset")
    sid = state["session_id"]
    FINDINGS_MEMORY.clear(sid)
    SOLUTIONS_MEMORY.clear(sid)
    print("[Reset] Cleared all memories.")
    return state


# ============================================================
#  GRAPH WIRING (feedback restarts from Researcher)
# ============================================================

workflow = StateGraph(GraphState)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Summarizer", summarizer_node)
workflow.add_node("Solver", solver_node)
workflow.add_node("Evaluator", evaluator_node)
workflow.add_node("Reset", reset_node)

workflow.add_edge(START, "Researcher")
workflow.add_edge("Researcher", "Summarizer")
workflow.add_edge("Summarizer", "Solver")
workflow.add_edge("Solver", "Evaluator")



# Conditional loop: restart from Researcher if incorrect (max 2 times)
def evaluation_router(state: GraphState):
    if state.get("needs_revision"):
        if state.get("iteration_count", 0) >= 2:
            print("🛑 Maximum feedback iterations reached. Ending process.")
            return "Reset"
        return "Researcher"
    return "Reset"

workflow.add_conditional_edges(
    "Evaluator",
    evaluation_router,
    {"Researcher": "Researcher", "Reset": "Reset"},
)

workflow.add_edge("Reset", END)

app = workflow.compile(checkpointer=MemorySaver())

# ============================================================
#  RUNNER
# ============================================================

def run_multi_agent(problem: str) -> dict[str,Any] |Any:
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    print(f"\n🚀 Starting Self-Improving Reasoning Session: {sid}\n")
    init_state = {
        "session_id": sid,
        "problem": problem,
        "meta": {},
        "iteration_count": 0,   # 🆕 initialize counter
    }
    final = app.invoke(init_state, config={"configurable": {"thread_id": sid}})
    print("\n✅ Process completed.\n")
    return final


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    problem = """At a community picnic, Sam brings 3 boxes of oranges. Each box contains 4 bags, and each bag has 8 oranges. He gives away one-quarter of all his oranges to volunteers. Then he uses 12 oranges to make juice. He splits the remaining oranges equally among 5 friends.
How many oranges does each friend receive??"""
    answer = run_multi_agent(problem)
    print("\n=== 🏁 FINAL SOLUTION ===\n")
    print(answer)

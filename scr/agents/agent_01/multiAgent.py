"""
Refactored Multi-Agent System
Clean, maintainable implementation with proper metadata tracking.
"""

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import uuid
import textwrap
from datetime import datetime
from pydantic import BaseModel

# LangGraph
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
#  GRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):
    """
    State that flows through the multi-agent graph.

    Core fields:
        session_id: Unique identifier for this run
        problem: The question/problem to solve
        metadata: Accumulated metadata (tokens, energy, etc.)

    Agent outputs:
        search_decision: Whether to search or not
        search_query: Generated search query
        findings: Raw search results (snippets + sources)
        summary: Summarized findings
        solution: Proposed solution
        evaluation: Evaluation feedback
        structured_output: Final formatted output

    Control flow:
        needs_revision: Whether solution needs improvement
        iteration_count: Number of feedback iterations
        feedback: Evaluation feedback for next iteration
    """
    # Core
    session_id: str
    problem: str
    metadata: AgentMetaData

    # Agent outputs
    search_decision: str
    search_query: Optional[str]
    findings: Dict[str, Any]  # {"snippets": [...], "sources": [...]}
    summary: str
    solution: str
    evaluation: str
    structured_output: StructuredOutput

    # Control flow
    needs_revision: bool
    iteration_count: int
    feedback: str


# ============================================================
#  MULTI-AGENT CLASS
# ============================================================

class MultiAgent(BaseMultiAgent):
    """
    Multi-agent reasoning system with metadata tracking.

    Workflow:
        Researcher → Summarizer → Solver → Evaluator
        └─ (feedback loop, max 2 iterations) ─┘
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        max_iterations: int = 2,
        use_web_search: bool = True,
        **kwargs
    ):
        super().__init__(model, temperature, base_url, api_key, **kwargs)
        self.max_iterations = max_iterations
        self.use_web_search = use_web_search
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("Researcher", self._researcher_node)
        workflow.add_node("Summarizer", self._summarizer_node)
        workflow.add_node("Solver", self._solver_node)
        workflow.add_node("Evaluator", self._evaluator_node)

        # Linear flow
        workflow.add_edge(START, "Researcher")
        workflow.add_edge("Researcher", "Summarizer")
        workflow.add_edge("Summarizer", "Solver")
        workflow.add_edge("Solver", "Evaluator")

        # Conditional feedback loop
        workflow.add_conditional_edges(
            "Evaluator",
            self._evaluation_router,
            {"Researcher": "Researcher", "END": END}
        )

        return workflow.compile(checkpointer=MemorySaver())

    # ============================================================
    #  AGENT NODES
    # ============================================================

    def _researcher_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Researcher Agent: Decides if web search is needed and generates query.
        """
        self._print_divider("Researcher Agent")

        problem = state["problem"]
        feedback = state.get("feedback", "")

        prompt = f"""You are the Researcher Agent.
Your task is to decide whether a web search is needed to answer the user's problem, and
if yes, generate the best possible concise search query.

Guidelines:
- Look carefully at the user's problem below.
- If it is purely mathematical, logical, or reasoning-based → respond ONLY "NO_SEARCH". No other text
- If it may depend on real-world facts, data, events, or statistics → respond ONLY "SEARCH: <optimal query>".
- The query should be short, focused, and maximize the likelihood of retrieving relevant answers. It must be a google query style search.
- Do NOT just repeat the problem text verbatim — rephrase it into an effective search query.

User Problem:
{problem}

Previous feedback from evaluation (if any):
{feedback}
"""

        # Call LLM with metadata tracking
        response = self._invoke_with_metadata(prompt, state)
        decision = response["content"].strip()

        print(f"[Researcher] Decision: {decision}")

        # Parse decision
        if decision.upper().startswith("NO_SEARCH"):
            findings = {
                "snippets": ["(No web search performed — reasoning sufficient.)"],
                "sources": []
            }
            search_query = None
            print("   ⚙️ No search required (problem can be reasoned internally).")
        elif "SEARCH:" in decision.upper():
            search_query = decision.split("SEARCH:", 1)[1].strip()
            print(f"   🔎 Generated query: {search_query}")

            if self.use_web_search:
                findings = self._perform_web_search(search_query)
            else:
                findings = {
                    "snippets": ["(Web search disabled in this configuration)"],
                    "sources": []
                }
        else:
            # Fallback: treat whole problem as query
            search_query = problem
            findings = self._perform_web_search(search_query) if self.use_web_search else {
                "snippets": ["(Web search disabled)"],
                "sources": []
            }

        return {
            "search_decision": decision,
            "search_query": search_query,
            "findings": findings,
            "metadata": state["metadata"]
        }

    def _summarizer_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Summarizer Agent: Condenses findings into a concise summary.
        """
        self._print_divider("Summarizer Agent")

        findings = state.get("findings", {})
        snippets = findings.get("snippets", [])
        all_snippets = "\n\n".join(snippets)

        prompt = f"""You are the Summarizer.
Summarize the following evidence into a concise factual summary (5–7 sentences). Do not add new info.

Evidence:
{all_snippets}
"""

        response = self._invoke_with_metadata(prompt, state)
        summary = response["content"]

        print(f"[Summarizer] Summary snippet:\n{textwrap.indent(summary[:400], '   ')}...")

        return {
            "summary": summary,
            "metadata": state["metadata"]
        }

    def _solver_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Solver Agent: Generates solution using summary and reasoning.
        """
        self._print_divider("Solver Agent")

        problem = state["problem"]
        summary = state.get("summary", "(no summary available)")
        feedback = state.get("feedback", "")

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

        response = self._invoke_with_metadata(prompt, state)
        solution = response["content"]

        print(f"[Solver] Solution snippet:\n{textwrap.indent(solution[:400], '   ')}...")

        return {
            "solution": solution,
            "metadata": state["metadata"]
        }

    def _evaluator_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Evaluator Agent: Validates solution and decides if revision is needed.
        """
        self._print_divider("Evaluator Agent")

        problem = state["problem"]
        solution = state.get("solution", "(no solution)")

        prompt = f"""You are the Evaluator Agent.
Evaluate the following solution to the problem.

Problem:
{problem}

Proposed Solution:
{solution}

Tasks:
1. Determine if the final answer is correct and logically consistent.
2. If correct, respond with "CORRECT: <brief confirmation>" No other text or characters before CORRECT. Finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
3. If not, respond with "INCORRECT: <specific feedback and what to adjust for a new iteration>".
"""

        response = self._invoke_with_metadata(prompt, state)
        evaluation = response["content"]

        print(f"[Evaluator] Evaluation:\n{textwrap.indent(evaluation[:400], '   ')}...")

        # Parse evaluation
        if evaluation.strip().upper().startswith("CORRECT"):
            print("✅ Solution validated as correct.")
            model_answer = extract_final_answer(evaluation)
            structured_output = StructuredOutput(
                reasoning_trace=solution,
                model_answer=model_answer
            )
            needs_revision = False
            feedback = ""
        else:
            print("⚠️ Evaluation suggests improvement; storing feedback.")
            model_answer = extract_final_answer(evaluation)
            structured_output = StructuredOutput(
                reasoning_trace=solution,
                model_answer=model_answer
            )
            needs_revision = True
            feedback = evaluation

        # Update iteration count
        iteration_count = state.get("iteration_count", 0)
        if needs_revision:
            iteration_count += 1

        return {
            "evaluation": evaluation,
            "structured_output": structured_output,
            "needs_revision": needs_revision,
            "feedback": feedback,
            "iteration_count": iteration_count,
            "metadata": state["metadata"]
        }

    # ============================================================
    #  ROUTING
    # ============================================================

    def _evaluation_router(self, state: GraphState) -> str:
        """Route based on evaluation: loop back or end."""
        if state.get("needs_revision"):
            if state.get("iteration_count", 0) >= self.max_iterations:
                print(f"🛑 Maximum feedback iterations ({self.max_iterations}) reached. Ending process.")
                return "END"
            print(f"🔄 Revision needed. Starting iteration {state.get('iteration_count', 0)}...")
            return "Researcher"
        return "END"

    # ============================================================
    #  HELPER METHODS
    # ============================================================

    def _invoke_with_metadata(self, prompt: str, state: GraphState) -> Dict[str, Any]:
        """
        Invoke LLM and update metadata in state.

        Returns:
            dict with "content" and updated metadata
        """
        session_id = state["session_id"]
        metadata = state["metadata"]

        messages = [{"role": "user", "content": prompt}]

        response_dict = self.invoke(messages)

        content = response_dict["choices"][0]["message"]["content"]

        # Extract usage
        usage = response_dict.get("usage", {})
        metadata.completion_tokens += usage.get("completion_tokens", 0)
        metadata.prompt_tokens += usage.get("prompt_tokens", 0)
        metadata.total_tokens += usage.get("total_tokens", 0)
        metadata.iterations += 1

        # Extract energy consumption if available
        energy_data = response_dict.get("energy_consumption", {})
        if energy_data:
            metadata.total_energy_joules += energy_data.get("joules", 0.0)
            metadata.total_duration_seconds += energy_data.get("duration_seconds", 0.0)

            # Update average watts
            if metadata.total_duration_seconds > 0:
                metadata.average_watts = metadata.total_energy_joules / metadata.total_duration_seconds

        return {"content": content}

    def _perform_web_search(self, query: str) -> Dict[str, Any]:
        """
        Perform web search (placeholder - integrate with DuckDuckGo or other).
        """
        try:
            from langchain_community.tools import DuckDuckGoSearchResults
            duckduckgo = DuckDuckGoSearchResults(max_results=5)

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

            print(f"   ✅ Collected {len(snippets)} snippets and {len(urls)} URLs.")
            return {"snippets": snippets, "sources": urls[:10]}

        except Exception as e:
            print(f"   ⚠️ Web search failed: {e}")
            return {
                "snippets": [f"(Web search failed: {e})"],
                "sources": []
            }

    def _print_divider(self, title: str):
        """Print formatted section divider."""
        print("\n" + "═" * 80)
        print(f"🧠 {title}")
        print("═" * 80 + "\n")

    # ============================================================
    #  PUBLIC API
    # ============================================================

    def run(self, problem: str) -> Dict[str, Any]:
        """
        Run the multi-agent system on a problem.

        Args:
            problem: The question/problem to solve

        Returns:
            Final state dict with solution and metadata
        """
        session_id = f"sess-{uuid.uuid4().hex[:8]}"
        print(f"\n🚀 Starting Multi-Agent Session: {session_id}\n")

        # Initialize state
        init_state = {
            "session_id": session_id,
            "problem": problem,
            "metadata": AgentMetaData(
                session_id=session_id,
                start_time=datetime.now(),
                model_name=self.model,
                temperature=self.temperature,
                base_url=self.base_url,
            ),
            "iteration_count": 0,
            "needs_revision": False,
            "feedback": "",
        }

        # Run graph
        final_state = self.graph.invoke(
            init_state,
            config={"configurable": {"thread_id": session_id}}
        )

        # Finalize metadata
        final_state["metadata"].end_time = datetime.now()

        print("\n✅ Process completed.\n")
        return final_state


# ============================================================
#  CONVENIENCE FUNCTION
# ============================================================

def run_multi_agent(
    problem: str,
    model: str = "llama3.2:1b",
    temperature: float = 0.3,
    base_url: Optional[str] = None,
    api_key: str = "EMPTY",
    max_iterations: int = 2,
    use_web_search: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run the multi-agent system.

    Args:
        problem: The question/problem to solve
        model: Model name
        temperature: Sampling temperature
        base_url: OpenAI-compatible base URL
        api_key: API key
        max_iterations: Maximum feedback iterations
        use_web_search: Whether to enable web search

    Returns:
        Final state dict with solution and metadata
    """
    agent = MultiAgent(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        max_iterations=max_iterations,
        use_web_search=use_web_search,
    )
    return agent.run(problem)


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    problem = """At a community picnic, Sam brings 3 boxes of oranges. Each box contains 4 bags, and each bag has 8 oranges. He gives away one-quarter of all his oranges to volunteers. Then he uses 12 oranges to make juice. He splits the remaining oranges equally among 5 friends.
How many oranges does each friend receive?"""

    result = run_multi_agent(
        problem=problem,
        use_web_search=False  # Disable for math problems
    )

    print("\n=== 🏁 FINAL RESULT ===\n")
    print(f"Answer: {result['structured_output'].model_answer}")
    print(f"\nMetadata:")
    print(f"  Total tokens: {result['metadata'].total_tokens}")
    print(f"  Total energy: {result['metadata'].total_energy_joules:.4f} J")
    print(f"  Average watts: {result['metadata'].average_watts:.4f} W")
    print(f"  Iterations: {result['metadata'].iterations}")

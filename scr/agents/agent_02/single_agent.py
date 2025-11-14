from ..base_agent import BaseAgent
from ..graph_state import GraphState, StructuredOutput
from ..utils import extract_final_answer
from langgraph.graph import StateGraph, END
from datetime import datetime
import uuid
import yaml
import os


class SingleAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts",
            "prompts.yaml"
        )
        with open(prompt_path, "r") as f:
            prompts = yaml.safe_load(f)
        return prompts["system_prompt"]

    def _agent_node(self, state: GraphState) -> GraphState:
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": state.problem}
        ]

        response = self.invoke(messages)

        full_content = response.get("content")
        final_answer = extract_final_answer(full_content)

        state.structured_output = StructuredOutput(
            model_answer=final_answer,
            reasoning_trace=full_content
        )
        state.iterations += response.get("iterations", 0)

        return state

    def _finalize_node(self, state: GraphState) -> GraphState:
        state.accumulate_agent_metadata(self.meta_data)
        state.end_time = datetime.now()
        return state

    def run(self, problem: str) -> GraphState:
        workflow = StateGraph(GraphState)

        workflow.add_node("agent", self._agent_node)
        workflow.add_node("finalize", self._finalize_node)

        workflow.set_entry_point("agent")
        workflow.add_edge("agent", "finalize")
        workflow.add_edge("finalize", END)

        graph = workflow.compile()

        initial_state = GraphState(
            session_id=str(uuid.uuid4()),
            problem=problem,
            structured_output=StructuredOutput(model_answer=None, reasoning_trace=None),
            start_time=datetime.now(),
            end_time=datetime.now(),
            model_name=self.model,
            temperature=self.temperature
        )

        final_state = graph.invoke(initial_state)

        return GraphState(**final_state)

"""Agent 执行轨迹采集

从 create_agent 返回的 messages 中提取 Thought → Action → Observation 链路。
"""

from dataclasses import dataclass, field
from typing import List, Any
import json
from langchain_core.messages import AIMessage, ToolMessage


@dataclass
class AgentStep:
    """单步执行记录"""
    step_number: int
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""


@dataclass
class AgentTrace:
    """完整执行轨迹"""
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: str = ""

    def to_dict(self) -> dict:
        return {
            "steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class TraceCollector:
    """从 create_agent 返回的 messages 中提取执行轨迹"""

    @staticmethod
    def from_messages(
        messages: List[Any], final_output: str = ""
    ) -> AgentTrace:
        """解析消息列表提取执行步骤

        create_agent 的消息序列:
          HumanMessage → AIMessage(tool_calls=[...]) → ToolMessage → AIMessage(tool_calls=[...]) → ... → AIMessage(content=最终回答)
        """
        trace = AgentTrace(final_answer=final_output)
        step_number = 0

        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    step_number += 1
                    step = AgentStep(
                        step_number=step_number,
                        action=tool_call.get("name", ""),
                        action_input=json.dumps(
                            tool_call.get("args", {}),
                            ensure_ascii=False,
                        ),
                    )

                    # 找对应的 ToolMessage
                    if i + 1 < len(messages) and isinstance(messages[i + 1], ToolMessage):
                        tool_msg = messages[i + 1]
                        if tool_msg.tool_call_id == tool_call.get("id"):
                            step.observation = str(tool_msg.content)

                    trace.steps.append(step)

        return trace

    @staticmethod
    def from_intermediate_steps(
        intermediate_steps: List[Any], final_output: str = ""
    ) -> AgentTrace:
        """兼容旧版 intermediate_steps 格式"""
        trace = AgentTrace(final_answer=final_output)

        for i, step in enumerate(intermediate_steps, start=1):
            action, observation = step

            agent_step = AgentStep(
                step_number=i,
                action=getattr(action, "tool", ""),
                action_input=str(getattr(action, "tool_input", "")),
                observation=str(observation) if observation else "",
            )

            log = getattr(action, "log", "")
            if log:
                if "Thought:" in log:
                    parts = log.split("Thought:", 1)
                    if len(parts) > 1:
                        thought_part = parts[1]
                        for marker in ["\nAction:", "\nObservation:"]:
                            if marker in thought_part:
                                thought_part = thought_part.split(marker, 1)[0]
                        agent_step.thought = thought_part.strip()

            trace.steps.append(agent_step)

        return trace

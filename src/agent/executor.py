"""Agent Executor 封装

使用 LangChain create_agent (1.3.x 新 API) 做工具路由。
"""

import logging
from typing import List, Any
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from src.config import settings
from src.agent.tools import ALL_TOOLS, set_retriever
from src.agent.trace import TraceCollector, AgentTrace, AgentStep
from src.rag.prompt import TOOL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class AgentExecutorWrapper:
    """LangChain Agent 封装 (使用 create_agent)"""

    def __init__(self, retriever=None, reranker=None):
        # 注入检索器到 tool 模块
        set_retriever(retriever, reranker)

        # 创建 LLM
        llm_kwargs = {
            "model": settings.llm_model,
            "temperature": 0,
        }
        if settings.effective_llm_base_url:
            llm_kwargs["base_url"] = settings.effective_llm_base_url

        self._llm = ChatOpenAI(
            openai_api_key=settings.effective_llm_api_key,
            **llm_kwargs,
        )

        # 创建 Agent (使用新版 create_agent API)
        self._agent = create_agent(
            model=self._llm,
            tools=ALL_TOOLS,
            system_prompt=TOOL_SYSTEM_PROMPT,
        )

    def run(
        self, question: str, chat_history: List[Any] | None = None
    ) -> dict:
        """执行 Agent 查询

        Returns:
            {
                "output": str,
                "trace": AgentTrace,
            }
        """
        try:
            # 构建消息列表
            messages = []
            if chat_history:
                messages.extend(chat_history)
            messages.append(HumanMessage(content=question))

            result = self._agent.invoke({"messages": messages})

            # 提取消息
            result_messages = result.get("messages", [])

            # 最终回答是最后一条 AIMessage
            output = ""
            for msg in reversed(result_messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    output = msg.content
                    break

            # 提取执行轨迹
            trace = TraceCollector.from_messages(
                messages=result_messages,
                final_output=output,
            )

            return {
                "output": output,
                "trace": trace,
            }

        except Exception as e:
            logger.error(f"Agent 执行失败: {e}")
            return {
                "output": f"处理请求时出错: {str(e)}",
                "trace": AgentTrace(final_answer=f"Error: {e}"),
            }

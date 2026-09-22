"""极简工具调用 Agent：模型决策 → 执行工具 → 回传结果 → 继续决策。

安装依赖：python -m pip install openai
PowerShell 配置：$env:OPENAI_API_KEY = "你的密钥"
运行：python agent_tool_loop.py
示例任务：请用工具计算 123 加 456，再把结果加上 789。
可选环境变量：OPENAI_MODEL、OPENAI_BASE_URL（需支持工具调用）。
"""

import json
import os

from openai import OpenAI


def add(a, b):
    # 工具是普通 Python 函数；参数来自模型，执行前仍需校验。
    if type(a) not in (int, float) or type(b) not in (int, float):
        raise ValueError("a 和 b 必须是数字。")
    return a + b


# 工具描述提供给模型；函数映射供本地程序执行，两者通过名称对应。
TOOLS = [{
    "type": "function",
    "function": {
        "name": "add",
        "description": "计算两个数字的和；需要做加法时使用。",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}]
TOOL_FUNCTIONS = {"add": add}


def agent_loop(client, task, max_steps=5):
    # 历史保存任务、模型的工具调用请求以及工具执行结果。
    messages = [
        {"role": "system", "content": "你是一个简洁可靠的助手。加法请调用工具，根据工具结果回答。"},
        {"role": "user", "content": task},
    ]
    for _ in range(max_steps):
        # 模型自行决定：直接回答，或请求一次/多次工具调用。
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        message = choice.message
        # 截断等异常结束不能当作完整回答或完整工具请求处理。
        if choice.finish_reason not in ("stop", "tool_calls"):
            raise RuntimeError(f"模型未正常完成输出：{choice.finish_reason}")
        if not message.tool_calls:
            return message.content or "模型未返回文本回答。"

        # 必须先保存包含 tool_calls 的助手消息，再逐一追加对应工具结果。
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            try:
                # 只执行注册过的函数，不用 eval 执行模型生成的代码。
                function = TOOL_FUNCTIONS[call.function.name]
                arguments = json.loads(call.function.arguments)
                result = {"result": function(**arguments)}
            except (KeyError, TypeError, ValueError) as exc:
                # 将参数或工具错误反馈给模型，使其有机会在下一轮修正。
                result = {"error": str(exc)}
            print(f"工具 {call.function.name}：{result}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,  # 用调用编号关联请求与结果。
                "content": json.dumps(result, ensure_ascii=False),
            })
        # 携带工具结果进入下一轮；由模型决定继续调用还是给出最终答案。
    raise RuntimeError(f"已达到 {max_steps} 轮模型调用上限，尚未得到最终回答。")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("请先设置环境变量 OPENAI_API_KEY。")
    task = input("请输入任务：").strip()
    if not task:
        raise SystemExit("任务不能为空。")
    # SDK 读取环境变量配置；请求失败直接报错，不与 Agent 迭代混淆。
    client = OpenAI(timeout=60.0, max_retries=0)
    print(f"\n最终回答：{agent_loop(client, task)}")


if __name__ == "__main__":
    main()

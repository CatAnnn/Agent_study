"""极简 Agent：接收任务 → 调用模型 → 人工验收 → 携带反馈重试。

安装依赖：python -m pip install openai
PowerShell 配置：$env:OPENAI_API_KEY = "你的密钥"
运行：python agent_loop.py
可选环境变量：OPENAI_MODEL、OPENAI_BASE_URL（兼容接口地址）。
"""

import os

from openai import OpenAI


def main():
    # 密钥从环境变量读取，避免直接写进代码；缺少密钥时立即退出。
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("请先设置环境变量 OPENAI_API_KEY。")
    # SDK 自动读取密钥和可选的 OPENAI_BASE_URL。
    # 设置请求超时并关闭 SDK 自动重试；下方循环负责根据反馈改进回答。
    client = OpenAI(timeout=60.0, max_retries=0)
    # 获取本次任务，去掉首尾空白；空任务无需调用模型。
    task = input("请输入任务：").strip()
    if not task:
        print("任务不能为空。")
        return

    # 对话历史就是最简单的状态，后续调用会携带之前的回答和反馈。
    # system 说明助手的行为要求，user 保存用户提出的原始任务。
    messages = [
        {"role": "system", "content": "你是一个简洁可靠的助手，请完成任务，并根据反馈修正回答。"},
        {"role": "user", "content": task},
    ]
    # 每轮执行“生成回答 → 人工验收 → 记录反馈”，最多调用模型 5 次。
    # 这是人工反馈循环；尚未实现由模型选择工具、执行工具并回传结果。
    for step in range(1, 6):
        # 将完整历史发送给模型；可用 OPENAI_MODEL 环境变量指定模型名。
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=messages,
        )
        # 读取第一条候选回复的文本，文本为空时使用空字符串。
        answer = response.choices[0].message.content or ""
        print(f"\n第 {step} 轮回答：\n{answer}")
        # 先保存本轮回答，让模型下一轮知道用户反馈针对的是哪次输出。
        messages.append({"role": "assistant", "content": answer})
        # 由用户担任验收者：输入“通过”结束，否则将输入作为修改意见。
        feedback = input("输入“通过”结束，或输入修改意见：").strip()
        # 空反馈只重新读取输入，不消耗一次模型调用。
        while not feedback:
            feedback = input("请输入“通过”或具体修改意见：").strip()
        # 正常终止条件：用户明确认可回答。
        if feedback == "通过":
            print("任务完成。")
            return
        # 将验收反馈交回模型，并限制轮数，避免无限循环。
        messages.append({"role": "user", "content": f"请根据以下反馈修改：{feedback}"})
    # 达到上限仍未通过时如实报告，不能将“循环结束”等同于“任务成功”。
    print("已达到 5 轮上限，任务尚未通过验收。")


# 直接运行文件时启动循环；作为模块导入时不自动运行。
if __name__ == "__main__":
    main()
    

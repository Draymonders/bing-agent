import os
# from src.llm import basic_model, reasoning_model, EmbeddingModel

# from src.agents.search_agent import search_by_pdf, search_by_web, graph, search_by_url_agent
from src.utils.util import show_graph
# from src.agents.classify_agent import classify_text
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.shot.few_shot import few_shot
# from src.agents.chat_agent import chat_with_agent, chat_with_msgs, trimmed_language_demo

# from src.agents.tool_agent import chat_with_agent, search_by_input, chat_with_msgs


if __name__ == '__main__':
    
    # search_by_pdf("./datas/nke-10k-2023.pdf")
    # classify_text("我讨厌你")
    # thread_id = "123"
    # chat_with_agent("你好，我是大饼", thread_id)
    # chat_with_agent("我的名字是？", thread_id)
    # chat_with_agent("我是谁？", thread_id+"1")
    # chat_with_msgs([
    #     HumanMessage(content="你好，我是大饼"),
    #     AIMessage(content="你好，我是AI助手，我可以回答你的问题。"),
    #     HumanMessage(content="我是谁？"),
    # ])
    # trimmed_language_demo()
    # print(os.environ["TAVILY_API_KEY"])
    # print("===555===")
    # search_by_input("黄金价格")
    # thread_id = "123"
    # chat_with_agent("hi im bob! and i live in san francisco", thread_id)
    # chat_with_agent("what's the weather where I live?", thread_id)

    # thread_id="456"
    # chat_with_msgs([HumanMessage(content="what's the weather in in san francisco")], thread_id)

    # search_by_web(
    #     "https://draymonders.github.io/cs-life/machine-learn/agent/langmanus/",
    #     "详细介绍下planner的作用",
    #     debug=False,
    # )
    # query = (
    #     "LangManus分为几个组件?"
    #     "介绍下LangManus的planner"
    # )
    # search_by_url_agent("https://draymonders.github.io/cs-life/machine-learn/agent/langmanus/", query, "123")
    # show_graph("./datas/rag_graph.png", graph=graph)

    few_shot()
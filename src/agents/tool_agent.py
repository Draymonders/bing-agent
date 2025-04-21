import os
from src.llm import basic_model, reasoning_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

# os.environ["tavily_api_key"] = "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"
memory = MemorySaver()
search = TavilySearchResults(max_results=2, )
tools=[search]
agent_executor = create_react_agent(basic_model, tools, checkpointer=memory)

def search_by_input(input):
    res = search.invoke({"query": input})
    print(res)

def chat_with_agent(input: str, thread_id: str):
    config = {
        "configurable": {"thread_id": thread_id}
    }
    for step in agent_executor.stream(
        { "messages": [HumanMessage(content=input)] },
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

def chat_with_msgs(msgs: list, thread_id: str):
    config = {
        "configurable": {"thread_id": thread_id}
    }
    # model_with_tools = basic_model.bind_tools(tools)
    resp = agent_executor.invoke({
        "messages": msgs
    }, config)
    for msg in resp['messages']:
        msg.pretty_print()
    # print("content: ", resp.content)
    # print("toolCalls: ", resp.tool_calls)
       

if __name__ == '__main__':
    search_by_input("黄金价格")
    print("===")

    thread_id = "123"
    chat_with_agent("hi im bob! and i live in san francisco", thread_id)
    chat_with_agent("what's the weather where I live?", thread_id)
    print("===")

    thread_id="456"
    chat_with_msgs([HumanMessage(content="what's the weather in in san francisco")], thread_id)

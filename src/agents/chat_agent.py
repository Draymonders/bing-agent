from src.llm import basic_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    resp = model = basic_model.invoke(state['messages'])
    return {"messages": resp}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def chat_with_agent(input: str, thread_id):
    input_msgs = [HumanMessage(content=input)]
    config = {"configurable": {
        "thread_id": thread_id,
    }}
    output = app.invoke({"messages": input_msgs}, config)
    output["messages"][-1].pretty_print()

def chat_with_msgs(msgs: list):
    resp = basic_model.invoke(msgs)
    print(resp.content)

def trimmed_language_demo():
    def dummy_token_counter(messages: list[BaseMessage]) -> int:
        cnt = 0
        default_content_len = 10
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                cnt += len(content) * default_content_len
            if isinstance(content, str):
                cnt += len(content)
        return cnt
    
    trimmer = trim_messages(
        max_tokens=60,
        strategy="last",
        allow_partial=False,
        token_counter=dummy_token_counter, # 豆包未实现 token counter
        # token_counter=basic_model.get_num_tokens,
        include_system=True,
        start_on="human",
    )

    msgs = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]
    res = trimmer.invoke(msgs)
    for msg in res:
        msg.pretty_print()


if __name__ == '__main__':
    # chat_with_agent("你好，你是？")
    # 连续会话
    chat_with_msgs([
        HumanMessage(content="你好，我是大饼"),
        AIMessage(content="你好，我是AI助手，我可以回答你的问题。"),
        HumanMessage(content="我是谁？"),
    ])
    print("===")

    thread_id = "123"
    # 会话连续性
    chat_with_agent("你好，我是大饼", thread_id)
    chat_with_agent("我的名字是？", thread_id)
    print("===")

    # 新的会话
    chat_with_agent("我是谁？", thread_id+"1")
    print("===")

    

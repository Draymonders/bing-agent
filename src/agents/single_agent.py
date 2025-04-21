from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
from src.llm import basic_model, reasoning_model


def auditor(model, question: str):
    # 简单模型调用
    messages = [
        SystemMessage(content="你是一个平台审核员，审核当前信息是否合规，不合规的情况请给出理由"),
        HumanMessage(content=question),
    ]

    return model.invoke(messages).content

def stream_worker(model, question: str):
    # 流式调用
    messages = [
        SystemMessage(content="你是一个平台审核员，审核当前信息是否合规，不合规的情况请给出理由"),
        HumanMessage(content=question),
    ]
    answer = ''
    for chunk in model.stream(messages):
        answer += chunk.content
        yield chunk.content

    return answer

def prompt_worker(model, question: str):
    # 模板变量注入
    sys_template = "Translate the following from English into {language}"
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", sys_template),
            ("user", "{text}"),
        ]
    )

    prompt = prompt_template.invoke({"language": "Chinese", "text": question})
    print(prompt)
    return model.invoke(prompt.messages).content

def structed_outputer(llm, question: str):
    class Sentiment(BaseModel):
        """情感倾向"""

        final_output: str = Field(description="情感倾向，取值为正向或负向")
    
    # 结构化输出
    # messages = [
    #     SystemMessage(content=""),
    #     HumanMessage(content=question),
    # ]
    llm_model = llm_model.with_structured_output(Sentiment)
    answer = llm_model.invoke("判断一下内容是积极还是消极的，以json格式输出")
    return answer.content

if __name__ == '__main__':
    print(structed_outputer(basic_model, "我好讨厌你"))
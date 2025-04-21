from langchain_core.prompts import ChatPromptTemplate
from src.llm import basic_model, reasoning_model
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

只提取 'Classification' function 的相关属性，以Json形式返回

Passage:
{input}
"""
)

class Classification(BaseModel):
    sentiment: str = Field(description="文本的情感，示例：积极、消极、中性")
    language: str = Field(description="文本的语言")

def classify_text(text: str):
    # 实践结果：豆包v1.5和deepseek的tool调用还不是很好！
    prompt = tagging_prompt.invoke({"input": text})
    print(prompt)
    print(reasoning_model.invoke(prompt).content)

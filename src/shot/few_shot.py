
from src.llm import basic_model, EmbeddingModel
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

samples = [
  {
    "flower_type": "玫瑰",
    "occasion": "爱情",
    "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
  },
  {
    "flower_type": "康乃馨",
    "occasion": "母亲节",
    "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
  },
  {
    "flower_type": "百合",
    "occasion": "庆祝",
    "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
  },
  {
    "flower_type": "向日葵",
    "occasion": "鼓励",
    "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
  }
]

def few_shot():
    template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}"
    prompt_sample = PromptTemplate(
        input_variables=["flower_type", "occasion", "ad_copy"],
        template=template
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        samples,
        EmbeddingModel(),
        # HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese"),
        Chroma,
        k=2
    )

    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=prompt_sample,
        suffix="帮我生成一份文案",
        # suffix="鲜花类型: {flower_type}\n场合: {occasion}",
        input_variables=["flower_type", "occasion"]
    )

    query = prompt.format(flower_type="兰花", occasion="独特新奇")
    print("query:", query)
    result = basic_model.invoke(query)
    print("llm resp:", result.content)

from volcenginesdkarkruntime import Ark
from langchain_core.embeddings.embeddings import Embeddings
from .conf import LLMConf, volce_endpoint
from typing import List

embedding_conf = LLMConf('4f16fa15-dc20-4a33-8443-a13b08e418db', volce_endpoint, 'ep-20250417202534-sgkbf')

class EmbeddingModel(Embeddings):
    def __init__(self, conf=embedding_conf):
        super().__init__()
        self.conf = conf
        self.cli = Ark(api_key=self.conf.api_key, base_url=self.conf.base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.get_embedding(text)

    def get_embedding(self, query_str,conf=embedding_conf):
        # 生成2048维向量
        resp = self.cli.multimodal_embeddings.create(
            model=self.conf.model,
            input=[
                {
                    "type": "text",
                    "text": query_str
                }
            ]
        )
        return resp.data['embedding']
EmbeddingDim = 2048
volce_endpoint = 'https://ark.cn-beijing.volces.com/api/v3'


class LLMConf:
    api_key: str = ''
    base_url: str = ''
    model: str = ''

    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
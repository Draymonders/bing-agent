# import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from .conf import LLMConf, volce_endpoint
from .type import LLMType


basic_conf = LLMConf('7379a4ce-03a8-46a1-8a12-9776705c67a2', volce_endpoint, 'ep-20250102142526-p6gcb') 
vision_conf = LLMConf('7379a4ce-03a8-46a1-8a12-9776705c67a2', volce_endpoint, 'ep-20250124115344-4sgfd')
reasoning_conf = LLMConf('ed84051d-e959-4d8b-ab5e-16680b8eb76f', volce_endpoint, 'ep-20250206114635-gfxcl')   

def get_llm(conf: LLMConf, model_type: LLMType, temperature = 0.0, streaming = False, verbose = True):
    if model_type == "resoning":
        return ChatDeepSeek(model=conf.model, api_key=conf.api_key, base_url=conf.base_url, temperature=temperature, streaming=streaming, verbose=verbose)
    return ChatOpenAI(model=conf.model, api_key=conf.api_key, base_url=conf.base_url, temperature=temperature, streaming=streaming, verbose=verbose)


basic_model = get_llm(basic_conf, "basic")
vision_model = get_llm(vision_conf, "vision")
reasoning_model = get_llm(reasoning_conf, "reasoning")


if __name__ == '__main__':
    # resp = basic_model.invoke("你好")
    # print(resp)
    resp = get_embedding("你好")
    print(resp)
import faiss
import bs4
import json
from IPython.display import Image, display
from typing_extensions import List, TypedDict

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.llm.conf import EmbeddingDim
from src.llm import EmbeddingModel, basic_model

embed_model = EmbeddingModel()
index = faiss.IndexFlatL2(EmbeddingDim)
vector_store = FAISS(
    embedding_function=embed_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
memory = MemorySaver()

@tool(response_format="content_and_artifact")
def retrieve_vectors_tool(query: str):
    """通过query检索相似的内容"""

    docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source {doc.metadata} Content: {doc.page_content}") 
        for doc in docs
    )
    return serialized, docs

agent_executor = create_react_agent(basic_model, [retrieve_vectors_tool], checkpointer=memory)




# 尝试langgraph构建检索问答
class State(TypedDict):
    query: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    docs = vector_store.similarity_search(state["query"], k=3)
    return {
        "context": docs
    }

def generate(state: State):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
        ("human", "Question: {question} , Context: {context} "),
    ])
    prompt = prompt_template.invoke({
        "question": state["query"],
        "context": "\n - ".join([doc.page_content for doc in state["context"]])
    })
    resp = basic_model.invoke(prompt)
    resp.pretty_print()
    return {
        "answer": resp.content
    }

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

@chain
def retriever(query: str) -> list[Document]:
    """Retrieve documents from the vector store."""
    return vector_store.similarity_search(query, k=1)

def search_by_pdf(file_path, debug=True):
    # 加载pdf，并做切分
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # 1000个字符切割
        chunk_overlap=200, # 有200个字符重叠
        add_start_index=True, # 增加起始索引
    )
    all_splits = text_splitter.split_documents(docs)
    if debug:
        print(f"生成{len(all_splits)}个切片")
        all_splits = all_splits[:10]
    # 向量化存储
    
    vector_store.add_documents(all_splits)
    if debug:
        print(f"向量化存储done")
    
    # 向量化检索
    queries = [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?"
    ]
    res = retriever.batch(queries)
    for i, docs in enumerate(res):
        print(f"第{i+1}个query, 查询出{len(docs)}个doc")
        for doc in docs:
            # 有id、metadata、page_content三个字段
            print(f"doc_id {doc.id}")

def store_vectors(url: str):
    # 加载网页数据
    bs4_strainer = bs4.SoupStrainer(
        class_=["md-content"],
        attrs={"data-md-component": "content"}

    )
    loader = WebBaseLoader(web_paths=(url,), bs_kwargs={"parse_only": bs4_strainer})
    docs = loader.load()
    print(f"加载 {len(docs)} 个文档")
    print(f"第一篇doc共有 {len(docs[0].page_content)}个字符")
    # print(docs[0].page_content[:100].strip())

    # 切分文档为切片
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # 1000个字符切割
        chunk_overlap=200, # 有200个字符重叠
        add_start_index=True, # 增加起始索引
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"生成{len(all_splits)}个切片")

    # 存到向量库
    doc_ids = vector_store.add_documents(all_splits)
    print(f"生成{len(doc_ids)}个doc_ids")

def search_by_web(url: str, query: str, debug=False):
    
    # 加载网页数据并存储到向量库
    store_vectors(url)
    
    # 相似检索放到Prompt中
    # docs = vector_store.similarity_search(query, k=3)
    # if debug:
    #     for doc in docs:
    #         print(f"*** doc_id: {doc.id} ***")
    #         print(doc.page_content)
    #         print("### ###")
    # print(f"查询出{len(docs)}个doc")
    # prompt = prompt_template.invoke({
    #     "question": query,
    #     "context": "\n - ".join([doc.page_content for doc in docs])
    # })

    # 调用模型
    # resp = basic_model.invoke(prompt)
    # resp.pretty_print()
    result = graph.invoke({"query": query})
    print(f'query: {result["query"]}')
    print(f'answer: {result["answer"]}')
    print(f'context: {len(result["context"])}')
    # print(result["context"][0].page_content[:50])
    # print(result["answer"])
    # print(json.dumps(result, ensure_ascii=False))

def search_by_url_agent(url: str, query, thread_id: str):
    store_vectors(url)
    
    for event in agent_executor.stream(
        { "messages": [{"role": "user", "content": query}] },
        stream_mode="values",
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    ):
        event["messages"][-1].pretty_print()

if __name__ == "__main__":
    search_by_pdf("../../data/nike.pdf")
    print("====")

    search_by_web(
        "https://draymonders.github.io/cs-life/machine-learn/agent/langmanus/",
        "详细介绍下planner的作用"
    )
    print("====")
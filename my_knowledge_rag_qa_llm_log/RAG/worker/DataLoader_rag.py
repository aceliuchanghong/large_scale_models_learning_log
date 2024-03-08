from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
import bs4

from my_knowledge_rag_qa_llm_log.RAG.config.config import proxies, test_url


def siteLoader(url="https://lilianweng.github.io/posts/2023-06-23-agent/", my_proxy=False):
    proxy = None
    if my_proxy:
        proxy = proxies
    # loader = WebBaseLoader(
    #     web_paths=(url,),
    #     bs_kwargs=dict(
    #         parse_only=bs4.SoupStrainer(
    #             class_=("post-content", "post-title", "post-header")
    #         )
    #     ), proxies=proxy
    # )
    loader = WebBaseLoader(
        web_paths=(url,)
        , proxies=proxy
    )
    docs = loader.load()
    return docs


def fileLoader(file="../README.md"):
    loader = TextLoader(file)
    docs = loader.load()
    return docs


if __name__ == '__main__':
    print(siteLoader(test_url))
    print(siteLoader())
    print(fileLoader())

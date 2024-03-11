# 使用 Chroma 向量存储和 OpenAIEmbeddings 模型在单个命令中嵌入和存储所有文档拆分
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

"""
https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
"""

from my_knowledge_rag_qa_llm_log.RAG.config.config import test_url, http_client
from my_knowledge_rag_qa_llm_log.RAG.worker.DataLoader_rag import siteLoader
from my_knowledge_rag_qa_llm_log.RAG.worker.Spliter_rag import split_docs


def save_splits(all_splits):
    # OpenAIEmbeddings 需要 OPENAI_API_KEY 此处还需要代理
    # vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(http_client=http_client))

    """
    Q:一直报错
    No sentence-transformers model found with name /mnt/chatGLM/embedding/text2vec-large-chinese. Creating a new one with MEAN pooling.
    A:https://huggingface.co/GanymedeNil/text2vec-large-chinese/discussions/10
    别人上传的模型少了文件
    """
    MODEL_PATH = os.environ.get('text2vec', 'GanymedeNil/text2vec-large-chinese')
    Embeddings_Models = {"text2vec": MODEL_PATH}
    model_kwargs = {'device': 'cpu'}
    # model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {'normalize_embeddings': False}

    # HuggingFaceEmbeddings
    vectorstore = Chroma.from_documents(documents=all_splits,
                                        embedding=HuggingFaceEmbeddings(model_name=Embeddings_Models["text2vec"],
                                                                        model_kwargs=model_kwargs,
                                                                        encode_kwargs=encode_kwargs))
    return vectorstore


if __name__ == '__main__':
    docs1 = siteLoader(test_url)
    all_splits1 = split_docs(docs1)
    vectorstore1 = save_splits(all_splits1)
    print(vectorstore1)

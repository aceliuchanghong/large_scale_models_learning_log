# 使用 Chroma 向量存储和 OpenAIEmbeddings 模型在单个命令中嵌入和存储所有文档拆分
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
"""
https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
"""

from my_knowledge_rag_qa_llm_log.RAG.config.config import test_url, http_client
from my_knowledge_rag_qa_llm_log.RAG.worker.DataLoader_rag import siteLoader
from my_knowledge_rag_qa_llm_log.RAG.worker.Spliter_rag import split_docs


def save_splits(all_splits):

    # OpenAIEmbeddings 需要 OPENAI_API_KEY 此处还需要代理
    # vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(http_client=http_client))

    # HuggingFaceEmbeddings
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())
    return vectorstore


if __name__ == '__main__':
    docs1 = siteLoader(test_url)
    all_splits1 = split_docs(docs1)
    vectorstore1 = save_splits(all_splits1)

    vectors = vectorstore1.get_vectors()
    for vector in vectors:
        print(vector)
    for doc_id, vector in vectorstore1.items():
        print(f"Document ID: {doc_id}")
        print(f"Vector: {vector}")

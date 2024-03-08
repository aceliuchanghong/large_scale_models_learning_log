from langchain_text_splitters import RecursiveCharacterTextSplitter

from my_knowledge_rag_qa_llm_log.RAG.config.config import test_url
from my_knowledge_rag_qa_llm_log.RAG.worker.DataLoader_rag import fileLoader, siteLoader


# 将文档拆分为 1000 个字符的块，块之间有 200 个字符重叠。重叠有助于减少将语句与与之相关的重要上下文分开的可能性
def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits


if __name__ == '__main__':
    docs1 = siteLoader(test_url)
    docs2 = fileLoader()
    all_splits1 = split_docs(docs1)
    all_splits2 = split_docs(docs2)
    print(all_splits1)
    print(all_splits2)

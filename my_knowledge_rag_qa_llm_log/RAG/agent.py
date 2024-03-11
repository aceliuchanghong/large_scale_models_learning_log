from langchain.prompts import PromptTemplate
import os
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from my_knowledge_rag_qa_llm_log.RAG.worker.utils import get_llm, load_embedding_model, load_documents, store_chroma, \
    format_docs


def get_rag(q="1+1等于几"):
    prompt = PromptTemplate.from_template("""根据下面的上下文（context）内容回答问题。
    如果你不知道答案，就回答不知道，不要试图编造答案。
    答案最多3句话，保持答案简洁。
    {context}
    问题：{question}
    """)
    llm = get_llm()
    # 加载embedding模型
    embeddings = load_embedding_model()
    # 加载数据库
    if not os.path.exists('VectorStore'):
        documents = load_documents()
        db = store_chroma(documents, embeddings)
    else:
        db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)

    retriever = db.as_retriever()

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain.invoke(q)


if __name__ == '__main__':
    get_rag()

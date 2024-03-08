from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic

from my_knowledge_rag_qa_llm_log.RAG.config.config import http_client, test_url
from my_knowledge_rag_qa_llm_log.RAG.worker.DataLoader_rag import siteLoader
from my_knowledge_rag_qa_llm_log.RAG.worker.Spliter_rag import split_docs
from my_knowledge_rag_qa_llm_log.RAG.worker.Store_rag import save_splits


# 搜索文档
def search_vectorstore_generate(vectorstore):
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, http_client=http_client)

    # ANTHROPIC_API_KEY
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.2, max_tokens=1024)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    response = rag_chain.invoke("summary this in 20 words")

    for chunk in rag_chain.stream("summary this in 20 words"):
        print(chunk, end="", flush=True)

    return response


if __name__ == '__main__':
    docs1 = siteLoader(test_url)
    all_splits1 = split_docs(docs1)
    vectorstore1 = save_splits(all_splits1)
    response1 = search_vectorstore_generate(vectorstore1)

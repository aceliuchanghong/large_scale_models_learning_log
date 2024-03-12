from langchain.prompts import PromptTemplate
import os
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from my_knowledge_rag_qa_llm_log.RAG.worker.utils import get_chatglm_llm, load_embedding_model, load_documents, \
    store_chroma, format_docs, get_claude_llm


class bot_agent:
    def __init__(self):
        self.prompt = PromptTemplate.from_template("""根据下面的上下文（context）内容回答问题。
                如果你不知道答案，就回答不知道，不要试图编造答案。
                答案最多3句话，保持答案简洁。
                {context}
                问题：{question}
                """)
        self.embeddings = load_embedding_model()
        self.llm = get_chatglm_llm()
        if not os.path.exists('VectorStore'):
            documents = load_documents()
            self.db = store_chroma(documents, self.embeddings)
        else:
            self.db = Chroma(persist_directory='VectorStore', embedding_function=self.embeddings)
        self.retriever = self.db.as_retriever()

    def get_rag(self, q="1+1等于几"):
        """
        对于回答里面如果出现中文夹杂英文情况,主要是 langchain 里面的修饰过的 prompt 是英文,所以模型就会英文回答
        解决办法就是不用langchain做最后一步的回答,自己写一个post请求
        :param q:
        :return:
        """
        rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )
        return rag_chain.invoke(q)


if __name__ == '__main__':
    my_bot_agent = bot_agent()
    my_bot_agent.get_rag()

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import Chroma
from my_knowledge_rag_qa_llm_log.RAG.config.config import proxies, test_url
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.llms import ChatGLM


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_documents(directory="documents"):
    """
    加载 documents 下的文件，进行拆分
    :param directory:
    :return:
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    # 将文档拆分为 1000 个字符的块，块之间有 200 个字符重叠。重叠有助于减少将语句与与之相关的重要上下文分开的可能性
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs


def siteLoader(url="https://lilianweng.github.io/posts/2023-06-23-agent/", my_proxy=False):
    """
    :param url:
    :param my_proxy:
    :return:
    """
    proxy = None
    if my_proxy:
        proxy = proxies
    loader = WebBaseLoader(
        web_paths=(url,)
        , proxies=proxy
    )
    docs = loader.load()
    return docs


def fileLoader(file="../README.md"):
    """
    :param file:
    :return:
    """
    loader = TextLoader(file)
    docs = loader.load()
    return docs


def load_embedding_model():
    """
    加载 embedding 模型
    :return:
    Q:一直报错
    No sentence-transformers model found with name /mnt/chatGLM/embedding/text2vec-large-chinese. Creating a new one with MEAN pooling.
    A:https://huggingface.co/GanymedeNil/text2vec-large-chinese/discussions/10
    别人上传的模型少了文件

    """
    model_kwargs = {'device': 'cpu'}
    # model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {'normalize_embeddings': False}
    MODEL_PATH = os.environ.get('text2vec', 'GanymedeNil/text2vec-large-chinese')
    Embeddings_Models = {"text2vec": MODEL_PATH}

    return HuggingFaceEmbeddings(
        model_name=Embeddings_Models["text2vec"],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    """
    使得文档向量化，存入向量数据库
    :param docs:
    :param embeddings:
    :param persist_directory:
    :return:
    """
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db


def get_llm():
    llm = ChatGLM(
        endpoint_url="http://127.0.0.1:8000",
        max_token=80000,
        top_n=0.9
    )
    return llm

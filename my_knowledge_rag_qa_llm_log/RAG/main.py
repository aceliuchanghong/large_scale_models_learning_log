import gradio as gr
import os
import time
import random

from my_knowledge_rag_qa_llm_log.RAG.worker.DataLoader_rag import textLoader, fileLoader
from my_knowledge_rag_qa_llm_log.RAG.worker.Retriever_Generator_rag import search_vectorstore_generate
from my_knowledge_rag_qa_llm_log.RAG.worker.Spliter_rag import split_docs
from my_knowledge_rag_qa_llm_log.RAG.worker.Store_rag import save_splits


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=True)


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


def add_file2(file):
    docs1 = fileLoader(file)
    all_splits1 = split_docs(docs1)
    vectorstore1 = save_splits(all_splits1)

    return 0


def bot_response(history):
    text = " ".join(history)
    docs = textLoader(text)
    all_splits = split_docs(docs)
    vectorstore = save_splits(all_splits)
    response = search_vectorstore_generate(vectorstore)
    return response


def bot(history):
    message = history[-1][0]
    # 如果最后一条消息是一个元组类型，通常这种情况下代表文件上传成功，因为上传成功后往往返回的是元组数据，比如 (文件名, 文件大小)
    if isinstance(message, tuple):
        response = "文件上传成功"
    else:
        # response = bot_response(history)
        response = random.choice(["How are you?", "I love you", "I'm very hungry"])
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "imgs", "avatar.jpg"))),
    )

    with gr.Row():
        btn = gr.UploadButton("📁", scale=3, file_types=["text"])
        txt = gr.Textbox(
            scale=20,
            show_label=False,
            label="chatInfo",
            placeholder="输入文字",
            container=False,
        )
        btn_submit = gr.Button(scale=6, value="Generate", variant="primary")
        btn_clear = gr.Button(scale=2, value="Clear", variant="secondary")

        # 上传文件处理逻辑
        file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
            bot, chatbot, chatbot, api_name="bot_file_response"
        )
        # 发送文本 点击按钮 处理逻辑
        btn_submit.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
            bot, chatbot, chatbot, api_name="bot_text_response"
        )
        # 发送文本 enter 处理逻辑
        txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
            bot, chatbot, chatbot, api_name="bot_text_response"
        )
        # 清除历史处理逻辑
        btn_clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == '__main__':
    # 通常，在使用 gr.Blocks()创建UI组件时，这些组件将会被添加到一个队列中(将定义的UI组件添加到可视化界面的队列中)
    demo.queue()
    demo.launch(share=False)

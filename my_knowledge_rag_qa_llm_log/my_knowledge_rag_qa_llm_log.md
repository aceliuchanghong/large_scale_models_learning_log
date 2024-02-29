### RAG
检索增强生成（RAG(Retrieval Augmented Generation)）是指对大型语言模型输出进行优化，使其能够在生成响应之前引用训练数据来源之外的权威知识库。

RAG 将其扩展为能访问特定领域或组织的内部知识库，所有这些都无需重新训练模型。这是一种经济高效地改进 LLM 输出的方法，让它在各种情境下都能保持相关性、准确性和实用性。

### What it is
基于 ChatGLM3 等大语言模型与 Langchain 等应用框架实现，开源、可离线部署的检索增强生成(RAG)大模型知识库项目。

### How

加载文件 -> 读取文本 -> 文本分割 -> 文本向量化 -> 问句向量化 -> 在文本向量中匹配出与问句向量最相似的 top k个 -> 匹配出的文本作为上下文和问题一起添加到 prompt中 -> 提交给 LLM生成回答。

### Target
实现完全本地化推理的知识库增强方案, 重点解决数据安全保护，私域化部署的痛点

## install
```
git clone https://github.com/chatchat-space/Langchain-Chatchat.git
```
安装库+依赖
```
conda create -n myRagLLM python=3.11.7
conda activate myRagLLM
pip install -r requirements.txt 
pip install -r requirements_api.txt
pip install -r requirements_webui.txt 
```
模型下载
```
#久
git lfs install
git clone https://huggingface.co/THUDM/chatglm3-6b
git clone https://huggingface.co/BAAI/bge-large-zh
#chatglm3-6b大小 11.6GB 
#bge-large-zh 3.12GB
```
初始化知识库和配置文件
```
python copy_config_example.py
#model_config.py里面修改下模型下载的位置,我这儿是改了的
python init_database.py --recreate-vs
```
![img.png](..%2Fusing_files%2Fimgs%2Frag_qa%2Fimg.png)

启动
```
python startup.py -a
```

### Reference(参考文档)
* [google gemma hugging face](https://huggingface.co/google/gemma-7b)
* [知识库教学Github页面](https://github.com/chatchat-space/Langchain-Chatchat)
* [知识库教学中文指南](https://www.maxada.cn/?post=305)
* [自己的git库日志](https://github.com/aceliuchanghong/my_glm_log)


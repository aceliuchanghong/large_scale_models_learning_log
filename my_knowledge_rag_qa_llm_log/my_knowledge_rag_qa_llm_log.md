### RAG
检索增强生成（RAG(Retrieval Augmented Generation)）是指对大型语言模型输出进行优化，使其能够在生成响应之前引用训练数据来源之外的权威知识库。

RAG 将其扩展为能访问特定领域或组织的内部知识库，所有这些都无需重新训练模型。这是一种经济高效地改进 LLM 输出的方法，让它在各种情境下都能保持相关性、准确性和实用性。

### What it is
基于 ChatGLM3 等大语言模型与 Langchain 等应用框架实现，开源、可离线部署的检索增强生成(RAG)大模型知识库项目。

### How

加载文件 -> 读取文本 -> 文本分割 -> 文本向量化 -> 问句向量化 -> 在文本向量中匹配出与问句向量最相似的 top k个 -> 匹配出的文本作为上下文和问题一起添加到 prompt中 -> 提交给 LLM生成回答。

### Target
实现完全本地化推理的知识库增强方案, 重点解决数据安全保护，私域化部署的痛点



### Reference(参考文档)
* [Github页面](https://github.com/chatchat-space/Langchain-Chatchat)
* [中文指南](https://www.maxada.cn/?post=305)
* [LangChain-ChatGLM-Webui](https://github.com/X-D-Lab/LangChain-ChatGLM-Webui/blob/master/docs/deploy.md)
* [自己的git库日志](https://github.com/aceliuchanghong/my_glm_log)


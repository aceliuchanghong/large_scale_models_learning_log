### Target 
FineTune Mistral (使用 Quantization  和 LoRA（Low-Rank Adaptation）来微调 Mistral 7b)

![img_2.png](..%2Fusing_files%2Fimgs%2Ffine_tune%2Fimg_2.png)

见上图:LoRA 固定模型参数(W0)不变,将小的可训练适配器层(ΔW = BA)附加到模型上,并且只训练适配器

### Concept
全微调（Full Fine-tuning）：全微调是指对整个预训练模型进行微调，包括所有的模型参数。在这种方法中，预训练模型的所有层和参数都会被更新和优化，以适应目标任务的需求。这种微调方法通常适用于任务和预训练模型之间存在较大差异的情况，或者任务需要模型具有高度灵活性和自适应能力的情况。Full Fine-tuning需要较大的计算资源和时间，但可以获得更好的性能。

部分微调（Repurposing）：部分微调是指在微调过程中只更新模型的顶层或少数几层，而保持预训练模型的底层参数不变。这种方法的目的是在保留预训练模型的通用知识的同时，通过微调顶层来适应特定任务。Repurposing通常适用于目标任务与预训练模型之间有一定相似性的情况，或者任务数据集较小的情况。由于只更新少数层，Repurposing相对于Full Fine-tuning需要较少的计算资源和时间，但在某些情况下性能可能会有所降低。

监督微调（Supervised Fine-tuning）：监督微调是指在进行微调时使用有标签的训练数据集。这些标签提供了模型在微调过程中的目标输出。在监督微调中，通常使用带有标签的任务特定数据集，例如分类任务的数据集，其中每个样本都有一个与之关联的标签。通过使用这些标签来指导模型的微调，可以使模型更好地适应特定任务。

无监督微调（Unsupervised Fine-tuning）：无监督微调是指在进行微调时使用无标签的训练数据集。这意味着在微调过程中，模型只能利用输入数据本身的信息，而没有明确的目标输出。这些方法通过学习数据的内在结构或生成数据来进行微调，以提取有用的特征或改进模型的表示能力。
### Steps
1. 准备数据集：收集和准备与目标任务相关的训练数据集。确保数据集质量和标注准确性，并进行必要的数据清洗和预处理。
2. 选择预训练模型/基础模型：根据目标任务的性质和数据集的特点，选择适合的预训练模型。
3. 设定微调策略：根据任务需求和可用资源，选择适当的微调策略。考虑是进行全微调还是部分微调，以及微调的层级和范围。
4. 设置超参数：确定微调过程中的超参数，如学习率、批量大小、训练轮数等。这些超参数的选择对微调的性能和收敛速度有重要影响。
5. 初始化模型参数：根据预训练模型的权重，初始化微调模型的参数。对于全微调，所有模型参数都会被随机初始化；对于部分微调，只有顶层或少数层的参数会被随机初始化。
6. 进行微调训练：使用准备好的数据集和微调策略，对模型进行训练。在训练过程中，根据设定的超参数和优化算法，逐渐调整模型参数以最小化损失函数。
7. 模型评估和调优：在训练过程中，使用验证集对模型进行定期评估，并根据评估结果调整超参数或微调策略。这有助于提高模型的性能和泛化能力。
8. 测试模型性能：在微调完成后，使用测试集对最终的微调模型进行评估，以获得最终的性能指标。这有助于评估模型在实际应用中的表现。
9. 模型部署和应用：将微调完成的模型部署到实际应用中，并进行进一步的优化和调整，以满足实际需求。
### Log_imgs
![img.png](..%2Fusing_files%2Fimgs%2Ffine_tune%2Fimg.png)

![img_1.png](..%2Fusing_files%2Fimgs%2Ffine_tune%2Fimg_1.png)

### FQ
1. [知识遗忘](https://github.com/THUDM/ChatGLM-6B/issues/1148)(learning_rate调整)
2. 过拟合
3. 模型不够好(多是数据集的问题,或者训练收敛时候就可以停止了step)
### Reference(参考文档)
* [Mistral微调教程](https://medium.com/@codersama/fine-tuning-mistral-7b-in-google-colab-with-qlora-complete-guide-60e12d437cca)
* [微调中文教程git](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md)
* [微调中文教程博客](https://blog.csdn.net/bmfire/article/details/131064677)









LoRA（Low-Rank Adaptation)

![img_2.png](..%2Fusing_files%2Fimgs%2Ffine_tune%2Fimg_2.png)

见上图:LoRA 固定模型参数(W0)不变,将小的可训练适配器层(ΔW = BA)附加到模型上,并且只训练适配器

监督微调（Supervised Fine-tuning）：监督微调是指在进行微调时使用有标签的训练数据集。这些标签提供了模型在微调过程中的目标输出。在监督微调中，通常使用带有标签的任务特定数据集，例如分类任务的数据集，其中每个样本都有一个与之关联的标签。通过使用这些标签来指导模型的微调，可以使模型更好地适应特定任务。

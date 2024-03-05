### start

构建训练数据集 按照官方的资料，训练集的基本格式如下

```
{
	"conversations": 
	[
		{"role": "user",
		"content": "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞"
		}, 
		{"role": "assistant", 
		"content": "简约而不简单的牛仔外套，白色的衣身十分百搭。衣身多处有做旧破洞设计，打破单调乏味，增加一丝造型看点。衣身后背处有趣味刺绣装饰，丰富层次感，彰显别样时尚。"
		}
    ]
}
```
多轮对话的训练集的格式
```
  {
    "conversations": [
      {
        "role": "system",
        "content": "<system prompt text>"
      },
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      },
      // ... 多轮
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      }
    ]
  }
  // ...
```


- 微调代码：finetune_hf.py
```shell
python3 finetune_hf.py data/self_cognition ../chatglm3-6b configs/lora.yaml
```
- 推理代码：inference_hf.py
```shell
python3 inference_hf.py output/checkpoint-3000/ --prompt "你是谁?"
```
- 合并代码：model_export_hf.py
```shell
python3 model_export_hf.py ./output/checkpoint-3000/ --out-dir ./chatglm3-6b-01
```
常见参数
```python
@dataclass
class TrainingConfig:
    model_name: str = field(default="./chatglm3-6b-base", metadata={"help": 'Huggingface Name of the model you want to train'})
    data_path: str = field(default="formatted_samples.json", metadata={"help": 'Path towards your training data'})
    output_dir: str = field(default='./trained_model', metadata={"help": 'The output dir for logs and checkpoints'})
    training_recipe: str = field(default="lora", metadata={"help": "Lora Training or Full Training"})
    optim: str = field(default='paged_adamw_8bit', metadata={"help": 'The optimizer to be used'})
    batch_size: int = field(default=4, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    n_epochs: int = field(default=5, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) 
    learning_rate: float = field(default=1e-4, metadata={"help": 'The learning rate'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='cosine', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=1, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='epoch', metadata={"help": 'When to save checkpoints'})
    save_total_limit: int = field(default=3, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    fp16: bool = field(default=False, metadata={"help": 'Whether to use fp16 mixed precision training'})
    tokenizer_type: str = field(default="llama", metadata={"help": "Tokenizer type. Should be \"llama\" for llama models to address tokenizer issue"})
    trust_remote_code: str = field(default=True, metadata={"help": "Whether to trust remote code."})
    compute_dtype: torch.dtype = field(default=torch.float16, metadata={"help":"Compute Datatype for models, either float16 or float32."})
    max_tokens: int = field(default=4096, metadata={"help":"Max tokens"})
    do_eval: bool = field(default=True, metadata={"help": "Whether to evaluate or not"})
    evaluation_strategy: str = field(default="epoch", metadata={"help": "When to evaluate, after certain number of steps or each epoch"})
    use_auth_token: str = field(default=False, metadata={"help": "auth token"})
    use_fast: bool = field(default=False, metadata={"help": "Whether to use fast tokenizer"})
    bits: Optional[int] = field(default=4, metadata={"help": "Number of bits to quantize the model to"})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help":"Lora dropout."})


hfparser = HfArgumentParser((TrainingConfig))
args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

print(args)

trainer = ChatTrainer(training_config=args)
trainer.train()
```
导入模型并合并
```python
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, PeftModel, get_peft_model

tokenizer = AutoTokenizer.from_pretrained("./chatglm3-6b-base", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm3-6b-base", trust_remote_code=True).half().cuda()

peft_model_id = './trained_model/checkpoint-35'
model = PeftModel.from_pretrained(model, peft_model_id)
```
或者 全参数微调不需要合并模型
```python
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, PeftModel, get_peft_model

tokenizer = AutoTokenizer.from_pretrained("./chatglm3-6b-base", trust_remote_code=True)
model = AutoModel.from_pretrained("./trained_model/checkpoint-14", trust_remote_code=True).half().cuda()
```
开始对话
```python
history = []
query = "你是谁"
role = "user"
inputs = tokenizer.build_chat_input(query, history=history, role=role)
inputs = inputs.to('cuda')
eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                        tokenizer.get_command("<|observation|>")]
gen_kwargs = {"max_length": 500, "num_beams": 1, "do_sample": True, "top_p": 0.8,
                      "temperature": 0.8}
outputs = model.generate(**inputs, **gen_kwargs, eos_token_id=eos_token_id)
outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
response = tokenizer.decode(outputs)
history = []
history.append({"role": "user", "content": "你是谁"})
response, history = model.process_response(response, history)
print(response)

query = "你能干嘛呀"
role = "user"
inputs = tokenizer.build_chat_input(query, history=history, role=role)
inputs = inputs.to('cuda')
outputs = model.generate(**inputs, **gen_kwargs, eos_token_id=eos_token_id)
outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
response = tokenizer.decode(outputs)
history.append({"role": role, "content": query})
response, history = model.process_response(response, history)
print(response)
```

### Reference(参考文档)

* [Github1页面](https://github.com/aceliuchanghong/chatglm3_6b_finetune)
* [Github2页面](https://github.com/aceliuchanghong/chatglm3-base-tuning)
* [讨论区](https://github.com/THUDM/ChatGLM3/discussions/253)

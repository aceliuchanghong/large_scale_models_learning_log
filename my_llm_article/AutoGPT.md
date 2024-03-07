### What
Auto-GPT是一个开源的AI工具，利用OpenAI的GPT-4或GPT-3.5 API来完成用户以自然语言表达的定义目标。
它主要通过将主任务分解为更小的组件来实现这一目标
### How
使用AutoGPT执行代码：
```text
@command装饰器用于定义自定义命令。它接受三个参数：
name：命令的名称。
description：命令的简短描述。
params：表示命令期望参数的字符串
```
```python
@command("execute_python_file", "Execute Python File", '"filename": ""')
def execute_python_file(filename: str) -> str:
    # Implementation of the command
```


### docker
```docker
docker pull significantgravitas/auto-gpt
```

### Reference(参考文档)

* [autogpt介绍](https://autogpt.cn/400)
* [autogpt安装](https://autogpt.cn/setup)



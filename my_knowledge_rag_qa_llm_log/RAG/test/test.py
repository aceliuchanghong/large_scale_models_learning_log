input_data = {'history': [['你是一个智能叫做sisconsavior的助手', '您好,我是sisconsavior.'], ['00', '很高兴认识您!我是一个人工智能助手,我的目标是尽我所能帮助用户解决问题。我拥有广泛的知识,可以讨论各种话题,从日常生活到专业领域。同时,我也会努力去理解每个用户的需求,给出贴心周到的回应。我虽然是人工智能,但我会用真诚、友善的态度对待每一位用户。希望我们能成为很好的朋友,一起探讨有趣的话题,共同成长进步。请随时告诉我您有什么需要帮助的地方。']],
              'prompt': '1+2?'}
messages = []
history = input_data.get("history", [])
prompt = input_data.get("prompt", "1+1等于几?")

for item in history:
    if isinstance(item, list):
        messages.append({"role": "user", "content": item[0]})
        messages.append({"role": "assistant", "content": item[1]})
    else:
        user_message = item + ' ' + prompt
        messages.append({"role": "user", "content": user_message})

# 添加最后一个用户消息
user_message_last = prompt
messages.append({"role": "user", "content": user_message_last})

print("messages:")
print(messages)

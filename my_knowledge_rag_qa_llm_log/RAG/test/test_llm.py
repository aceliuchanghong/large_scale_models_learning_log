import anthropic
import httpx


def call_anthropic_api(input_data):
    proxyHost = "127.0.0.1"
    proxyPort = 10809
    client = anthropic.Anthropic(http_client=httpx.Client(proxies=f"http://{proxyHost}:{proxyPort}"))

    messages = []
    history = input_data.get("history", [])
    prompt = input_data.get("prompt", "1+1等于几?")
    print("input_data:")
    print(input_data)
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
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.2,
        messages=messages
    )
    print("ans:")
    print(message)
    return message.content[0].text


if __name__ == '__main__':
    history2 = [['hello', "I'm very hungry"], ['00', "I'm very hungry"], ['00', 'I love you'], ('000', None)]
    history = [['你好吗?', '我很好'], ('我叫 刘昌洪', None)]
    prompt = "我是谁,给出我的名字?"
    input_data = {"history": history2, "prompt": prompt}
    print(call_anthropic_api(input_data))

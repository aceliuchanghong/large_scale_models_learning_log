import httpx

# 是否公开 gradio 链接
share = False
# 代理信息
proxies = {
    "http": "http://127.0.0.1:10809",
    "https": "http://127.0.0.1:10809"
}
# 测试爬取的 url
test_url = "https://python.langchain.com/docs/modules/data_connection/document_loaders/"

proxyHost = "127.0.0.1"
proxyPort = 10809
http_client = httpx.Client(proxies=f"http://{proxyHost}:{proxyPort}")

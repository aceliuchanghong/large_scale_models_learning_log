
### 页面操作
(数据集处理 认真准备数据集！以免后面出现各种报错，和炼出不理想的模型！)
1. 进入页面,点击‘Open UVR5-WebUI’ \
![img_3.png](..%2Fusing_files%2Fimgs%2Fimg_3.png)
才会出现 tools/uvr5/webui.py "cuda" True 9873 Tre
![img_2.png](..%2Fusing_files%2Fimgs%2Fimg_2.png)
先用HP2模型处理一遍（提取人声）
![img_4.png](..%2Fusing_files%2Fimgs%2Fimg_4.png)
![img_5.png](..%2Fusing_files%2Fimgs%2Fimg_5.png)
![img_6.png](..%2Fusing_files%2Fimgs%2Fimg_6.png)

然后将输出的干声文件再用onnx_dereverb最后用DeEcho-Aggressive（去混响），输出格式选wav。输出的文件默认在GPT-SoVITS-beta\GPT-SoVITS-beta\output\uvr5_opt这个文件夹下，处理完的音频（vocal）的是人声，(instrument)是伴奏，记得把instrument删掉。结束后记得到WebUI关闭UVR5节省显存。


2. 切割音频
![img_7.png](..%2Fusing_files%2Fimgs%2Fimg_7.png)
![img_8.png](..%2Fusing_files%2Fimgs%2Fimg_8.png)

3. 打标(打标就是给每个音频配上文字,标指的是标注)
![img_9.png](..%2Fusing_files%2Fimgs%2Fimg_9.png)
![img_10.png](..%2Fusing_files%2Fimgs%2Fimg_10.png)
4. 训练(按照图片步骤)
![img_11.png](..%2Fusing_files%2Fimgs%2Fimg_11.png)
4.1 微调训练 \
然后先点开启SoVITS训练，训练完后再点开启GPT训练，不可以一起训练
![img_12.png](..%2Fusing_files%2Fimgs%2Fimg_12.png)
![img_13.png](..%2Fusing_files%2Fimgs%2Fimg_13.png)
训练结束截图
![img_14.png](..%2Fusing_files%2Fimgs%2Fimg_14.png)
![img_15.png](..%2Fusing_files%2Fimgs%2Fimg_15.png)
5. 推理 \
5.1 开启推理界面 先点一下刷新模型，下拉选择模型推理，e代表轮数，s代表步数。不是轮数越高越好
![img_16.png](..%2Fusing_files%2Fimgs%2Fimg_16.png)
![img_17.png](..%2Fusing_files%2Fimgs%2Fimg_17.png)
5.2上传一段参考音频，建议是数据集中的音频
![img_18.png](..%2Fusing_files%2Fimgs%2Fimg_18.png)


6. 下载自己的模型本地

![img_19.png](..%2Fusing_files%2Fimgs%2Fimg_19.png)
![img_20.png](..%2Fusing_files%2Fimgs%2Fimg_20.png)
![img_21.png](..%2Fusing_files%2Fimgs%2Fimg_21.png)

或者模型转存到google drive
```
from google.colab import drive
drive.mount('/content/drive')

!cp -r /content/sample_data/README.md /content/drive/MyDrive/vits
```
### Reference(参考文档)
* [Github页面](https://github.com/RVC-Boss/GPT-SoVITS)
* [GPT-SoVITS中文指南](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e)
* [GPT-SoVITS操作页面](https://huggingface.co/spaces/aoxiang1221/gpt-vits)
* [\[GPT-SoVITS\]用 AI 快速复制声音，搭配 Colab](https://medium.com/dean-lin/gpt-sovits-%E7%94%A8-ai-%E5%BF%AB%E9%80%9F%E8%A4%87%E8%A3%BD%E4%BD%A0%E7%9A%84%E8%81%B2%E9%9F%B3-%E6%90%AD%E9%85%8D-colab-%E5%85%8D%E8%B2%BB%E5%85%A5%E9%96%80-f6a620cf7fc6)
* [自己的git库日志](https://github.com/aceliuchanghong/my_openvoice_log)

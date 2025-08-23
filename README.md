## 依赖
```
conda create --name sticker python=3.9
```

```
pip install huggingface_hub requests pillow torch torchvision
```


## main.py 使用示例
```
python main.py \       
  --image "http://images.cocodataset.org/val2017/000000039769.jpg" \
  --texts "一只猫的照片,一只狗的照片"
```


## prepare_model.py 运行流程
```
正在检测模型...
未检测到模型。

请选择要使用的模型：
1) chinese-clip-vit-base-patch16 (默认，平衡型，753MB)
2) chinese-clip-vit-large-patch14 (高性能，3GB)
3) chinese-clip-rn50 (快速，体积小)
请输入选择([1],2,3): 
选择模型: chinese-clip-vit-base-patch16

请选择模型下载方式：
1) 将模型下载到当前目录
2) chinese-clip-vit-large-patch14 (高性能，3GB)
请输入选择([1],2): 

请选择 HuggingFace 镜像站点：
1) 不设置镜像站，使用官方站点
2) 镜像站：https://hf-mirror.com/
3) 镜像站：https://hf-cdn.sufy.com/
0) 自定义站点
请输入选择(0,[1],2,3): 0
请输入自定义站点：hf-mirror.com
镜像站设置为：https://hf-mirror.com/

开始下载模型...
下载完成！
```


# 表情包查询（网页）

## 需求分析

Q: 此项目的主要功能是什么？  
A: 这个项目包含的程序将会启动一个网页，用户与网页交互以搜索需要的表情包。

Q: 此项目有哪些特色功能？  
A: 1. 对标签含有 `H` 的表情包模糊处理，适用于NSFW表情包。2. 随机抽一张/多张表情包。3. 支持多语言搜索（由模型与向量数据库支持）。4. 通过Emoji筛选符合条件的表情。

Q: 用户如何在网页上搜索表情包？  
A: 网页上显示搜索框和按钮，用户通过输入关键词搜索表情，可通过Emoji筛选需要的表情。

Q: 此项目用到了什么技术？  
A: 前端：HTML+CSS+JS；后端：python flask；数据库：；模型：

Q: 我应该如何添加自己的表情包？  
A: 在环境变量文件 `.env` 中添加这个表情包所在文件夹的路径，然后在网页上点击“扫描”按钮，程序将自动扫描并索引你的表情包。在扫描之前，你可以在文件夹中新建 `tags.txt` 作为标签文件，此文件的每一行作为一个标签可被程序读取。

Q: 我应该如何修改表情包的相关信息？  
A: 在网页上找到需要修改的表情包，选中并选择“编辑信息”或“重新索引”。

Q: 此项目如何实现图片搜索功能？  
A: 在搜索框使用自然语言搜索关键词即可，可根据每个图片的标签筛选。

Q: 此项目如何处理程序运行的异常情况？  
A: 在用户不刻意试探程序的边界情况（比如在搜索框输入很长的文本）时，暂未想到网页运行时可能会出现的异常，代码有错误的话网页也跑不起来。暂时忽略异常处理，遇到异常时再解决。

Q: 此项目的灵感来源和开发动机？  
A: 多年以前我使用过一个QQ斗图模块，在模块中搜索关键词即可筛选出高质量的表情包。虽然现在的QQ有表情包联想功能，但经常无法联想到合适的图片，图片乐趣程度也不及我的表情包收藏夹，为了解决“发出合适的高质量表情包”的需求，我打算开发此项目。

## 其他相关链接

- [MyGO表情包搜尋器](https://mygo.miyago9267.com/)
- [chn-lee-yumi/MaterialSearch](https://github.com/chn-lee-yumi/MaterialSearch)
- [自制表情包搜索引擎 演示 - BD4SUR](https://www.bilibili.com/video/BV1vJ4m1e7MN)
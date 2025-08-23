# 表情贴纸语义搜索

> 项目开发中  
> 10000 张表情索引大约需要 10 分钟，npz 文件大小 21 MB。  

## 部署流程

下载模型 -> 生成配置文件 -> 启动 app.py -> 打开网页

## Q&A

Q: 此项目的主要功能是什么？  
A: 这个项目包含的程序将会启动一个网页，用户与网页交互以搜索需要的表情包。

Q: 此项目用到了什么技术？  
A: 前端：HTML+CSS+JS；后端：python flask；数据存储：image_features.npz；模型：chinese-clip-vit-base-patch16

Q: 我应该如何添加自己的表情包？  
A: 将你的表情包放入 `stickers` 文件夹下，再启动索引获取新增的表情贴纸文件信息。

Q: 此项目如何实现图片搜索功能？  
A: 在搜索框使用中文搜索关键词即可。

Q: 此项目的灵感来源和开发动机？  
A: 多年以前我使用过一个QQ斗图模块，在模块中搜索关键词即可筛选出高质量的表情包。虽然现在的QQ有表情包联想功能，但经常无法联想到合适的图片，图片乐趣程度也不及我的表情包收藏夹，为了解决“发出合适的高质量表情包”的需求，我打算开发此项目。

## 其他相关链接

- [chn-lee-yumi/MaterialSearch](https://github.com/chn-lee-yumi/MaterialSearch)
- [自制表情包搜索引擎 演示 - BD4SUR](https://www.bilibili.com/video/BV1vJ4m1e7MN)
- [MyGO表情包搜尋器](https://mygo.miyago9267.com/)

# 中文语义图片搜索

这是一个基于 [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) 模型的本地图片语义搜索引擎，对于搜索表情包尤其好用。它允许你使用自然的中文描述来查找你电脑上的视觉素材，并提供了一个简单易用的Web界面进行交互。

![screenshot](assets/screenshot.jpg)


## 🌟 主要功能

*   **中文语义搜索**: 使用强大的AI模型，可通过自然语言描述来查找图片。
*   **Web用户界面**: 提供一个基于Flask的直观网页，用于搜索、预览和管理图片。
*   **高性能索引**:
    *   首次为你的图片库创建特征索引，实现快速搜索。
    *   支持增量更新，仅处理新增或修改过的文件。
*   **格式转换与复制**:
    *   内置高质量的 `webm` 到 `gif` 的转换功能。
    *   支持一键将文件（或其转换后的格式）复制到系统剪贴板，方便在微信、QQ等应用中使用。
*   **跨平台部署**: 提供了在 Windows 和 macOS 上的详细部署指南。


## 🚀 部署指南

> 此部分由 AI 编写，供参考 (o゜▽゜)o☆  

在开始之前，请确保您的系统满足以下基本要求：

*   Python 3.8 或更高版本
*   Git 命令行工具
*   至少 8GB 内存

---

### **通用预备步骤：安装 FFmpeg**

本项目使用 FFmpeg 进行视频/GIF文件的处理和格式转换，这是一个**必须**的外部依赖。

#### **macOS 用户 (推荐使用 Homebrew)**

打开“终端” (Terminal) 应用，运行以下命令：

```bash
brew install ffmpeg
```

Homebrew 会自动处理安装和路径配置，安装完成后即可使用。

#### **Windows 用户**

**方法一：使用 Winget 包管理器 (推荐)**

打开“命令提示符” (cmd) 或 “PowerShell”，运行以下命令：

```bash
winget install "FFmpeg Team.FFmpeg"
```

Winget 会自动下载并安装 FFmpeg，并通常会将其添加到系统路径中。

**方法二：手动安装**

1.  访问 [FFmpeg 官方下载页面](https://ffmpeg.org/download.html)。
2.  在 Windows 图标下，点击推荐的 `gyan.dev` 或 `BtbN` 链接。
3.  下载一个 `release-full.7z` 版本的压缩包。
4.  解压到一个你喜欢的位置，例如 `D:\ffmpeg`。
5.  **将 FFmpeg 添加到系统环境变量 `Path` 中**：
    *   在Windows搜索栏搜索 “编辑系统环境变量” 并打开它。
    *   在弹出的窗口中，点击 “环境变量...” 按钮。
    *   在 “系统变量” 区域找到名为 `Path` 的变量，选中它，然后点击 “编辑...”。
    *   点击 “新建”，然后输入你刚刚解压的 FFmpeg 文件夹中的 `bin` 目录的完整路径，例如：`D:\ffmpeg\bin`。
    *   一路点击“确定”保存所有设置。

**验证安装**：
无论使用哪种方法，新开一个终端或命令提示符窗口，输入 `ffmpeg -version`。如果能看到版本信息输出，说明 FFmpeg 已成功安装并配置。

---

### **项目部署步骤**

#### **1. 克隆项目仓库**

打开终端或命令提示符，进入你想要存放项目的目录，然后运行：

```bash
git clone https://github.com/oftx/StickerSearch.git
cd StickerSearch
```

#### **2. 创建并激活Python虚拟环境 (推荐)**

为保证项目依赖隔离，强烈建议使用虚拟环境。

##### **💻 macOS 用户 (尤其 Apple Silicon M 系列芯片)**

对于 M 系列芯片的 Mac，使用 `conda` 来管理环境可以更好地处理复杂的科学计算包（如 PyTorch），充分利用 Metal (MPS) 进行加速。

1.  **安装 Conda**: 如果你尚未安装，请从 [Miniconda官网](https://docs.conda.io/en/latest/miniconda.html) 下载并安装适用于 macOS 的版本。

2.  **创建并激活 Conda 环境**:
    ```bash
    # 创建一个名为 stickersearch 的新环境，并指定 Python 版本
    conda create -n stickersearch python=3.9 -y

    # 激活环境
    conda activate stickersearch
    ```
    激活成功后，你的命令行提示符前会有一个 `(stickersearch)` 标记。

##### **🪟 Windows 和 Intel Mac 用户**

使用 Python 内置的 `venv` 即可。

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# 在 Windows 上:
venv\Scripts\activate

# 在 macOS (Intel) 上:
source venv/bin/activate
```
激活成功后，你的命令行提示符前会有一个 `(venv)` 标记。

#### **3. 安装Python依赖**

请确保你已经激活了虚拟环境，然后根据你的操作系统安装依赖。

> 中国大陆依赖下载较慢可以在命令末尾添加镜像参数 `-i https://pypi.tuna.tsinghua.edu.cn/simple`

##### **💻 macOS 用户**
```bash
pip install -r requirements.txt
```

##### **🪟 Windows 用户**

Windows 需要一些额外的库以实现复制图像的功能：
```bash
pip install pywin32
```

部署完成！接下来请看使用说明。

## 📖 使用说明

#### **第一步：准备AI模型**

在开始使用前，你需要下载项目所需的中文CLIP模型。有下面两种方式：

##### 方式一：使用交互式脚本 (推荐)

项目提供了一个交互式脚本来简化这个过程。在项目根目录下，运行脚本：
```bash
python prepare_model_script.py
```

脚本会引导你完成以下选择：
1.  **选择模型**: 对于大多数用户，直接按回车选择**默认模型**即可。
2.  **选择下载位置**: 推荐直接按回车，将模型下载到当前项目文件夹下。
3.  **选择镜像站点**: 如果你访问Hugging Face官网速度慢，可以选择一个镜像站（如 `hf-mirror.com`）来加速下载。
4.  **设置代理**: 如果你需要通过代理访问网络，可以在此输入你的代理地址。

脚本会自动下载模型（约750MB），并在项目根目录创建一个 `config.json` 文件来记录模型路径。

##### 方式二：手动下载 (适用于高级用户或脚本下载失败时)

如果你无法使用上述脚本，或者更喜欢手动控制，请按照以下步骤操作：

1.  **安装 Git LFS**: 模型文件较大，需要使用 Git Large File Storage (LFS) 来克隆。如果尚未安装，请先安装它。
    *   访问 [Git LFS 官网](https://git-lfs.com/) 并根据你的操作系统进行安装。
    *   安装完成后，打开终端或命令提示符，运行一次性设置命令：
        ```bash
        git lfs install
        ```

2.  **克隆模型仓库**: 在本项目的 **根目录** 下，打开终端并运行以下命令来克隆默认的模型：
    ```bash
    git clone https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16
    ```
    这个命令会在你的项目文件夹内创建一个名为 `chinese-clip-vit-base-patch16` 的子文件夹，并下载所有模型文件。

3.  **创建配置文件**: 在项目的 **根目录** 下，手动创建一个名为 `config.json` 的文件。

4.  **编辑配置文件**: 在 `config.json` 文件中，填入以下内容，以告知程序模型的存放位置：
    ```json
    {
        "model_path": "chinese-clip-vit-base-patch16"
    }
    ```
    > **注意**: 请确保 `"model_path"` 的值与你刚刚克隆下来的文件夹名称完全一致。

完成以上步骤后，模型就已经准备就绪，可以进行下一步了。

#### **第二步：运行Web应用**

模型准备好后，启动Web服务器：
```bash
python app.py
```
当你在终端看到类似 `* Running on http://127.0.0.1:5001` 的输出时，说明服务已成功启动。

#### **第三步：创建/更新索引**

1.  打开你的浏览器，访问 **[http://127.0.0.1:5001](http://127.0.0.1:5001)**。
2.  点击页面右上角的 **设置图标 (⚙️)**。
3.  在弹出的“索引设置”窗口中，输入你的图片文件夹的路径，绝对路径和相对路径皆可。例如，`D:\stickers` 或 `stickers`。
> 把表情贴纸文件夹放入项目根目录下可使用相对路径
4.  点击 **“开始/更新索引”** 按钮。
5.  首次建立索引会花费一些时间，因为需要为每张图片提取特征。请耐心等待，直到下方显示“索引更新成功！”的摘要信息。

> 经过实测，索引 10000 张图片大约需要 10 分钟，npz 文件大小约 20 MB。  

索引完成后，你就可以在主页面的搜索框中输入中文描述来开始你的语义搜索之旅了！


## 其他相关链接

- [chn-lee-yumi/MaterialSearch](https://github.com/chn-lee-yumi/MaterialSearch)
- [自制表情包搜索引擎 演示 - BD4SUR](https://www.bilibili.com/video/BV1vJ4m1e7MN)
- [MyGO表情包搜尋器](https://mygo.miyago9267.com/)

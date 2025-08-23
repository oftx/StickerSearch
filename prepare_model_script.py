import os
from huggingface_hub import snapshot_download

# --- 配置 ---
MODELS = {
    "1": {
        "name": "OFA-Sys/chinese-clip-vit-base-patch16",
        "desc": "chinese-clip-vit-base-patch16 (默认，平衡型，753MB)"
    },
    "2": {
        "name": "OFA-Sys/chinese-clip-vit-large-patch14",
        "desc": "chinese-clip-vit-large-patch14 (高性能，3GB)"
    },
    "3": {
        "name": "OFA-Sys/chinese-clip-rn50",
        "desc": "chinese-clip-rn50 (快速，体积小)"
    }
}

MIRRORS = {
    "1": None, # 官方站点
    "2": "https://hf-mirror.com/",
    "3": "https://hf-cdn.sufy.com/"
}

# --- 辅助函数 ---

def select_model():
    """交互式地提示用户选择一个模型。"""
    print("\n请选择要使用的模型：")
    for key, value in MODELS.items():
        print(f"{key}) {value['desc']}")
    
    choice = input("请输入选择([1],2,3): ")
    if not choice.strip() or choice not in MODELS:
        choice = "1" # 默认选项
    
    selected_model = MODELS[choice]["name"]
    print(f"选择模型: {selected_model}")
    return selected_model

def select_download_path(model_name):
    """交互式地提示用户选择下载位置。"""
    model_folder_name = model_name.split('/')[-1]
    
    print("\n请选择模型下载方式：")
    print("1) 将模型下载到当前目录")
    print("2) 自定义下载目录")
    
    choice = input("请输入选择([1],2): ")
    if not choice.strip() or choice == "1":
        # 下载到当前目录下的一个子文件夹
        local_dir = os.path.join(os.getcwd(), model_folder_name)
    else:
        custom_path = input("请输入自定义目录路径: ")
        local_dir = os.path.join(custom_path, model_folder_name)
        
    print(f"模型将下载到: {local_dir}")
    return local_dir

def select_mirror():
    """交互式地提示用户选择 Hugging Face 镜像。"""
    print("\n请选择 HuggingFace 镜像站点：")
    print("1) 不设置镜像站，使用官方站点")
    print("2) 镜像站：https://hf-mirror.com/")
    print("3) 镜像站：https://hf-cdn.sufy.com/")
    print("0) 自定义站点")

    choice = input("请输入选择(0,[1],2,3): ")
    if not choice.strip() or choice not in ["0", "2", "3"]:
        choice = "1"

    if choice == "0":
        custom_site = input("请输入自定义站点 (例如: hf-mirror.com): ").strip()
        # 确保以 https:// 开头
        if not custom_site.startswith(('http://', 'https://')):
            custom_site = 'https://' + custom_site
        endpoint = custom_site
    else:
        endpoint = MIRRORS.get(choice)

    if endpoint:
        print(f"镜像站设置为：{endpoint}")
        # 设置环境变量，huggingface_hub 会自动使用
        os.environ['HF_ENDPOINT'] = endpoint
    else:
        print("将使用官方站点进行下载。")

def set_proxy():
    """交互式地提示用户设置代理。"""
    print() # 增加一个换行以获得更好的格式
    proxy_url = input("设置代理（留空不设置）：").strip()
    if proxy_url:
        # 为 http 和 https 同时设置环境变量
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        print(f"代理已设置为：{proxy_url}")
    else:
        # 确保没有从环境中继承代理设置
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        print("不使用代理。")

def check_model_existence(local_dir):
    """检查模型是否似乎已经下载。"""
    # 简单地检查一个通用文件是否存在，例如 config.json
    return os.path.exists(os.path.join(local_dir, 'config.json'))

# --- 主执行流程 ---

if __name__ == "__main__":
    print("正在检测模型...")
    
    # 此处假设模型不存在并总是询问。
    # 更复杂的检查可以记住上次选择的模型。
    print("未检测到模型。")

    # 1. 选择模型
    model_repo_id = select_model()
    
    # 2. 选择下载路径
    download_dir = select_download_path(model_repo_id)

    # 3. 如果模型已存在于该位置，则在重新下载前进行确认
    if check_model_existence(download_dir):
        print(f"\n警告：在目录 '{download_dir}' 中似乎已存在模型文件。")
        overwrite = input("是否要重新下载并覆盖？(y/N): ").lower()
        if overwrite != 'y':
            print("操作取消。")
            exit()

    # 4. 选择镜像
    select_mirror()
    
    # 5. 设置代理
    set_proxy()

    # 6. 下载模型
    print("\n开始下载模型...")
    try:
        snapshot_download(
            repo_id=model_repo_id,
            local_dir=download_dir
        )
        print("下载完成！")

        config_data = {"model_path": download_dir}
        config_filename = "config.json"
        with open(config_filename, "w", encoding="utf-8") as f:
            import json
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        print(f"模型路径配置已保存到 {config_filename} 文件中。")
    except Exception as e:
        print(f"\n下载过程中发生错误: {e}")
        print("请检查您的网络连接、镜像站点、代理设置或目录权限。")
# main_indexer.py
import argparse
import os
from image_search_core import ImageIndexer
from image_search_core.config import DEFAULT_INDEX_FILE, DEFAULT_IMAGE_DIR

def main():
    parser = argparse.ArgumentParser(description="创建或更新图片文件夹的特征索引。")
    parser.add_argument('image_dir', type=str, help="包含图片的文件夹路径。")
    parser.add_argument('--index_file', type=str, default=DEFAULT_INDEX_FILE, help=f"索引文件路径 (默认: {DEFAULT_INDEX_FILE})。")
    args = parser.parse_args()

    if args.image_dir == DEFAULT_IMAGE_DIR:
        os.makedirs(args.image_dir, exist_ok=True)

    try:
        indexer = ImageIndexer(image_dir=args.image_dir, index_file=args.index_file)
        indexer.update()
    except (ValueError, RuntimeError) as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
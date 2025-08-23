# main_searcher.py
import argparse
from image_search_core import ImageSearcher
from image_search_core.config import DEFAULT_INDEX_FILE

def main():
    parser = argparse.ArgumentParser(description="在图片索引中进行中文语义搜索。")
    parser.add_argument('query', type=str, help='用于搜索的中文描述。')
    parser.add_argument('--top_k', type=int, default=5, help="显示最匹配的前 K 个结果 (默认: 5)。")
    parser.add_argument('--index_file', type=str, default=DEFAULT_INDEX_FILE, help=f"索引文件路径 (默认: {DEFAULT_INDEX_FILE})。")
    args = parser.parse_args()

    try:
        searcher = ImageSearcher(index_file=args.index_file)
        results = searcher.search(query=args.query, top_k=args.top_k)
        
        if not results:
            print("未找到匹配结果。")
            return

        print(f"\n--- “{args.query}” 的 Top {len(results)} 搜索结果 ---")
        for i, res in enumerate(results):
            print(f"  {i+1}. 匹配度: {res['score']:.4f} - 图片: {res['path']}")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
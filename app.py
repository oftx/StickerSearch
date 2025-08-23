import os
import json
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
from image_search_core import ImageIndexer, ImageSearcher
from image_search_core.config import CONFIG_FILE, DEFAULT_IMAGE_DIR

app = Flask(__name__)

def get_persisted_image_dir():
    """从 config.json 读取并返回持久化的图片根目录。"""
    if not os.path.exists(CONFIG_FILE):
        return None
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        # 确保返回的是绝对路径，以供后续操作使用
        abs_path = config_data.get('image_base_dir')
        return os.path.abspath(abs_path) if abs_path else None
    except (IOError, json.JSONDecodeError):
        return None

# --- 网页端点 ---

@app.route('/')
def home():
    """渲染主页，并将持久化的目录路径传递给模板。"""
    persisted_dir = get_persisted_image_dir()
    # 如果配置文件中没有，就使用默认的
    dir_to_show = persisted_dir or DEFAULT_IMAGE_DIR
    return render_template('index.html', last_indexed_dir=dir_to_show)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """安全地提供图片文件服务。"""
    image_base_dir = get_persisted_image_dir()
    if image_base_dir and os.path.exists(os.path.join(image_base_dir, filename)):
        return send_from_directory(image_base_dir, filename)
    return "图片根目录未配置或图片不存在。", 404

# --- API 端点 ---

@app.route('/api/index', methods=['POST'])
def api_index_images():
    """API 端点：触发图片索引过程，并将目录路径持久化。"""
    data = request.get_json()
    if not data or 'image_dir' not in data:
        return jsonify({"status": "error", "message": "请求体中缺少 'image_dir' 参数。"}), 400

    image_dir = data['image_dir']
    if not os.path.isdir(image_dir):
        # 如果目录不存在，尝试创建它（特别是对于默认目录）
        try:
            os.makedirs(image_dir, exist_ok=True)
            print(f"目录 '{image_dir}' 不存在，已自动创建。")
        except OSError as e:
            return jsonify({"status": "error", "message": f"目录不存在且创建失败: {e}"}), 400
    
    try:
        indexer = ImageIndexer(image_dir=image_dir)
        summary = indexer.update()
        return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/search', methods=['GET'])
def api_search_images():
    """API 端点：执行语义搜索。"""
    query = request.args.get('query')
    top_k = request.args.get('top_k', 9, type=int)

    if not query:
        return jsonify({"status": "error", "message": "缺少 'query' URL 参数。"}), 400
    
    image_base_dir = get_persisted_image_dir()
    if not image_base_dir:
        return jsonify({"status": "error", "message": "图片根目录未在配置中设置。请先运行一次索引来配置路径。"}), 400

    try:
        searcher = ImageSearcher()
        results = searcher.search(query=query, top_k=top_k)

        # 遍历搜索结果，处理可能为相对路径的情况
        for item in results:
            path_from_index = item['path']
            
            # 步骤 1: 确保我们有一个绝对路径
            # 如果从索引读出的路径不是绝对路径，则根据当前工作目录转换它
            if not os.path.isabs(path_from_index):
                absolute_path = os.path.abspath(path_from_index)
            else:
                absolute_path = path_from_index

            # 步骤 2: 使用绝对路径来计算相对于图片库根目录的路径，用于生成URL
            # 这一步现在是安全的，因为 absolute_path 保证是绝对路径
            relative_path = os.path.relpath(absolute_path, image_base_dir)
            
            # 步骤 3: 生成最终的 URL
            item['url'] = url_for('serve_image', filename=relative_path)

        return jsonify({"status": "success", "results": results})
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "索引文件 'image_features.npz' 不存在。请先运行索引。"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
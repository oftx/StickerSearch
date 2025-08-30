# app.py

import os
import json
import glob
from datetime import datetime
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
from PIL import Image
import cv2

from image_search_core import ImageIndexer, ImageSearcher
from image_search_core.config import CONFIG_FILE, DEFAULT_IMAGE_DIR
from image_search_core.utils import copy_file_to_clipboard
from image_search_core.converter import convert_webm_to_gif, convert_webp

app = Flask(__name__)

# --- 定义转换后文件的存放目录 ---
CONVERTED_DIR = 'stickers_converted'


def get_persisted_image_dir():
    """从 config.json 读取并返回持久化的图片根目录。"""
    if not os.path.exists(CONFIG_FILE):
        return None
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        abs_path = config_data.get('image_base_dir')
        return os.path.abspath(abs_path) if abs_path else None
    except (IOError, json.JSONDecodeError):
        return None

# --- 网页端点 ---

@app.route('/')
def home():
    """渲染主页。"""
    persisted_dir = get_persisted_image_dir()
    dir_to_show = persisted_dir or DEFAULT_IMAGE_DIR
    return render_template('index.html', last_indexed_dir=dir_to_show)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """安全地提供原始图片文件服务。"""
    image_base_dir = get_persisted_image_dir()
    if image_base_dir and os.path.exists(os.path.join(image_base_dir, filename)):
        return send_from_directory(image_base_dir, filename)
    return "图片根目录未配置或图片不存在。", 404

# --- 已修复的路由：路径与目录名匹配 ---
@app.route('/stickers_converted/<path:filename>')
def serve_converted_image(filename):
    """安全地提供转换后的图片文件服务。"""
    converted_dir_abs = os.path.abspath(CONVERTED_DIR)
    if os.path.exists(os.path.join(converted_dir_abs, filename)):
        return send_from_directory(converted_dir_abs, filename)
    return "转换后的图片不存在。", 404
    

# --- API 端点 ---

@app.route('/api/index', methods=['POST'])
def api_index_images():
    """API 端点：触发图片索引过程。"""
    data = request.get_json()
    if not data or 'image_dir' not in data:
        return jsonify({"status": "error", "message": "请求体中缺少 'image_dir' 参数。"}), 400

    image_dir = data['image_dir']
    if not os.path.isdir(image_dir):
        try:
            os.makedirs(image_dir, exist_ok=True)
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
    """API 端点：执行语义搜索并返回包含元数据的详细结果。"""
    query = request.args.get('query')
    top_k = request.args.get('top_k', 20, type=int)

    if not query:
        return jsonify({"status": "error", "message": "缺少 'query' URL 参数。"}), 400
    
    image_base_dir = get_persisted_image_dir()
    if not image_base_dir:
        return jsonify({"status": "error", "message": "图片根目录未配置。请先运行一次索引。"}), 400

    try:
        searcher = ImageSearcher()
        results = searcher.search(query=query, top_k=top_k)
        
        detailed_results = []
        for item in results:
            path_from_index = item['path']
            absolute_path = os.path.abspath(path_from_index)
            if not os.path.exists(absolute_path):
                continue
            
            relative_path = os.path.relpath(absolute_path, image_base_dir).replace('\\', '/')
            item['path'] = relative_path
            item['url'] = url_for('serve_image', filename=relative_path)
            item['type'] = os.path.splitext(relative_path)[1].lower()

            try:
                item['filename'] = os.path.basename(absolute_path)
                item['size'] = os.path.getsize(absolute_path)
                item['date'] = datetime.fromtimestamp(os.path.getmtime(absolute_path)).isoformat() + 'Z'
                
                dimensions = [0, 0]
                try:
                    if absolute_path.lower().endswith(('.webm', '.mp4', '.gif')):
                        cap = cv2.VideoCapture(absolute_path)
                        if cap.isOpened():
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            dimensions = [width, height]
                        cap.release()
                    else:
                        with Image.open(absolute_path) as img:
                            width, height = img.size
                            dimensions = [width, height]
                except Exception as e_dim:
                    print(f"警告: 无法获取文件尺寸 {absolute_path}。错误: {e_dim}")

                item['dimensions'] = dimensions
                detailed_results.append(item)

            except OSError as e_meta:
                print(f"警告: 无法获取文件元数据 {absolute_path}。错误: {e_meta}")
                continue

        return jsonify({"status": "success", "results": detailed_results})
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "索引文件 'image_features.npz' 不存在。请先运行索引。"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/copy', methods=['POST'])
def api_copy_file():
    """
    API 端点：将文件对象复制到剪贴板。
    根据 'location' 参数从不同目录查找文件。
    """
    data = request.get_json()
    path = data.get('path')
    location = data.get('location', 'index') # 'index' or 'converted'
    preferred_format = data.get('preferred_format')

    if not path:
        return jsonify({"status": "error", "message": "请求体中缺少 'path' 参数。"}), 400

    path_to_copy = None

    if location == 'converted':
        base_dir = os.path.abspath(CONVERTED_DIR)
        path_to_copy = os.path.join(base_dir, path)
    else: # Default is 'index'
        base_dir = get_persisted_image_dir()
        if not base_dir:
            return jsonify({"status": "error", "message": "图片根目录未配置。"}), 500
        
        path_to_copy = os.path.join(base_dir, path)
        if preferred_format:
            base, _ = os.path.splitext(path_to_copy)
            preferred_path = base + preferred_format
            if os.path.exists(preferred_path):
                path_to_copy = preferred_path

    if not path_to_copy or not os.path.exists(path_to_copy):
        return jsonify({"status": "error", "message": f"文件未找到: {path}"}), 404

    success, message = copy_file_to_clipboard(path_to_copy)
    
    if success:
        message = f"已复制到剪贴板: {os.path.basename(path_to_copy)}"
        return jsonify({"status": "success", "message": message, "copied_file": os.path.basename(path_to_copy)})
    else:
        return jsonify({"status": "error", "message": message}), 500

@app.route('/api/find_relatives', methods=['GET'])
def api_find_relatives():
    """
    API 端点：查找同一目录下以及转换目录中所有同基本名的文件，并返回它们的详细信息。
    """
    relative_path = request.args.get('path')
    if not relative_path:
        return jsonify({"status": "error", "message": "缺少 'path' URL 参数。"}), 400

    image_base_dir = get_persisted_image_dir()
    if not image_base_dir:
        return jsonify({"status": "error", "message": "图片根目录未配置。"}), 500
        
    full_path = os.path.join(image_base_dir, relative_path)
    base_name = os.path.splitext(os.path.basename(full_path))[0]

    source_directory = os.path.dirname(full_path)
    source_search_pattern = os.path.join(source_directory, base_name + ".*")
    
    converted_dir_abs = os.path.abspath(CONVERTED_DIR)
    converted_search_pattern = os.path.join(converted_dir_abs, base_name + ".*")

    found_files_abs = set(glob.glob(source_search_pattern) + glob.glob(converted_search_pattern))

    detailed_files = []
    for f_abs in sorted(list(found_files_abs)):
        try:
            file_url = ''
            file_rel_path = ''
            
            # 判断文件来源并生成正确的 URL 和相对路径
            # url_for 会根据已修复的路由，自动生成正确的 /stickers_converted/... 路径
            if f_abs.startswith(converted_dir_abs):
                file_rel_path = os.path.relpath(f_abs, converted_dir_abs).replace('\\', '/')
                file_url = url_for('serve_converted_image', filename=file_rel_path)
            elif f_abs.startswith(image_base_dir):
                file_rel_path = os.path.relpath(f_abs, image_base_dir).replace('\\', '/')
                file_url = url_for('serve_image', filename=file_rel_path)
            else:
                continue

            file_info = {
                'path': file_rel_path,
                'url': file_url,
                'filename': os.path.basename(f_abs),
                'format': os.path.splitext(f_abs)[1].lower().replace('.', ''),
                'size': os.path.getsize(f_abs),
                'dimensions': [0, 0]
            }
            
            # 获取尺寸
            try:
                if f_abs.lower().endswith(('.webm', '.mp4', '.gif')):
                    cap = cv2.VideoCapture(f_abs)
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        file_info['dimensions'] = [width, height]
                    cap.release()
                else:
                    with Image.open(f_abs) as img:
                        width, height = img.size
                        file_info['dimensions'] = [width, height]
            except Exception as e_dim:
                print(f"警告: 无法获取文件尺寸 {f_abs}。错误: {e_dim}")
            
            detailed_files.append(file_info)
        except Exception as e_meta:
            print(f"警告: 无法获取文件元数据 {f_abs}。错误: {e_meta}")
            continue
            
    return jsonify({"status": "success", "files": detailed_files})

@app.route('/api/convert', methods=['POST'])
def api_convert_file():
    """
    API 端点：执行文件格式转换，并将结果保存到 'stickers_converted' 目录。
    """
    data = request.get_json()
    source_path_rel = data.get('source_path')
    target_format = data.get('target_format')
    params = data.get('params', {})
    
    if not all([source_path_rel, target_format]):
        return jsonify({"status": "error", "message": "缺少 source_path 或 target_format 参数。"}), 400

    output_dir_abs = os.path.abspath(CONVERTED_DIR)
    os.makedirs(output_dir_abs, exist_ok=True)
    
    image_base_dir = get_persisted_image_dir()
    if not image_base_dir:
        return jsonify({"status": "error", "message": "图片根目录未配置。"}), 500

    source_path_abs = os.path.join(image_base_dir, source_path_rel)
    if not os.path.exists(source_path_abs):
        return jsonify({"status": "error", "message": f"源文件不存在: {source_path_rel}"}), 404

    base_name = os.path.splitext(os.path.basename(source_path_rel))[0]
    new_filename = f"{base_name}.{target_format.lower()}"
    output_path_abs = os.path.join(output_dir_abs, new_filename)
    
    try:
        source_ext = os.path.splitext(source_path_abs)[1].lower()
        if source_ext == '.webm':
            convert_webm_to_gif(source_path_abs, output_path_abs, params)
        elif source_ext == '.webp':
            convert_webp(source_path_abs, output_path_abs, target_format, params)
        else:
            return jsonify({"status": "error", "message": f"不支持从 {source_ext} 格式转换。"}), 400
        
        # 返回新的文件名，供前端使用
        return jsonify({"status": "success", "message": "转换成功！", "output_path": new_filename})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
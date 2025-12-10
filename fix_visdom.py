import os
import requests
import time

# Visdom 的静态文件目录
BASE_DIR = "/opt/anaconda3/envs/laser_py310/lib/python3.10/site-packages/visdom/static"

# ==============================================================================
# 靠谱的下载源配置 (全部使用 jsDelivr CDN，避开 GitHub Raw)
# ==============================================================================
FILES_TO_FIX = {
    # 1. Visdom 核心源码 (通过 CDN 加速 GitHub)
    "css/style.css": "https://cdn.jsdelivr.net/gh/facebookresearch/visdom@master/py/visdom/static/css/style.css",
    "css/network.css": "https://cdn.jsdelivr.net/gh/facebookresearch/visdom@master/py/visdom/static/css/network.css",
    "js/main.js": "https://cdn.jsdelivr.net/gh/facebookresearch/visdom@master/py/visdom/static/js/main.js",

    # 2. D3 Selection Multi (修正为存在的版本)
    "js/d3-selection-multi.v1.js": "https://cdn.jsdelivr.net/npm/d3-selection-multi@1.0.1/build/d3-selection-multi.min.js",
    
    # 3. Plotly (如果之前下载成功了，脚本会检测并跳过，如果损坏会覆盖)
    "js/plotly-plotly.min.js": "https://cdn.jsdelivr.net/npm/plotly.js@2.11.1/dist/plotly.min.js",
}

def download_file(url, local_path):
    full_path = os.path.join(BASE_DIR, local_path)
    
    # 简单的重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[{attempt+1}/{max_retries}] Downloading {local_path} from CDN...")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            # 设置短一点的连接超时，长一点的读取超时
            r = requests.get(url, headers=headers, timeout=(5, 20))
            
            if r.status_code == 200:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "wb") as f:
                    f.write(r.content)
                print(f"✅ [SUCCESS] Saved to {local_path}")
                return # 下载成功，退出重试循环
            elif r.status_code == 404:
                 print(f"❌ [404 ERROR] File not found on CDN: {url}")
                 break # 404 不需要重试
            else:
                print(f"⚠️ [WARN] Status {r.status_code}. Retrying...")
                
        except Exception as e:
            print(f"⚠️ [WARN] Connection error: {e}. Retrying...")
            time.sleep(1) # 等一秒再试
    
    print(f"❌ [FAILED] Could not download {local_path} after {max_retries} attempts.")

def fix_filenames():
    js_dir = os.path.join(BASE_DIR, "js")
    # 必须保证这两对文件都存在， Visdom 代码里两边都可能引用
    pairs = [
        ("layout-bin-packer.js", "layout_bin_packer.js"),
        ("d3.min.js", "d3.v3.min.js")
    ]
    
    print("\nSynchronizing filenames (fixing underscore/dash issues)...")
    for f1, f2 in pairs:
        p1 = os.path.join(js_dir, f1)
        p2 = os.path.join(js_dir, f2)
        
        if os.path.exists(p1) and not os.path.exists(p2):
            os.system(f"cp {p1} {p2}")
            print(f"🔄 Copied {f1} -> {f2}")
        elif os.path.exists(p2) and not os.path.exists(p1):
            os.system(f"cp {p2} {p1}")
            print(f"🔄 Copied {f2} -> {f1}")
        elif os.path.exists(p1) and os.path.exists(p2):
            print(f"✅ {f1}/{f2} pair exists.")
        else:
            print(f"⚠️ Warning: Neither {f1} nor {f2} found. Layout might break.")

if __name__ == "__main__":
    print(f"Target Directory: {BASE_DIR}\n")
    
    # 先清理掉那几个 0kb 或者损坏的 html 错误文件
    for path in FILES_TO_FIX.keys():
        full_p = os.path.join(BASE_DIR, path)
        if os.path.exists(full_p):
            # 如果文件小于 1KB，很可能是之前下载的 404 错误页面，删掉重下
            if os.path.getsize(full_p) < 1000:
                print(f"🗑️ Deleting corrupted/small file: {path}")
                os.remove(full_p)

    for path, url in FILES_TO_FIX.items():
        download_file(url, path)
    
    fix_filenames()
    print("\n🎉 Repair Complete. Please restart Visdom.")
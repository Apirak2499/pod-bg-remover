#!/usr/bin/env python3
"""
POD Background Remover - Web App v2
====================================
Flask web application with AIPassiveLab theme.
Settings page with API guide + clean main UI.
Supports local mode + production deployment (Railway/Render/VPS).
"""

import os
import sys
import time
import json
import uuid
import shutil
import threading
import re
import hashlib
from pathlib import Path
from io import BytesIO
from datetime import datetime

from flask import (
    Flask, render_template_string, request, jsonify,
    send_from_directory, send_file
)

# CORS support for embedding in AIPassiveLab.com
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
from werkzeug.utils import secure_filename

try:
    from PIL import Image
except ImportError:
    os.system(f"{sys.executable} -m pip install pillow --break-system-packages -q")
    from PIL import Image

try:
    import requests as http_requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests --break-system-packages -q")
    import requests as http_requests

try:
    import numpy as np
except ImportError:
    os.system(f"{sys.executable} -m pip install numpy --break-system-packages -q")
    import numpy as np

# ========================
# AI Upscale (Real-ESRGAN + Pillow Enhance fallback)
# ========================
UPSCALER_AVAILABLE = False
UPSCALER_MODE = 'pillow'  # 'realesrgan' or 'pillow'
upscaler_model = None

try:
    from PIL import ImageFilter, ImageEnhance
except ImportError:
    pass


def load_upscaler():
    """Try to load Real-ESRGAN. Falls back to Pillow enhance."""
    global UPSCALER_AVAILABLE, UPSCALER_MODE, upscaler_model
    try:
        from RealESRGAN import RealESRGAN
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights('weights/RealESRGAN_x4.pth', download=True)
        upscaler_model = model
        UPSCALER_AVAILABLE = True
        UPSCALER_MODE = 'realesrgan'
        print(f"[UPSCALE] Real-ESRGAN x4 loaded on {device}")
        return True
    except Exception:
        # Fallback to Pillow enhance (always available)
        UPSCALER_AVAILABLE = True
        UPSCALER_MODE = 'pillow'
        print("[UPSCALE] Using Pillow Enhanced upscale (LANCZOS + Sharpen)")
        return True


def upscale_image_realesrgan(img):
    """Upscale using Real-ESRGAN (AI x4)"""
    global upscaler_model
    orig_w, orig_h = img.size
    print(f"  [UPSCALE-AI] Input: {orig_w}x{orig_h}")

    has_alpha = img.mode == 'RGBA'
    if has_alpha:
        r, g, b, a = img.split()
        rgb_img = Image.merge('RGB', (r, g, b))
        alpha_img = a
    else:
        rgb_img = img.convert('RGB')
        alpha_img = None

    sr_img = upscaler_model.predict(rgb_img)

    if has_alpha:
        new_size = sr_img.size
        alpha_up = alpha_img.resize(new_size, Image.LANCZOS)
        r2, g2, b2 = sr_img.split()
        result = Image.merge('RGBA', (r2, g2, b2, alpha_up))
    else:
        result = sr_img

    print(f"  [UPSCALE-AI] Output: {result.size[0]}x{result.size[1]}")
    return result


def upscale_image_pillow(img, scale=4):
    """Upscale using Pillow LANCZOS + Sharpen + Enhance"""
    orig_w, orig_h = img.size
    new_w, new_h = orig_w * scale, orig_h * scale
    print(f"  [UPSCALE-PILLOW] Input: {orig_w}x{orig_h} -> {new_w}x{new_h} (x{scale})")

    has_alpha = img.mode == 'RGBA'
    if has_alpha:
        r, g, b, a = img.split()
        rgb_img = Image.merge('RGB', (r, g, b))
        alpha_img = a
    else:
        rgb_img = img.convert('RGB')
        alpha_img = None

    # Upscale RGB with LANCZOS
    rgb_up = rgb_img.resize((new_w, new_h), Image.LANCZOS)

    # Sharpen to improve clarity
    rgb_up = rgb_up.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Sharpness(rgb_up)
    rgb_up = enhancer.enhance(1.5)
    # Slightly boost contrast for POD designs
    contrast = ImageEnhance.Contrast(rgb_up)
    rgb_up = contrast.enhance(1.05)

    if has_alpha:
        alpha_up = alpha_img.resize((new_w, new_h), Image.LANCZOS)
        r2, g2, b2 = rgb_up.split()
        result = Image.merge('RGBA', (r2, g2, b2, alpha_up))
    else:
        result = rgb_up

    print(f"  [UPSCALE-PILLOW] Output: {result.size[0]}x{result.size[1]} (x{scale} Enhanced)")
    return result


def upscale_image(img, scale=4):
    """Upscale image using best available method"""
    global UPSCALER_MODE, upscaler_model
    if UPSCALER_MODE == 'realesrgan' and upscaler_model is not None and scale == 4:
        try:
            return upscale_image_realesrgan(img)
        except Exception as e:
            print(f"  [UPSCALE] Real-ESRGAN failed ({e}), falling back to Pillow")
    return upscale_image_pillow(img, scale)


def smart_upscale(img, target_w=4500, target_h=5400, scale=4):
    """
    Only upscale if the image is too small for the target canvas.
    If image needs to be scaled up more than 1.3x, apply upscale first.
    """
    img_w, img_h = img.size
    scale_needed = min(target_w / img_w, target_h / img_h)

    if scale_needed > 1.3:
        print(f"  [SMART] Image needs {scale_needed:.1f}x scale -> applying x{scale} upscale ({UPSCALER_MODE})")
        return upscale_image(img, scale)
    else:
        print(f"  [SMART] Image only needs {scale_needed:.1f}x scale -> LANCZOS is enough")
        return img


# ========================
# App Config
# ========================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Production: use /tmp for uploads/outputs (ephemeral storage)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RENDER') or os.environ.get('PRODUCTION'):
    DATA_DIR = '/tmp/pod_app'
else:
    DATA_DIR = BASE_DIR

app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(DATA_DIR, 'outputs')
CONFIG_FILE = os.path.join(DATA_DIR, 'config.json')

# Enable CORS for AIPassiveLab.com embedding
if CORS:
    CORS(app, origins=['https://aipassivelab.com', 'http://localhost:*'])

CANVAS_W = 4500
CANVAS_H = 5400
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

jobs = {}
jobs_lock = threading.Lock()

# Auto-cleanup: remove jobs older than this (seconds)
JOB_MAX_AGE = 3600  # 1 hour


def cleanup_old_jobs():
    """Remove jobs and their files older than JOB_MAX_AGE"""
    now = time.time()
    to_remove = []
    with jobs_lock:
        for jid, job in jobs.items():
            start = job.get('start_time', 0)
            if start > 0 and (now - start) > JOB_MAX_AGE and job.get('status') in ('completed', 'stopped', 'error'):
                to_remove.append(jid)
        for jid in to_remove:
            del jobs[jid]
    for jid in to_remove:
        for folder_key in ('UPLOAD_FOLDER', 'OUTPUT_FOLDER'):
            d = os.path.join(app.config[folder_key], jid)
            if os.path.isdir(d):
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
        zip_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{jid}.zip')
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass
    if to_remove:
        print(f"[CLEANUP] Removed {len(to_remove)} old job(s)")


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[CONFIG] Error reading config.json: {e}, using defaults")
            return {}
    return {}


def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        print(f"[CONFIG] Error saving config.json: {e}")


def safe_filename(filename):
    """
    Secure filename that preserves Thai/Unicode characters.
    Falls back to hash if filename becomes empty.
    """
    name, ext = os.path.splitext(filename)
    name = name.replace('/', '').replace('\\', '').replace('\0', '')
    name = re.sub(r'[<>:"|?*]', '', name)
    name = name.strip('. ')
    if not name:
        name = hashlib.md5(filename.encode('utf-8')).hexdigest()[:12]
    ext = ext.lower()
    if ext and not re.match(r'^\\.[a-zA-Z0-9]+$', ext):
        ext = ''
    return name + ext

# ========================
# Core Image Processing
# ========================
def remove_bg_photoroom(image_path, api_key):
    with open(image_path, 'rb') as f:
        response = http_requests.post(
            'https://sdk.photoroom.com/v1/segment',
            files={'image_file': f},
            headers={'x-api-key': api_key},
            timeout=120,
        )
    if response.status_code == 200:
        return Image.open(BytesIO(response.content)).convert('RGBA')
    elif response.status_code == 402:
        raise Exception('API credit exhausted!')
    elif response.status_code in (401, 403):
        raise Exception('Invalid API key!')
    else:
        raise Exception(f'API error {response.status_code}')


def remove_bg_removebg(image_path, api_key):
    with open(image_path, 'rb') as f:
        response = http_requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': f},
            data={'size': 'full'},
            headers={'X-Api-Key': api_key},
            timeout=120,
        )
    if response.status_code == 200:
        return Image.open(BytesIO(response.content)).convert('RGBA')
    elif response.status_code == 402:
        raise Exception('API credit exhausted!')
    else:
        raise Exception(f'API error {response.status_code}')


def trim_transparent(img):
    img = img.convert('RGBA')
    arr = np.array(img)
    alpha = arr[:, :, 3]
    high_threshold = 200
    rows = np.any(alpha > high_threshold, axis=1)
    cols = np.any(alpha > high_threshold, axis=0)
    if not rows.any() or not cols.any():
        rows = np.any(alpha > 128, axis=1)
        cols = np.any(alpha > 128, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img.crop((int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1))


def place_on_canvas(img, canvas_w=4500, canvas_h=5400):
    img = trim_transparent(img)
    img_w, img_h = img.size
    padding_pct = 0.005
    usable_w = int(canvas_w * (1 - 2 * padding_pct))
    usable_h = int(canvas_h * (1 - 2 * padding_pct))
    scale = min(usable_w / img_w, usable_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    pad_y = int(canvas_h * padding_pct)
    x = (canvas_w - new_w) // 2
    y = pad_y
    canvas.paste(img_resized, (x, y), img_resized)
    return canvas


def process_single_image(image_path, output_path, api_key, engine, use_upscale=False, upscale_scale=4):
    if 'PhotoRoom' in engine:
        img_no_bg = remove_bg_photoroom(image_path, api_key)
    elif 'remove.bg' in engine:
        img_no_bg = remove_bg_removebg(image_path, api_key)
    else:
        raise Exception('Unknown engine')

    # AI Upscale (if enabled)
    if use_upscale:
        img_no_bg = smart_upscale(img_no_bg, CANVAS_W, CANVAS_H, upscale_scale)

    final = place_on_canvas(img_no_bg, CANVAS_W, CANVAS_H)
    final.save(str(output_path), 'PNG', optimize=True, dpi=(300, 300))
    return final.size


def process_job(job_id, file_paths, api_key, engine, use_upscale=False, upscale_scale=4):
    # Cleanup old jobs before starting new work
    cleanup_old_jobs()

    with jobs_lock:
        job = jobs[job_id]
        job['status'] = 'processing'
    job['start_time'] = time.time()
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(output_dir, exist_ok=True)
    total = len(file_paths)
    success = 0
    errors = 0
    for i, fpath in enumerate(file_paths):
        if job.get('stop'):
            job['status'] = 'stopped'
            break
        fname = os.path.basename(fpath)
        job['current_file'] = fname
        job['current_index'] = i + 1
        try:
            out_name = Path(fname).stem + '.png'
            out_path = os.path.join(output_dir, out_name)
            process_single_image(fpath, out_path, api_key, engine, use_upscale, upscale_scale)
            success += 1
            job['last_output'] = out_name
        except Exception as e:
            errors += 1
            job['last_error'] = f'{fname}: {str(e)[:80]}'
            if '402' in str(e) or 'credit' in str(e).lower():
                job['status'] = 'error'
                job['error_msg'] = 'API credits exhausted!'
                break
        job['success'] = success
        job['errors'] = errors
        job['progress'] = ((i + 1) / total) * 100
        job['elapsed'] = time.time() - job['start_time']
    if job['status'] == 'processing':
        job['status'] = 'completed'
    job['elapsed'] = time.time() - job['start_time']
    job['success'] = success
    job['errors'] = errors
    job['progress'] = 100


# ========================
# Shared CSS
# ========================
SHARED_CSS = r'''
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #111118;
        --bg-card: #161620;
        --bg-card-hover: #1c1c2a;
        --border: #1e1e30;
        --cyan: #00d2eb;
        --cyan-dim: rgba(0, 210, 235, 0.15);
        --indigo: #6366f1;
        --indigo-dim: rgba(99, 102, 241, 0.12);
        --yellow: #fbbf24;
        --green: #22c55e;
        --red: #ef4444;
        --text: #ffffff;
        --text-secondary: #94a3b8;
        --text-dim: #64748b;
        --font: 'Noto Sans Thai', 'Inter', system-ui, sans-serif;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: var(--font);
        background: var(--bg-primary);
        color: var(--text);
        min-height: 100vh;
    }
    a { color: var(--cyan); text-decoration: none; }
    a:hover { text-decoration: underline; }

    .navbar {
        position: sticky; top: 0; z-index: 100;
        background: rgba(10, 10, 15, 0.9);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid var(--border);
        padding: 0 2rem; height: 64px;
        display: flex; align-items: center; justify-content: space-between;
    }
    .nav-brand {
        display: flex; align-items: center; gap: 12px;
        text-decoration: none; color: var(--text);
        font-weight: 700; font-size: 1.15rem;
    }
    .nav-brand:hover { text-decoration: none; }
    .nav-logo {
        width: 36px; height: 36px; border-radius: 10px;
        background: linear-gradient(135deg, var(--indigo), var(--cyan));
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 16px; color: white;
    }
    .nav-brand .accent { color: var(--cyan); }
    .nav-actions { display: flex; gap: 8px; align-items: center; }
    .nav-link {
        color: var(--text-secondary); text-decoration: none;
        padding: 8px 14px; border-radius: 8px; font-size: 14px;
        transition: all 0.2s; display: flex; align-items: center; gap: 6px;
    }
    .nav-link:hover { color: var(--text); background: rgba(255,255,255,0.06); text-decoration: none; }
    .nav-link.active { color: var(--cyan); }
    .nav-btn {
        padding: 8px 20px; border-radius: 8px; font-size: 14px;
        font-weight: 600; cursor: pointer; border: none; transition: all 0.2s;
        text-decoration: none; display: inline-flex; align-items: center; gap: 6px;
    }
    .nav-btn:hover { text-decoration: none; }
    .nav-btn-outline { background: transparent; color: var(--text); border: 1px solid var(--border); }
    .nav-btn-primary { background: var(--cyan); color: #000; }

    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px; padding: 24px;
        transition: border-color 0.3s;
    }
    .card:hover { border-color: rgba(99,102,241,0.3); }
    .card-title {
        font-size: 12px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.2px;
        color: var(--text-dim); margin-bottom: 16px;
        display: flex; align-items: center; gap: 8px;
    }

    .btn {
        display: inline-flex; align-items: center; justify-content: center; gap: 8px;
        padding: 12px 24px; border-radius: 12px;
        font-family: var(--font); font-size: 15px; font-weight: 600;
        cursor: pointer; border: none; transition: all 0.2s;
    }
    .btn-primary {
        background: linear-gradient(135deg, var(--cyan), #00b4d8);
        color: #000; font-weight: 700; width: 100%;
    }
    .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 8px 25px rgba(0,210,235,0.3); }
    .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }
    .btn-danger { background: var(--red); color: white; }
    .btn-secondary { background: var(--bg-secondary); color: var(--text); border: 1px solid var(--border); }
    .btn-sm { padding: 8px 16px; font-size: 13px; border-radius: 8px; }

    .form-label { display: block; font-size: 13px; font-weight: 500; color: var(--text-secondary); margin-bottom: 6px; }
    .form-input, .form-select {
        width: 100%; padding: 10px 14px;
        background: var(--bg-secondary); border: 1px solid var(--border);
        border-radius: 10px; color: var(--text);
        font-family: var(--font); font-size: 14px; outline: none;
        transition: border-color 0.2s;
    }
    .form-input:focus, .form-select:focus { border-color: var(--cyan); box-shadow: 0 0 0 3px rgba(0,210,235,0.1); }
    .form-input::placeholder { color: var(--text-dim); }

    .toast {
        position: fixed; bottom: 20px; right: 20px;
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 12px; padding: 16px 20px; font-size: 14px;
        z-index: 1000; box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        transform: translateY(100px); opacity: 0; transition: all 0.3s;
    }
    .toast.show { transform: translateY(0); opacity: 1; }
    .toast.success { border-color: rgba(34,197,94,0.3); }
    .toast.error { border-color: rgba(239,68,68,0.3); }

    .footer {
        text-align: center; padding: 40px 2rem;
        border-top: 1px solid var(--border);
        color: var(--text-dim); font-size: 13px;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.5;} }
    .processing { animation: pulse 1.5s infinite; color: var(--cyan); }
'''


# ========================
# Main Page HTML
# ========================
MAIN_HTML = r'''
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>POD BG Remover — AIPassiveLab</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+Thai:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        ''' + SHARED_CSS + r'''

        .hero {
            text-align: center; padding: 50px 2rem 36px;
            position: relative;
        }
        .hero::before {
            content: ''; position: absolute; top: 0; left: 50%;
            transform: translateX(-50%);
            width: 600px; height: 350px;
            background: radial-gradient(ellipse, rgba(0,210,235,0.06) 0%, transparent 70%);
            pointer-events: none;
        }
        .hero-badge {
            display: inline-flex; align-items: center; gap: 8px;
            background: var(--cyan-dim); border: 1px solid rgba(0,210,235,0.2);
            border-radius: 50px; padding: 6px 18px; font-size: 13px;
            color: var(--cyan); margin-bottom: 18px;
        }
        .hero-badge::before { content: ''; width: 6px; height: 6px; background: var(--cyan); border-radius: 50%; }
        .hero h1 { font-size: 2.6rem; font-weight: 800; line-height: 1.2; margin-bottom: 14px; }
        .hero h1 .gradient {
            background: linear-gradient(135deg, var(--cyan), var(--indigo));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero p { color: var(--text-secondary); font-size: 1rem; max-width: 520px; margin: 0 auto; }

        .app-container { max-width: 1100px; margin: 0 auto; padding: 0 2rem 60px; }
        .app-grid { display: grid; grid-template-columns: 1fr 320px; gap: 24px; }
        @media (max-width: 860px) { .app-grid { grid-template-columns: 1fr; } }

        /* No-API warning */
        .api-warning {
            background: rgba(251,191,36,0.08);
            border: 1px solid rgba(251,191,36,0.25);
            border-radius: 12px; padding: 16px 20px;
            display: flex; align-items: center; gap: 12px;
            margin-bottom: 20px; font-size: 14px;
        }
        .api-warning .icon { font-size: 20px; flex-shrink: 0; }
        .api-warning a { color: var(--yellow); font-weight: 600; }

        /* Upload */
        .upload-zone {
            border: 2px dashed var(--border); border-radius: 16px;
            padding: 50px 20px; text-align: center;
            transition: all 0.3s; cursor: pointer; position: relative; overflow: hidden;
        }
        .upload-zone:hover, .upload-zone.dragover { border-color: var(--cyan); background: rgba(0,210,235,0.03); }
        .upload-zone .upload-icon { font-size: 52px; margin-bottom: 14px; opacity: 0.6; }
        .upload-zone h3 { font-size: 17px; font-weight: 600; margin-bottom: 8px; }
        .upload-zone p { font-size: 13px; color: var(--text-dim); }
        .upload-zone input[type="file"] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
        .file-badge {
            display: inline-flex; align-items: center; gap: 6px;
            background: var(--cyan-dim); border: 1px solid rgba(0,210,235,0.2);
            border-radius: 50px; padding: 5px 16px; font-size: 13px;
            color: var(--cyan); margin-top: 14px;
        }

        /* Stats */
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .stat-item { background: var(--bg-secondary); border-radius: 12px; padding: 16px; text-align: center; }
        .stat-value {
            font-size: 22px; font-weight: 800;
            background: linear-gradient(135deg, var(--cyan), var(--indigo));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .stat-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }

        /* Progress */
        .progress-bar-bg { width: 100%; height: 8px; background: var(--bg-secondary); border-radius: 10px; overflow: hidden; }
        .progress-bar-fill { height: 100%; border-radius: 10px; background: linear-gradient(90deg, var(--cyan), var(--indigo)); transition: width 0.3s; width: 0%; }
        .progress-info { display: flex; justify-content: space-between; margin-top: 8px; font-size: 13px; color: var(--text-secondary); }
        .progress-status { font-size: 13px; color: var(--text-secondary); margin-top: 10px; min-height: 18px; }

        /* Preview */
        .preview-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 6px; margin-top: 10px; }
        .preview-thumb {
            aspect-ratio: 5/6; border-radius: 8px; background: var(--bg-secondary);
            overflow: hidden; position: relative; border: 1px solid var(--border);
        }
        .preview-thumb img { width: 100%; height: 100%; object-fit: cover; }
        .preview-thumb .check {
            position: absolute; top: 3px; right: 3px; width: 18px; height: 18px;
            border-radius: 50%; background: var(--green); color: #000;
            display: flex; align-items: center; justify-content: center;
            font-size: 10px; font-weight: 700;
        }

        /* Download */
        .download-list { max-height: 260px; overflow-y: auto; margin-top: 10px; }
        .download-item {
            display: flex; align-items: center; justify-content: space-between;
            padding: 8px 10px; background: var(--bg-secondary);
            border-radius: 8px; margin-bottom: 4px; font-size: 12px;
        }
        .download-item .fname { color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 170px; }
        .download-item a { color: var(--cyan); font-weight: 600; font-size: 11px; white-space: nowrap; }

        /* Result banner */
        .result-banner {
            background: linear-gradient(135deg, rgba(0,210,235,0.08), rgba(99,102,241,0.08));
            border: 1px solid rgba(0,210,235,0.2);
            border-radius: 12px; padding: 20px; text-align: center; margin-top: 20px;
        }
        .result-banner h3 { color: var(--cyan); font-size: 17px; margin-bottom: 6px; }
        .result-banner p { color: var(--text-secondary); font-size: 14px; }

        /* Gear icon */
        .gear-icon {
            width: 38px; height: 38px; border-radius: 10px;
            background: rgba(255,255,255,0.04); border: 1px solid var(--border);
            display: flex; align-items: center; justify-content: center;
            cursor: pointer; transition: all 0.2s; font-size: 18px;
        }
        .gear-icon:hover { background: rgba(255,255,255,0.08); border-color: var(--cyan); }

        /* Engine pill on main page */
        .engine-pill {
            display: inline-flex; align-items: center; gap: 8px;
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 50px; padding: 8px 18px; font-size: 13px;
            color: var(--text-secondary); margin-bottom: 20px;
        }
        .engine-pill .dot { width: 8px; height: 8px; border-radius: 50%; }
        .engine-pill .dot.connected { background: var(--green); }
        .engine-pill .dot.disconnected { background: var(--red); }

        /* Upscale toggle */
        .upscale-row {
            display: flex; align-items: center; justify-content: space-between;
            background: var(--bg-secondary); border: 1px solid var(--border);
            border-radius: 12px; padding: 14px 18px; margin-bottom: 20px;
        }
        .upscale-info { display: flex; align-items: center; gap: 10px; }
        .upscale-info .icon { font-size: 20px; }
        .upscale-info .label { font-size: 14px; font-weight: 600; }
        .upscale-info .sublabel { font-size: 11px; color: var(--text-dim); }
        .upscale-status {
            font-size: 11px; padding: 3px 10px; border-radius: 6px;
            display: inline-flex; align-items: center; gap: 4px;
        }
        .upscale-status.ready { background: rgba(34,197,94,0.12); color: var(--green); border: 1px solid rgba(34,197,94,0.2); }
        .upscale-status.notready { background: rgba(239,68,68,0.08); color: var(--text-dim); border: 1px solid var(--border); }
        .upscale-status.gpu { background: rgba(99,102,241,0.12); color: var(--indigo); border: 1px solid rgba(99,102,241,0.2); }

        /* Toggle switch */
        .toggle-switch { position: relative; width: 44px; height: 24px; flex-shrink: 0; }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider {
            position: absolute; inset: 0; cursor: pointer;
            background: var(--border); border-radius: 24px; transition: 0.3s;
        }
        .toggle-slider::before {
            content: ''; position: absolute; width: 18px; height: 18px;
            left: 3px; bottom: 3px; background: white; border-radius: 50%; transition: 0.3s;
        }
        .toggle-switch input:checked + .toggle-slider { background: var(--cyan); }
        .toggle-switch input:checked + .toggle-slider::before { transform: translateX(20px); }

        /* Scale selector */
        .scale-selector {
            display: none; align-items: center; gap: 6px;
            margin-top: 10px; padding-top: 12px;
            border-top: 1px solid var(--border);
        }
        .scale-selector.show { display: flex; }
        .scale-selector .scale-label { font-size: 12px; color: var(--text-dim); margin-right: 4px; }
        .scale-btn {
            padding: 6px 16px; border-radius: 8px; font-size: 13px; font-weight: 700;
            cursor: pointer; border: 1px solid var(--border); transition: all 0.2s;
            background: var(--bg-primary); color: var(--text-secondary);
            font-family: var(--font);
        }
        .scale-btn:hover { border-color: var(--cyan); color: var(--text); }
        .scale-btn.active {
            background: var(--cyan); color: #000; border-color: var(--cyan);
            box-shadow: 0 2px 10px rgba(0,210,235,0.3);
        }
    </style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar">
    <a href="/" class="nav-brand">
        <div class="nav-logo">A</div>
        <span class="accent">AI</span>PassiveLab
    </a>
    <div class="nav-actions">
        <a href="https://aipassivelab.com" target="_blank" class="nav-link">หน้าแรก</a>
        <a href="/" class="nav-link active">โปรแกรม</a>
        <a href="/settings" class="gear-icon" title="Settings">⚙️</a>
    </div>
</nav>

<!-- Hero -->
<section class="hero">
    <div class="hero-badge">POD Background Remover</div>
    <h1>ลบพื้นหลังภาพ <span class="gradient">POD</span></h1>
    <p>ลบพื้นหลัง วางเทมเฟรม 4500×5400 px 300 DPI อัตโนมัติ รองรับ batch หลายพันภาพ</p>
</section>

<!-- Main App -->
<div class="app-container">

    <!-- API Warning (shown if no key) -->
    <div class="api-warning" id="apiWarning" style="{{ 'display:none;' if has_api_key else '' }}">
        <span class="icon">⚠️</span>
        <div>ยังไม่ได้ตั้งค่า API Key — <a href="/settings">ไปหน้า Settings</a> เพื่อใส่ API Key ก่อนเริ่มใช้งาน</div>
    </div>

    <!-- Engine indicator -->
    <div style="text-align:center;">
        <div class="engine-pill">
            <span class="dot {{ 'connected' if has_api_key else 'disconnected' }}"></span>
            <span>{{ engine_name }}</span>
            <span style="color:var(--text-dim);">•</span>
            <a href="/settings" style="font-size:12px;">เปลี่ยน</a>
        </div>
    </div>

    <div class="app-grid">
        <!-- Left Column -->
        <div>
            <!-- Upload Zone -->
            <div class="card" style="margin-bottom:20px;">
                <div class="card-title">📁 อัปโหลดภาพ</div>
                <div class="upload-zone" id="uploadZone">
                    <div class="upload-icon">📷</div>
                    <h3>ลากไฟล์มาวางที่นี่ หรือคลิกเลือก</h3>
                    <p>รองรับ PNG, JPG, WEBP, BMP, TIFF — สูงสุด 2,000 ภาพต่อครั้ง</p>
                    <input type="file" id="fileInput" multiple accept=".png,.jpg,.jpeg,.webp,.bmp,.tiff,.tif">
                    <div class="file-badge" id="fileBadge" style="display:none;">
                        <span id="fileCountText">0 ไฟล์</span>
                    </div>
                </div>
            </div>

            <!-- AI Upscale Toggle -->
            <div class="upscale-row" id="upscaleRow" style="flex-wrap:wrap;">
                <div style="display:flex; align-items:center; justify-content:space-between; width:100%;">
                    <div class="upscale-info">
                        <span class="icon">🔬</span>
                        <div>
                            <div class="label">AI Upscale <span class="upscale-status notready" id="upscaleStatus">...</span></div>
                            <div class="sublabel">เพิ่มความคมชัดของภาพ (ฟรี, ใช้เวลาเพิ่มต่อภาพ)</div>
                        </div>
                    </div>
                    <label class="toggle-switch">
                        <input type="checkbox" id="upscaleToggle" onchange="toggleUpscaleOptions()">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                <div class="scale-selector" id="scaleSelector" style="width:100%;">
                    <span class="scale-label">ระดับ:</span>
                    <button class="scale-btn" onclick="setScale(2)" id="scaleBtn2">x2</button>
                    <button class="scale-btn" onclick="setScale(3)" id="scaleBtn3">x3</button>
                    <button class="scale-btn active" onclick="setScale(4)" id="scaleBtn4">x4</button>
                </div>
            </div>

            <!-- Buttons -->
            <div style="display:flex; gap:12px; margin-bottom:20px;">
                <button class="btn btn-primary" id="startBtn" onclick="startProcessing()">
                    ⚡ เริ่มประมวลผล
                </button>
                <button class="btn btn-danger btn-sm" id="stopBtn" onclick="stopProcessing()" style="width:110px; display:none;">
                    ⏹ หยุด
                </button>
            </div>

            <!-- Progress -->
            <div class="card" id="progressCard" style="display:none; margin-bottom:20px;">
                <div class="card-title">⏳ Progress</div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" id="progressBar"></div></div>
                <div class="progress-info">
                    <span id="progressText">0%</span>
                    <span id="progressCount">0 / 0</span>
                </div>
                <div class="progress-status" id="progressStatus"></div>
            </div>

            <!-- Result -->
            <div class="result-banner" id="resultBanner" style="display:none;">
                <h3 id="resultTitle"></h3>
                <p id="resultText"></p>
            </div>
        </div>

        <!-- Right Column -->
        <div>
            <!-- Stats -->
            <div class="card" style="margin-bottom:16px;">
                <div class="card-title">📊 สถิติ</div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="statTotal">0</div>
                        <div class="stat-label">Total</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statSuccess">0</div>
                        <div class="stat-label">Success</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statErrors">0</div>
                        <div class="stat-label">Errors</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statTime">0s</div>
                        <div class="stat-label">Time</div>
                    </div>
                </div>
                <div style="margin-top:10px; text-align:center; font-size:13px; color:var(--text-dim);">
                    Cost: <span id="statCost" style="color:var(--yellow);">$0.00</span>
                </div>
            </div>

            <!-- Downloads -->
            <div class="card" id="downloadCard" style="display:none; margin-bottom:16px;">
                <div class="card-title">📥 ดาวน์โหลด</div>
                <button class="btn btn-secondary btn-sm" onclick="downloadAll()" style="width:100%; margin-bottom:10px;">
                    📦 ดาวน์โหลดทั้งหมด (ZIP)
                </button>
                <div class="download-list" id="downloadList"></div>
            </div>

            <!-- Preview -->
            <div class="card">
                <div class="card-title">👁️ Preview</div>
                <div class="preview-grid" id="previewGrid">
                    <div style="grid-column:1/-1; text-align:center; padding:24px; color:var(--text-dim); font-size:13px;">ยังไม่มีภาพ</div>
                </div>
            </div>
        </div>
    </div>
</div>

<footer class="footer">
    POD Background Remover — Powered by <a href="https://aipassivelab.com" target="_blank">AIPassiveLab</a>
</footer>

<div class="toast" id="toast"></div>

<script>
let selectedFiles = [];
let currentJobId = null;
let pollInterval = null;

const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');

fileInput.addEventListener('change', e => {
    selectedFiles = Array.from(e.target.files);
    updateFileCount();
});
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault(); uploadZone.classList.remove('dragover');
    const exts = ['.png','.jpg','.jpeg','.webp','.bmp','.tiff','.tif'];
    selectedFiles = Array.from(e.dataTransfer.files).filter(f => exts.includes('.'+f.name.split('.').pop().toLowerCase()));
    updateFileCount();
});

function updateFileCount() {
    const badge = document.getElementById('fileBadge');
    if (selectedFiles.length > 0) {
        badge.style.display = 'inline-flex';
        document.getElementById('fileCountText').textContent = selectedFiles.length + ' ไฟล์พร้อมประมวลผล';
        document.getElementById('statTotal').textContent = selectedFiles.length;
    } else { badge.style.display = 'none'; }
}

let upscaleScale = 4;

function toggleUpscaleOptions() {
    const on = document.getElementById('upscaleToggle').checked;
    const sel = document.getElementById('scaleSelector');
    if (on) { sel.classList.add('show'); } else { sel.classList.remove('show'); }
}

function setScale(n) {
    upscaleScale = n;
    [2,3,4].forEach(s => {
        const btn = document.getElementById('scaleBtn'+s);
        btn.classList.toggle('active', s === n);
    });
}

// Check upscaler status on page load
async function checkUpscaler() {
    try {
        const resp = await fetch('/api/check-upscaler');
        const data = await resp.json();
        const badge = document.getElementById('upscaleStatus');
        if (data.mode === 'realesrgan') {
            if (data.gpu) {
                badge.className = 'upscale-status gpu';
                badge.textContent = '⚡ AI + GPU';
            } else {
                badge.className = 'upscale-status ready';
                badge.textContent = '✓ AI + CPU';
            }
        } else {
            badge.className = 'upscale-status ready';
            badge.textContent = '✓ Enhanced';
        }
    } catch(e) {
        const badge = document.getElementById('upscaleStatus');
        badge.className = 'upscale-status ready';
        badge.textContent = '✓ Enhanced';
    }
}
checkUpscaler();

async function startProcessing() {
    if (!selectedFiles.length) { showToast('กรุณาเลือกไฟล์ภาพก่อน','error'); return; }
    const formData = new FormData();
    for (const f of selectedFiles) formData.append('files', f);
    const useUpscale = document.getElementById('upscaleToggle').checked;
    formData.append('upscale', useUpscale ? 'true' : 'false');
    formData.append('upscale_scale', upscaleScale.toString());
    document.getElementById('startBtn').disabled = true;
    document.getElementById('startBtn').innerHTML = '<span class="processing">⏳ กำลังอัปโหลด...</span>';
    document.getElementById('stopBtn').style.display = 'block';
    document.getElementById('progressCard').style.display = 'block';
    document.getElementById('resultBanner').style.display = 'none';
    document.getElementById('previewGrid').innerHTML = '';
    try {
        const resp = await fetch('/api/process', { method:'POST', body: formData });
        const data = await resp.json();
        if (data.error) { showToast(data.error,'error'); resetUI(); return; }
        currentJobId = data.job_id;
        document.getElementById('startBtn').innerHTML = '<span class="processing">⚡ กำลังประมวลผล...</span>';
        pollInterval = setInterval(pollProgress, 1000);
    } catch(err) { showToast('Upload failed: '+err.message,'error'); resetUI(); }
}

async function pollProgress() {
    if (!currentJobId) return;
    try {
        const resp = await fetch('/api/status/'+currentJobId);
        const job = await resp.json();
        document.getElementById('progressBar').style.width = job.progress+'%';
        document.getElementById('progressText').textContent = Math.round(job.progress)+'%';
        document.getElementById('progressCount').textContent = (job.current_index||0)+' / '+job.total;
        if (job.current_file) document.getElementById('progressStatus').innerHTML = '<span class="processing">🔄 '+job.current_file+'</span>';
        document.getElementById('statSuccess').textContent = job.success||0;
        document.getElementById('statErrors').textContent = job.errors||0;
        if (job.elapsed) {
            const m=Math.floor(job.elapsed/60), s=Math.floor(job.elapsed%60);
            document.getElementById('statTime').textContent = m>0 ? m+'m'+s+'s' : s+'s';
        }
        const costPer = job.engine==='PhotoRoom' ? 0.02 : 0.13;
        document.getElementById('statCost').textContent = '$'+((job.success||0)*costPer).toFixed(2);
        if (job.last_output) updatePreview(job.job_id, job.last_output, job.success);
        if (['completed','stopped','error'].includes(job.status)) { clearInterval(pollInterval); onJobComplete(job); }
    } catch(e) { console.error(e); }
}

function updatePreview(jobId, filename, count) {
    if (document.getElementById('thumb-'+count)) return;
    const grid = document.getElementById('previewGrid');
    const div = document.createElement('div');
    div.className='preview-thumb'; div.id='thumb-'+count;
    div.innerHTML = '<img src="/api/preview/'+jobId+'/'+filename+'"><div class="check">✓</div>';
    grid.appendChild(div);
}

function onJobComplete(job) {
    document.getElementById('progressBar').style.width = '100%';
    document.getElementById('progressText').textContent = '100%';
    document.getElementById('progressStatus').innerHTML = '';
    const banner = document.getElementById('resultBanner');
    banner.style.display = 'block';
    if (job.status==='completed') {
        document.getElementById('resultTitle').textContent = '🎉 เสร็จสมบูรณ์!';
        document.getElementById('resultText').textContent = 'สำเร็จ '+job.success+' ภาพ, ผิดพลาด '+job.errors+' ภาพ';
    } else if (job.status==='stopped') {
        document.getElementById('resultTitle').textContent = '⏹ หยุดการทำงาน';
        document.getElementById('resultText').textContent = 'สำเร็จ '+job.success+' ภาพ';
    } else {
        document.getElementById('resultTitle').textContent = '❌ เกิดข้อผิดพลาด';
        document.getElementById('resultText').textContent = job.error_msg||'Unknown error';
    }
    if (job.success>0) { document.getElementById('downloadCard').style.display='block'; loadDownloadList(job.job_id); }
    resetUI();
    showToast('ประมวลผลเสร็จ: '+job.success+' สำเร็จ','success');
}

async function loadDownloadList(jobId) {
    const resp = await fetch('/api/files/'+jobId);
    const data = await resp.json();
    const list = document.getElementById('downloadList');
    list.innerHTML = '';
    data.files.forEach(f => {
        const div = document.createElement('div'); div.className='download-item';
        div.innerHTML = '<span class="fname">'+f+'</span><a href="/api/download/'+jobId+'/'+f+'" download>⬇ Download</a>';
        list.appendChild(div);
    });
}

function downloadAll() { if (currentJobId) window.location.href='/api/download-zip/'+currentJobId; }

async function stopProcessing() {
    if (currentJobId) await fetch('/api/stop/'+currentJobId, {method:'POST'});
    showToast('กำลังหยุด...','error');
}

function resetUI() {
    document.getElementById('startBtn').disabled = false;
    document.getElementById('startBtn').innerHTML = '⚡ เริ่มประมวลผล';
    document.getElementById('stopBtn').style.display = 'none';
}

function showToast(msg, type='info') {
    const t = document.getElementById('toast');
    t.textContent = msg; t.className = 'toast show '+type;
    setTimeout(()=>{ t.className='toast'; }, 3000);
}
</script>
</body>
</html>
'''


# ========================
# Settings Page HTML
# ========================
SETTINGS_HTML = r'''
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Settings — POD BG Remover</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+Thai:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        ''' + SHARED_CSS + r'''

        .settings-container { max-width: 720px; margin: 0 auto; padding: 40px 2rem 60px; }

        .page-header {
            margin-bottom: 32px;
        }
        .page-header .back-link {
            display: inline-flex; align-items: center; gap: 6px;
            color: var(--text-secondary); font-size: 14px;
            margin-bottom: 16px;
        }
        .page-header .back-link:hover { color: var(--cyan); text-decoration: none; }
        .page-header h1 { font-size: 1.8rem; font-weight: 800; }
        .page-header p { color: var(--text-secondary); font-size: 14px; margin-top: 6px; }

        .section { margin-bottom: 28px; }
        .section-title {
            font-size: 15px; font-weight: 700; margin-bottom: 14px;
            display: flex; align-items: center; gap: 8px;
        }

        .form-group { margin-bottom: 16px; }

        .api-guide {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px; padding: 20px;
            margin-top: 16px;
        }
        .api-guide h4 {
            font-size: 14px; font-weight: 700; margin-bottom: 12px;
            display: flex; align-items: center; gap: 8px;
        }
        .api-guide ol {
            padding-left: 20px; font-size: 13px; color: var(--text-secondary);
            line-height: 1.8;
        }
        .api-guide ol a { color: var(--cyan); font-weight: 600; }
        .api-guide .price-tag {
            display: inline-flex; align-items: center; gap: 4px;
            background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.2);
            border-radius: 6px; padding: 2px 10px; font-size: 12px;
            color: var(--yellow); margin-left: 8px;
        }

        .save-status {
            display: inline-flex; align-items: center; gap: 6px;
            font-size: 13px; color: var(--green); margin-left: 12px;
            opacity: 0; transition: opacity 0.3s;
        }
        .save-status.show { opacity: 1; }

        .input-row { display: flex; gap: 10px; }
        .input-row .form-input { flex: 1; }
        .input-row .btn { flex-shrink: 0; width: 90px; }

        /* Tab switcher */
        .tab-row {
            display: flex; gap: 4px; margin-bottom: 20px;
            background: var(--bg-secondary); border-radius: 12px;
            padding: 4px; border: 1px solid var(--border);
        }
        .tab-btn {
            flex: 1; padding: 10px 16px; border-radius: 10px;
            font-family: var(--font); font-size: 14px; font-weight: 600;
            cursor: pointer; border: none;
            background: transparent; color: var(--text-dim);
            transition: all 0.2s;
        }
        .tab-btn.active { background: var(--bg-card); color: var(--text); box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
        .tab-btn:hover:not(.active) { color: var(--text-secondary); }

        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar">
    <a href="/" class="nav-brand">
        <div class="nav-logo">A</div>
        <span class="accent">AI</span>PassiveLab
    </a>
    <div class="nav-actions">
        <a href="/" class="nav-link">← กลับหน้าหลัก</a>
    </div>
</nav>

<div class="settings-container">
    <div class="page-header">
        <a href="/" class="back-link">← กลับหน้าหลัก</a>
        <h1>⚙️ Settings</h1>
        <p>ตั้งค่า API Key สำหรับลบพื้นหลังภาพ</p>
    </div>

    <!-- Engine Selector -->
    <div class="section">
        <div class="section-title">🔧 เลือก Engine</div>
        <div class="tab-row">
            <button class="tab-btn {{ 'active' if engine == 'PhotoRoom' else '' }}" onclick="switchEngine('PhotoRoom')">
                PhotoRoom
            </button>
            <button class="tab-btn {{ 'active' if engine == 'remove.bg' else '' }}" onclick="switchEngine('remove.bg')">
                remove.bg
            </button>
        </div>
    </div>

    <!-- PhotoRoom Tab -->
    <div class="tab-content {{ 'active' if engine == 'PhotoRoom' else '' }}" id="tab-PhotoRoom">
        <div class="card">
            <div class="section-title">
                🔑 PhotoRoom API Key
                <span class="price-tag">$0.02 / ภาพ</span>
                <span class="save-status" id="saveStatus-PhotoRoom">✓ บันทึกแล้ว</span>
            </div>
            <div class="form-group">
                <label class="form-label">API Key</label>
                <div class="input-row">
                    <input type="password" id="apiKey-PhotoRoom" class="form-input"
                           placeholder="sk_pr_xxx..." value="{{ photoroom_key }}">
                    <button class="btn btn-secondary btn-sm" onclick="toggleKey('PhotoRoom')" id="toggleBtn-PhotoRoom">Show</button>
                </div>
            </div>
            <button class="btn btn-primary" onclick="saveKey('PhotoRoom')" style="margin-top:8px;">
                💾 บันทึก API Key
            </button>

            <!-- Guide -->
            <div class="api-guide">
                <h4>📖 วิธีสมัครและดึง API Key</h4>
                <ol>
                    <li>ไปที่ <a href="https://www.photoroom.com/api" target="_blank">photoroom.com/api</a></li>
                    <li>คลิก <strong>"Get API Key"</strong> หรือ <strong>"Start for free"</strong></li>
                    <li>สมัครสมาชิกด้วย Email หรือ Google Account</li>
                    <li>หลังลงเข้า Dashboard คลิกที่ <strong>"API Keys"</strong> ในเมนูซ้ายบน</li>
                    <li>คลิก <strong>"Create new key"</strong> จะได้ Key ขึ้นต้นด้วย <code>sk_pr_...</code></li>
                    <li>คัดลอก Key มาวางในช่องด้านบน แล้วกด <strong>"บันทึก"</strong></li>
                </ol>
                <div style="margin-top:14px; padding-top:14px; border-top:1px solid var(--border); font-size:12px; color:var(--text-dim);">
                    💡 <strong>Basic Plan (ฟรี)</strong> — ได้ 10 credits ฟรี ใช้ v1 API ลบพื้นหลังไม่ได้เลย<br>
                    💰 <strong>ราคา</strong> — ประมาณ $0.02 ต่อภาพ (ซื้อ credits)
                </div>
            </div>
        </div>
    </div>

    <!-- remove.bg Tab -->
    <div class="tab-content {{ 'active' if engine == 'remove.bg' else '' }}" id="tab-remove-bg">
        <div class="card">
            <div class="section-title">
                🔑 remove.bg API Key
                <span class="price-tag">$0.13 / ภาพ</span>
                <span class="save-status" id="saveStatus-remove-bg">✓ บันทึกแล้ว</span>
            </div>
            <div class="form-group">
                <label class="form-label">API Key</label>
                <div class="input-row">
                    <input type="password" id="apiKey-remove-bg" class="form-input"
                           placeholder="xxxxxxxx..." value="{{ removebg_key }}">
                    <button class="btn btn-secondary btn-sm" onclick="toggleKey('remove-bg')" id="toggleBtn-remove-bg">Show</button>
                </div>
            </div>
            <button class="btn btn-primary" onclick="saveKey('remove-bg')" style="margin-top:8px;">
                💾 บันทึก API Key
            </button>

            <div class="api-guide">
                <h4>📖 วิธีสมัครและดึง API Key</h4>
                <ol>
                    <li>ไปที่ <a href="https://www.remove.bg/api" target="_blank">remove.bg/api</a></li>
                    <li>คลิก <strong>"Get API Key"</strong></li>
                    <li>สมัครสมาชิกด้วย Email</li>
                    <li>หลังลงเข้า ไปที่ <a href="https://www.remove.bg/dashboard#api-key" target="_blank">Dashboard → API Keys</a></li>
                    <li>คลิก <strong>"Show"</strong> เพื่อดู API Key</li>
                    <li>คัดลอก Key มาวางในช่องด้านบน แล้วกด <strong>"บันทึก"</strong></li>
                </ol>
                <div style="margin-top:14px; padding-top:14px; border-top:1px solid var(--border); font-size:12px; color:var(--text-dim);">
                    💡 <strong>Free Plan</strong> — ได้ 50 preview credits ฟรี (ข้นาดเล็ก)<br>
                    💰 <strong>ราคา Full HD</strong> — ประมาณ $0.13 ต่อภาพ (ต้องซื้อ credits)
                </div>
            </div>
        </div>
    </div>
</div>

<footer class="footer">
    POD Background Remover — Powered by <a href="https://aipassivelab.com" target="_blank">AIPassiveLab</a>
</footer>

<div class="toast" id="toast"></div>

<script>
function switchEngine(engine) {
    // Update tabs
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    if (engine === 'PhotoRoom') {
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
        document.getElementById('tab-PhotoRoom').classList.add('active');
    } else {
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
        document.getElementById('tab-remove-bg').classList.add('active');
    }

    // Save engine choice
    fetch('/api/save-settings', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({engine: engine})
    });
}

function toggleKey(type) {
    const inp = document.getElementById('apiKey-'+type);
    const btn = document.getElementById('toggleBtn-'+type);
    if (inp.type === 'password') { inp.type='text'; btn.textContent='Hide'; }
    else { inp.type='password'; btn.textContent='Show'; }
}

async function saveKey(type) {
    const key = document.getElementById('apiKey-'+type).value.trim();
    if (!key) { showToast('กรุณาใส่ API Key','error'); return; }

    const engine = type === 'PhotoRoom' ? 'PhotoRoom' : 'remove.bg';
    const resp = await fetch('/api/save-settings', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ engine: engine, api_key: key })
    });
    const data = await resp.json();
    if (data.ok) {
        const status = document.getElementById('saveStatus-'+type);
        status.classList.add('show');
        setTimeout(()=> status.classList.remove('show'), 3000);
        showToast('บันทึก API Key สำเร็จ!','success');
    }
}

function showToast(msg, type='info') {
    const t = document.getElementById('toast');
    t.textContent = msg; t.className = 'toast show '+type;
    setTimeout(()=>{ t.className='toast'; }, 3000);
}
</script>
</body>
</html>
'''


# ========================
# Routes
# ========================
@app.route('/')
def index():
    config = load_config()
    engine = config.get('engine', 'PhotoRoom')
    api_key = config.get('api_key', '')
    engine_name = 'PhotoRoom API ($0.02/img)' if 'PhotoRoom' in engine else 'remove.bg API ($0.13/img)'
    return render_template_string(
        MAIN_HTML,
        has_api_key=bool(api_key),
        engine_name=engine_name,
        engine=engine
    )


@app.route('/settings')
def settings():
    config = load_config()
    engine = config.get('engine', 'PhotoRoom')
    return render_template_string(
        SETTINGS_HTML,
        engine=engine,
        photoroom_key=config.get('api_key', '') if 'PhotoRoom' in engine else config.get('photoroom_key', ''),
        removebg_key=config.get('removebg_key', '') if 'remove.bg' in engine else config.get('removebg_key', ''),
    )


@app.route('/api/save-settings', methods=['POST'])
def api_save_settings():
    data = request.get_json()
    config = load_config()

    if 'engine' in data:
        config['engine'] = data['engine']

    if 'api_key' in data:
        engine = data.get('engine', config.get('engine', 'PhotoRoom'))
        config['api_key'] = data['api_key']
        if 'PhotoRoom' in engine:
            config['photoroom_key'] = data['api_key']
        else:
            config['removebg_key'] = data['api_key']

    save_config(config)
    return jsonify({'ok': True})


@app.route('/api/process', methods=['POST'])
def api_process():
    config = load_config()
    api_key = config.get('api_key', '').strip()
    engine = config.get('engine', 'PhotoRoom')

    if not api_key:
        return jsonify({'error': 'ยังไม่ได้ตั้งค่า API Key — กรุณาไปหน้า Settings ก่อน'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    job_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(uuid.uuid4())[:8]
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(upload_dir, exist_ok=True)

    file_paths = []
    for f in files:
        if f.filename:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext in SUPPORTED_FORMATS:
                fname = safe_filename(f.filename)
                fpath = os.path.join(upload_dir, fname)
                f.save(fpath)
                file_paths.append(fpath)

    if not file_paths:
        return jsonify({'error': 'No valid image files'}), 400

    with jobs_lock:
        jobs[job_id] = { 'job_id': job_id, 'status': 'queued',
        'total': len(file_paths), 'success': 0, 'errors': 0,
        'progress': 0, 'current_file': '', 'current_index': 0,
        'elapsed': 0, 'engine': engine,
        'last_output': None, 'last_error': None, 'stop': False,
    }

    # Check upscale flag from form data
    use_upscale = request.form.get('upscale', 'false').lower() == 'true'
    upscale_scale = int(request.form.get('upscale_scale', '4'))
    if upscale_scale not in (2, 3, 4):
        upscale_scale = 4

    thread = threading.Thread(
        target=process_job, args=(job_id, file_paths, api_key, engine, use_upscale, upscale_scale), daemon=True
    )
    thread.start()

    return jsonify({'job_id': job_id, 'total': len(file_paths), 'upscale': use_upscale, 'scale': upscale_scale})


@app.route('/api/check-upscaler')
def api_check_upscaler():
    """Check upscaler status — always available (Pillow fallback)"""
    mode = 'pillow'
    gpu = False
    device_name = 'Pillow Enhanced'
    try:
        import RealESRGAN
        import torch
        gpu = torch.cuda.is_available()
        mode = 'realesrgan'
        device_name = 'Real-ESRGAN (GPU)' if gpu else 'Real-ESRGAN (CPU)'
    except ImportError:
        pass
    return jsonify({'available': True, 'gpu': gpu, 'mode': mode, 'device': device_name})


@app.route('/api/status/<job_id>')
def api_status(job_id):
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        return jsonify(dict(jobs[job_id]))
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])


@app.route('/api/stop/<job_id>', methods=['POST'])
def api_stop(job_id):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]['stop'] = True
    return jsonify({'ok': True})


@app.route('/api/files/<job_id>')
def api_files(job_id):
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    if not os.path.isdir(output_dir):
        return jsonify({'files': []})
    files = sorted(f for f in os.listdir(output_dir) if f.endswith('.png'))
    return jsonify({'files': files})


@app.route('/api/download/<job_id>/<filename>')
def api_download(job_id, filename):
    return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], job_id), filename, as_attachment=True)


@app.route('/api/preview/<job_id>/<filename>')
def api_preview(job_id, filename):
    return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], job_id), filename)


@app.route('/api/download-zip/<job_id>')
def api_download_zip(job_id):
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    if not os.path.isdir(output_dir):
        return jsonify({'error': 'No outputs'}), 404
    zip_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{job_id}.zip')
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', output_dir)
    return send_file(zip_path, as_attachment=True, download_name=f'POD_Output_{job_id[:15]}.zip')


# ========================
# Initialize upscaler at module level (works with Gunicorn)
# ========================
load_upscaler()
print(f"[INIT] Upscale mode: {UPSCALER_MODE}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 50)
    print("  POD Background Remover - Web App v2")
    print("  AIPassiveLab Theme")
    print("=" * 50)
    print(f"\n  Upscale mode: {UPSCALER_MODE}")
    print(f"\n  Open: http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)

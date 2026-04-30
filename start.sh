#!/bin/bash

FACE_DETECTION_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$FACE_DETECTION_DIR"

echo "=========================================="
echo "  实时人脸识别系统 - 启动脚本"
echo "=========================================="

if [ ! -d "model" ]; then
    mkdir -p model
fi

if [ ! -f "model/frozen_inference_graph_face.pb" ]; then
    echo ""
    echo "❌ 错误：模型文件 model/frozen_inference_graph_face.pb 不存在！"
    echo ""
    echo "请下载模型文件："
    echo "  wget -O model/frozen_inference_graph_face.pb \\"
    echo "    'https://github.com/yeephycho/tensorflow-face-detection/raw/master/model/frozen_inference_graph_face.pb'"
    echo ""
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "📦 首次运行，创建 Python 虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📥 安装依赖..."
    pip install --upgrade pip
    pip install -r backend/requirements.txt gunicorn
else
    source venv/bin/activate
fi

mkdir -p face_db

echo ""
echo "🚀 启动人脸识别系统..."
echo "   本地访问: http://localhost:5000"
echo "   公网访问: http://你的公网IP:5000"
echo "   按 Ctrl+C 停止服务"
echo ""

cd backend
export FLASK_DEBUG=0
export PORT=5000

if command -v gunicorn &> /dev/null; then
    exec gunicorn --bind 0.0.0.0:5000 \
        --workers 1 \
        --timeout 120 \
        --threads 4 \
        app:app
else
    exec python app.py
fi

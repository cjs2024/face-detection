# 免费部署指南（Render + Vercel）

本系统采用**前后端分离**架构，无需信用卡，完全免费部署。

---

## 🏗️ 架构

```
用户浏览器
   │
   ├── 前端（HTML/CSS/JS）→ Vercel 免费托管
   │
   └── 后端（Flask API）  → Render 免费层
         │
         └── TensorFlow 人脸检测/比对/搜索
```

**注意**：Render 免费层每次部署后人脸库会清空。如需持久化可升级付费计划。

---

## 📦 第一步：上传代码到 GitHub

1. 访问 https://github.com 新建仓库
2. 仓库名：`tensorflow-face-detection`
3. 上传整个项目目录到仓库

**重要**：确保 `model/` 目录已包含模型文件 `frozen_inference_graph_face.pb`

---

## 🚀 第二步：部署后端到 Render

### 方法 A：Blueprint 部署（推荐）

1. 访问 https://dashboard.render.com/blueprints
2. 点击 **"New Blueprint Instance"**
3. Connect 你的 GitHub 仓库
4. 上传本目录下的 `render.yaml` 文件
5. 点击 **"Create Blueprint"**

### 方法 B：手动部署

1. 打开 https://render.com → 用 GitHub 账号登录
2. 点击 **"New +"** → **"Web Service"**
3. Connect 你的 GitHub 仓库
4. 设置：

| 配置项 | 值 |
|--------|-----|
| Name | `face-detection-api` |
| Region | Singapore |
| Branch | `main` |
| Root Directory | （留空） |
| Runtime | `Python 3.10` |
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `cd backend && python app.py` |

5. **Free Tier** 自动已选
6. 点击 **"Create Web Service"**

### 部署完成

Render 会提供 URL，例如：
```
https://face-detection-api.onrender.com
```

---

## 🌐 第三步：部署前端到 Vercel

### 1. 修改后端地址

部署前端之前，先修改 `frontend/public/js/api.js` 中的后端地址：

```javascript
// 第 14 行 - 改成你的 Render URL
_apiBaseUrl = 'https://你的render-url.onrender.com/api';
```

### 2. 部署到 Vercel

1. 访问 https://vercel.com → 用 GitHub 账号登录
2. 点击 **"Add New..."** → **"Project"**
3. Import 你的 GitHub 仓库
4. 设置：

| 配置项 | 值 |
|--------|-----|
| Framework Preset | `Other` |
| Root Directory | `./frontend` |
| Build Command | `node build.js` |
| Output Directory | `public` |

5. 点击 **"Deploy"**

---

## ✅ 访问网站

部署完成后：
```
https://你的项目名.vercel.app
```

---

## ⚠️ 重要限制

### Render 免费层
- **休眠**：15 分钟无请求服务会自动休眠
- **人脸库清空**：每次重新部署后人脸库会清空
- **冷启动慢**：首次请求可能需要 10-30 秒

### 消除休眠（可选）
1. Render 升级到 $7/月 Starter 计划
2. 或使用 Railway（每月 $5 可以保持活跃）
3. 或使用 Oracle Cloud Always Free（需信用卡）

---

## 🔧 本地开发调试

```bash
# 后端
cd backend
pip install -r requirements.txt
python app.py
# 访问 http://localhost:10000

# 前端（另开终端）
cd frontend
npx serve public -l 3000
# 访问 http://localhost:3000
```

---

## 📁 项目结构

```
tensorflow-face-detection/
├── backend/
│   ├── app.py              ← Flask 后端（端口 10000）
│   └── requirements.txt
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   ├── css/style.css
│   │   └── js/api.js, main.js
│   ├── vercel.json         ← Vercel 配置
│   └── build.js            ← Vercel 构建脚本
├── protos/
├── model/                   ← 人脸检测模型
├── Dockerfile
├── render.yaml              ← Render Blueprint 配置
└── DEPLOY.md
```
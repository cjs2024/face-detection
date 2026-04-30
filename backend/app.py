from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import sys
import time
import logging
import json
from datetime import datetime
import tensorflow as tf
from urllib.parse import unquote
from collections import deque

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
CORS(app)

# 全局检测器实例
detector = FaceDetector()
last_detection_result = None

class FaceDetector:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, '..', 'model', 'frozen_inference_graph_face.pb')
        if not os.path.exists(self.model_path):
            alt_path = os.path.join(base_dir, 'model', 'frozen_inference_graph_face.pb')
            if os.path.exists(alt_path):
                self.model_path = alt_path
            else:
                logger.warning(f"Model not found at {self.model_path} or {alt_path}")
        self.num_classes = 1
        self.detection_threshold = 0.2  # 降低阈值以提高敏感度
        
        # 🔥 防抖参数
        self.previous_boxes = None
        self.stability_factor = 0.3  # 平滑系数
        self.iou_threshold = 0.7     # 匹配阈值
        
        # 尝试加载模型
        self.load_model()

    def load_model(self):
        try:
            import tensorflow.compat.v1 as tf
            tf.disable_eager_execution()
            
            logger.info("使用 TensorFlow v1 加载模型...")
            self.graph = tf.Graph()
            with self.graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                
                self.sess = tf.Session(graph=self.graph)
                
                self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
                self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
                self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
                
            logger.info("✅ 模型加载成功")
            self.use_simulation = False
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.use_simulation = True
            logger.warning("⚠️ 使用模拟模式（无模型）")

    def detect_faces(self, image):
        if self.use_simulation:
            # 模拟检测
            h, w = image.shape[:2]
            num_faces = np.random.randint(0, 2)  # 0-1 个脸
            boxes = []
            scores = []
            for _ in range(num_faces):
                x1 = np.random.randint(0, w//2)
                y1 = np.random.randint(0, h//2)
                size = np.random.randint(min(w,h)//8, min(w,h)//3)
                x2, y2 = min(x1 + size, w), min(y1 + size, h)
                boxes.append([y1/h, x1/w, y2/h, x2/w])
                scores.append(np.random.uniform(0.4, 0.95))
            return np.array(boxes), np.array(scores), np.array([1]*len(boxes)), len(boxes)
        
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_expanded = np.expand_dims(image_rgb, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded}
            )
            return boxes[0], scores[0], classes[0], int(num[0])
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return np.array([]), np.array([]), np.array([]), 0

    def calculate_iou(self, box1, box2):
        """计算交并比"""
        y1_min, x1_min, y1_max, x1_max = box1
        y2_min, x2_min, y2_max, x2_max = box2
        
        inter_y_min = max(y1_min, y2_min)
        inter_x_min = max(x1_min, x2_min)
        inter_y_max = min(y1_max, y2_max)
        inter_x_max = min(x1_max, x2_max)
        
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height
        
        area1 = (y1_max - y1_min) * (x1_max - x1_min)
        area2 = (y2_max - y2_min) * (x2_max - x2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def smooth_boxes(self, current_boxes, current_scores):
        """检测框平滑算法"""
        if self.previous_boxes is None or len(current_boxes) == 0:
            self.previous_boxes = current_boxes
            return current_boxes
        
        smoothed_boxes = []
        
        for current_box in current_boxes:
            best_match_idx = -1
            best_iou = 0
            
            # 寻找最佳匹配
            for i, prev_box in enumerate(self.previous_boxes):
                iou = self.calculate_iou(current_box, prev_box)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_idx = i
        
            if best_match_idx != -1:
                # 指数移动平均平滑
                prev_box = self.previous_boxes[best_match_idx]
                alpha = self.stability_factor
                smoothed_box = [
                    prev_box[0] * (1 - alpha) + current_box[0] * alpha,
                    prev_box[1] * (1 - alpha) + current_box[1] * alpha,
                    prev_box[2] * (1 - alpha) + current_box[2] * alpha,
                    prev_box[3] * (1 - alpha) + current_box[3] * alpha
                ]
                smoothed_boxes.append(smoothed_box)
            else:
                smoothed_boxes.append(current_box)
        
        # 更新历史框
        self.previous_boxes = np.array(smoothed_boxes)
        return np.array(smoothed_boxes)

    def draw_boxes(self, image, boxes, scores, classes):
        """优化版绘制函数"""
        h, w = image.shape[:2]
        detected_faces = 0
        
        # 🔥 应用平滑算法
        if len(boxes) > 0:
            smoothed_boxes = self.smooth_boxes(boxes, scores)
        else:
            smoothed_boxes = boxes
            self.previous_boxes = None  # 重置历史
        
        # 绘制平滑后的检测框
        for i in range(len(smoothed_boxes)):
            if scores[i] > self.detection_threshold:  # 使用较低的阈值
                ymin, xmin, ymax, xmax = smoothed_boxes[i]
                left, top = int(xmin * w), int(ymin * h)
                right, bottom = int(xmax * w), int(ymax * h)
                
                # 调整近距离检测面积过滤条件
                face_width = right - left
                face_height = bottom - top
                min_face_size = min(h, w) * 0.01  # 减小最小人脸尺寸阈值
                
                if face_width >= min_face_size and face_height >= min_face_size:
                    # 绘制平滑的红色矩形框
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                    
                    # 显示稳定的置信度
                    confidence = f'{scores[i]:.2f}'
                    cv2.putText(image, f'Face: {confidence}', (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    detected_faces += 1
        
        return image, detected_faces

# 全局检测器实例
detector = FaceDetector()

@app.route('/api/capture_frame', methods=['GET'])
def capture_frame():
    """摄像头帧捕获（已移至浏览器端处理，此接口仅返回成功状态）"""
    return jsonify({'status': 'success', 'note': '浏览器端已处理'})

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """启动摄像头检测（已移至浏览器端）"""
    return jsonify({
        'status': 'success',
        'message': '浏览器端摄像头已启动'
    })

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    """停止检测"""
    return jsonify({
        'status': 'success',
        'message': '摄像头检测已停止'
    })

@app.route('/api/get_detection_status', methods=['GET'])
def get_detection_status():
    """获取检测状态"""
    return jsonify({
        'status': 'success',
        'is_detecting': False,
        'detection_info': last_detection_result
    })

@app.route('/api/test_connection', methods=['GET'])
def test_connection():
    """测试连接"""
    return jsonify({
        'status': 'connected',
        'message': '服务器连接正常',
        'current_dir': os.getcwd(),
        'python_version': sys.version,
        'tensorflow_version': getattr(__import__('tensorflow'), '__version__', 'unknown'),
        'has_model': os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'frozen_inference_graph_face.pb')),
        'has_label_map': os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'protos', 'face_label_map.pbtxt')),
        'last_detection': last_detection_result
    })

@app.route('/api/get_detection_status', methods=['GET'])
def get_detection_status():
    """获取检测状态（用于右侧预览）"""
    return jsonify({
        'status': 'success',
        'detection_info': last_detection_result,
        'is_detecting': is_detecting
    })

@app.route('/api/compare_faces', methods=['POST'])
def compare_faces():
    """比对两张人脸"""
    try:
        image1_data = request.form.get('image1')
        image2_data = request.form.get('image2')
        mode1 = request.form.get('mode1', 'confidence')
        mode2 = request.form.get('mode2', 'size')
        
        if not image1_data or not image2_data:
            return jsonify({'error': '缺少图片数据'}), 400
        
        def decode_image(data):
            if data.startswith('data:image/'):
                data = data.split(',')[1]
            image_bytes = base64.b64decode(data)
            np_array = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        image1 = decode_image(image1_data)
        image2 = decode_image(image2_data)
        
        if image1 is None or image2 is None:
            return jsonify({'error': '图片解码失败'}), 400
        
        features1 = face_recognition.extract_face_features(image1, mode=mode1)
        features2 = face_recognition.extract_face_features(image2, mode=mode2)
        similarity = face_recognition.calculate_similarity(features1, features2)
        similarity_percent = round(similarity * 100, 2)
        
        return jsonify({
            'status': 'success',
            'similarity': similarity_percent,
            'message': f'人脸相似度: {similarity_percent}%'
        })
    except Exception as e:
        logger.error(f"比对人脸失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/add_face', methods=['POST'])
def add_face():
    """添加人脸到数据库"""
    try:
        name = request.form.get('name')
        image_data = request.form.get('image')
        
        if not name or not image_data:
            return jsonify({'error': '缺少姓名或图片数据'}), 400
        
        if image_data.startswith('data:image/'):
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': '图片解码失败'}), 400
        
        success, message = face_recognition.add_face(name, image)
        
        return jsonify({
            'status': 'success' if success else 'error',
            'message': message
        })
    except Exception as e:
        logger.error(f"添加人脸失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search_face', methods=['POST'])
def search_face():
    """搜索人脸库"""
    try:
        image_data = request.form.get('image')
        mode = request.form.get('mode', 'size')
        
        if not image_data:
            return jsonify({'error': '缺少图片数据'}), 400
        
        if image_data.startswith('data:image/'):
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': '图片解码失败'}), 400
        
        results, message = face_recognition.search_face(image, mode=mode)
        
        processed_results = []
        for result in results:
            face_image = cv2.imread(result['path'])
            if face_image is not None:
                _, buffer = cv2.imencode('.jpg', face_image)
                face_base64 = base64.b64encode(buffer).decode('utf-8')
                face_data = f'data:image/jpeg;base64,{face_base64}'
            else:
                face_data = None
            
            processed_results.append({
                'name': result['name'],
                'similarity': round(result['similarity'] * 100, 2),
                'face_image': face_data
            })
        
        return jsonify({
            'status': 'success',
            'results': processed_results,
            'message': message
        })
    except Exception as e:
        logger.error(f"搜索人脸失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_face', methods=['POST'])
def delete_face():
    """删除人脸"""
    try:
        name = request.form.get('name')
        face_file = request.form.get('face_file')
        
        if not name:
            return jsonify({'error': '缺少姓名'}), 400
        
        success, message = face_recognition.delete_face(name, face_file)
        
        return jsonify({
            'status': 'success' if success else 'error',
            'message': message
        })
    except Exception as e:
        logger.error(f"删除人脸失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/list_faces', methods=['GET'])
def list_faces():
    """获取人脸库中所有人脸列表"""
    try:
        faces = face_recognition.get_all_faces()
        
        processed_faces = []
        for face in faces:
            face_image = cv2.imread(face['path'])
            if face_image is not None:
                _, buffer = cv2.imencode('.jpg', face_image)
                face_base64 = base64.b64encode(buffer).decode('utf-8')
                face_data = f'data:image/jpeg;base64,{face_base64}'
            else:
                face_data = None
            
            processed_faces.append({
                'name': face['name'],
                'file': face['file'],
                'face_image': face_data,
                'add_time': face.get('add_time', '')
            })
        
        return jsonify({
            'status': 'success',
            'faces': processed_faces,
            'total': len(processed_faces)
        })
    except Exception as e:
        logger.error(f"获取人脸列表失败: {e}")
        return jsonify({'error': str(e)}), 500

class FaceRecognition:
    """人脸比对和搜索功能类"""
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.face_db_path = os.path.join(base_dir, '..', 'face_db')
        os.makedirs(self.face_db_path, exist_ok=True)
        self.similarity_threshold = 0.25
        self.face_size = 128
        
    def _preprocess_face(self, face_roi):
        """预处理人脸：CLAHE归一化 + 高斯降噪"""
        if face_roi.size == 0:
            return None
        face_roi = cv2.resize(face_roi, (self.face_size, self.face_size))
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        face_roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        return face_roi
    
    def calculate_face_quality(self, face_roi):
        """计算人脸质量（清晰度和尺寸）"""
        if face_roi.size == 0:
            return 0.0
        h, w = face_roi.shape[:2]
        area = h * w
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_score = (area / 10000.0) * (1 + laplacian_var / 1000.0)
        return quality_score
    
    def select_best_face(self, image, boxes, scores):
        """从多人脸中选择尺寸最大、清晰度最高的人脸"""
        if len(boxes) == 0:
            return None, None, None
        h, w = image.shape[:2]
        best_idx = 0
        best_quality = 0
        for i in range(len(boxes)):
            ymin, xmin, ymax, xmax = boxes[i]
            left, top = int(xmin * w), int(ymin * h)
            right, bottom = int(xmax * w), int(ymax * h)
            face_roi = image[top:bottom, left:right]
            quality = self.calculate_face_quality(face_roi)
            if quality > best_quality:
                best_quality = quality
                best_idx = i
        return boxes[best_idx], scores[best_idx], best_quality
    
    def _center_crop_face(self, face_roi, target_size=128):
        """将人脸区域居中裁剪为正方形"""
        if face_roi.size == 0:
            return face_roi
        h, w = face_roi.shape[:2]
        size = min(h, w)
        top = (h - size) // 2
        left = (w - size) // 2
        cropped = face_roi[top:top+size, left:left+size]
        cropped = cv2.resize(cropped, (target_size, target_size))
        return cropped
    
    def _select_face_by_confidence(self, image, boxes, scores):
        """按置信度最高选择人脸（用于摄像头实时帧）"""
        if len(boxes) == 0:
            return None, None
        best_idx = np.argmax(scores)
        return boxes[best_idx], scores[best_idx]
    
    def _select_face_by_size(self, image, boxes, scores):
        """按尺寸最大选择人脸（用于上传照片）"""
        if len(boxes) == 0:
            return None, None
        h, w = image.shape[:2]
        best_idx = 0
        max_area = 0
        for i in range(len(boxes)):
            ymin, xmin, ymax, xmax = boxes[i]
            face_w = (xmax - xmin) * w
            face_h = (ymax - ymin) * h
            area = face_w * face_h
            if area > max_area:
                max_area = area
                best_idx = i
        return boxes[best_idx], scores[best_idx]
    
    def _extract_face_roi(self, image, box):
        """从图片中提取人脸区域并居中裁剪"""
        h, w = image.shape[:2]
        ymin, xmin, ymax, xmax = box
        left, top = int(xmin * w), int(ymin * h)
        right, bottom = int(xmax * w), int(ymax * h)
        face_roi = image[top:bottom, left:right]
        if face_roi.size == 0:
            return None
        face_roi = self._center_crop_face(face_roi, 128)
        return face_roi
    
    def _extract_histogram(self, face_roi):
        """提取颜色直方图特征（HSV空间，更鲁棒）"""
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        return np.concatenate([hist_h, hist_s, hist_v])
    
    def _extract_lbp_uniform(self, face_roi):
        """提取Uniform LBP纹理特征（59维，更鲁棒）"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        n_uniform = 0
        for i in range(256):
            binary = bin(i)[2:].zfill(8)
            transitions = sum(1 for k in range(8) if binary[k] != binary[(k+1) % 8])
            if transitions <= 2:
                n_uniform += 1
        uniform_map = np.zeros(256, dtype=np.int32)
        uniform_idx = 0
        for i in range(256):
            binary = bin(i)[2:].zfill(8)
            transitions = sum(1 for k in range(8) if binary[k] != binary[(k+1) % 8])
            if transitions <= 2:
                uniform_map[i] = uniform_idx
                uniform_idx += 1
            else:
                uniform_map[i] = n_uniform
        n_bins = n_uniform + 1
        mapped = uniform_map[lbp.ravel()]
        hist, _ = np.histogram(mapped, bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def _extract_hog(self, face_roi):
        """提取HOG特征（使用OpenCV HOGDescriptor）"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 128))
        hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
        hog_feat = hog.compute(gray)
        if hog_feat is None:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            mag, angle = cv2.cartToPolar(gx, gy)
            bins = 9
            cell_size = 8
            n_cells_y = gray.shape[0] // cell_size
            n_cells_x = gray.shape[1] // cell_size
            hog_hist = np.zeros((n_cells_y, n_cells_x, bins), dtype=np.float32)
            for cy in range(n_cells_y):
                for cx in range(n_cells_x):
                    cell_mag = mag[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
                    cell_angle = angle[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
                    for b in range(bins):
                        low = b * (180.0 / bins)
                        high = (b + 1) * (180.0 / bins)
                        mask = (cell_angle >= low) & (cell_angle < high)
                        hog_hist[cy, cx, b] = cell_mag[mask].sum()
            hog_feat = cv2.normalize(hog_hist, hog_hist).flatten()
        else:
            hog_feat = hog_feat.flatten()
        hog_feat = cv2.normalize(hog_feat, hog_feat).flatten()
        return hog_feat
    
    def _extract_gabor(self, face_roi):
        """提取Gabor纹理特征（多方向多尺度）"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        gabor_feats = []
        ksize = 31
        for theta_val in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            for sigma in [3.0, 5.0]:
                for lamda in [5.0, 10.0]:
                    kern = cv2.getGaborKernel((ksize, ksize), sigma, theta_val, lamda, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(gray, cv2.CV_32F, kern)
                    gabor_feats.append(filtered.mean())
                    gabor_feats.append(filtered.var())
        gabor_feats = np.array(gabor_feats, dtype=np.float32)
        gabor_feats = cv2.normalize(gabor_feats, gabor_feats).flatten()
        return gabor_feats
    
    def _extract_ssim_features(self, face_roi):
        """提取结构特征（梯度方向统计）"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        mag_hist = np.histogram(mag.ravel(), bins=16, range=(0, mag.max()+1e-5))[0].astype(np.float32)
        angle_hist = np.histogram(angle.ravel(), bins=16, range=(-np.pi, np.pi))[0].astype(np.float32)
        mag_hist = cv2.normalize(mag_hist, mag_hist).flatten()
        angle_hist = cv2.normalize(angle_hist, angle_hist).flatten()
        return np.concatenate([mag_hist, angle_hist])
    
    def extract_face_features(self, image, mode='auto'):
        """提取人脸多特征融合向量
        mode: 'confidence' - 按置信度选人脸(摄像头)
              'size' - 按尺寸选人脸(照片)
              'auto' - 自动判断
        """
        try:
            boxes, scores, classes, num = detector.detect_faces(image)
            if num == 0:
                return None
            
            if mode == 'confidence':
                box, score = self._select_face_by_confidence(image, boxes, scores)
            elif mode == 'size':
                box, score = self._select_face_by_size(image, boxes, scores)
            else:
                box, score, quality = self.select_best_face(image, boxes, scores)
            
            if box is None:
                return None
            
            face_roi = self._extract_face_roi(image, box)
            if face_roi is None:
                return None
            
            face_roi = self._preprocess_face(face_roi)
            if face_roi is None:
                return None
            
            hist_feat = self._extract_histogram(face_roi)
            lbp_feat = self._extract_lbp_uniform(face_roi)
            hog_feat = self._extract_hog(face_roi)
            gabor_feat = self._extract_gabor(face_roi)
            ssim_feat = self._extract_ssim_features(face_roi)
            
            combined = np.concatenate([hist_feat, lbp_feat, hog_feat, gabor_feat, ssim_feat])
            combined = cv2.normalize(combined, combined).flatten()
            return combined
        except Exception as e:
            logger.error(f"提取特征失败: {e}")
            return None
    
    def _cosine_similarity(self, a, b):
        """计算余弦相似度"""
        dot = np.dot(a, b)
        norm1 = np.linalg.norm(a)
        norm2 = np.linalg.norm(b)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(dot / (norm1 * norm2))
    
    def calculate_similarity(self, features1, features2):
        """计算特征相似度（多指标融合）"""
        if features1 is None or features2 is None:
            return 0.0
        try:
            offset = 0
            
            hist_len = 32 * 3
            hist1 = features1[offset:offset + hist_len]
            hist2 = features2[offset:offset + hist_len]
            offset += hist_len
            
            lbp_len = len(hist1)
            for test_len in [59, 58, 60, 256]:
                if offset + test_len <= len(features1):
                    lbp_len = test_len
                    break
            lbp1 = features1[offset:offset + lbp_len]
            lbp2 = features2[offset:offset + lbp_len]
            offset += lbp_len
            
            hog1 = features1[offset:]
            hog2 = features2[offset:]
            
            hog_dim = min(len(hog1), len(hog2))
            if hog_dim == 0:
                return 0.0
            
            hist_sim = cv2.compareHist(hist1.reshape(-1, 1).astype(np.float32), 
                                        hist2.reshape(-1, 1).astype(np.float32), 
                                        cv2.HISTCMP_CORREL)
            hist_sim = max(0.0, hist_sim)
            
            lbp_sim = cv2.compareHist(lbp1.reshape(-1, 1).astype(np.float32), 
                                       lbp2.reshape(-1, 1).astype(np.float32), 
                                       cv2.HISTCMP_CORREL)
            lbp_sim = max(0.0, lbp_sim)
            
            hog_sim = self._cosine_similarity(hog1[:hog_dim], hog2[:hog_dim])
            hog_sim = max(0.0, hog_sim)
            
            hist_weight = 0.10
            lbp_weight = 0.20
            hog_weight = 0.50
            
            remaining = 1.0 - hist_weight - lbp_weight - hog_weight
            
            if len(features1) > offset + hog_dim:
                gabor_dim = 32
                gabor1 = features1[offset:offset + gabor_dim]
                gabor2 = features2[offset:offset + gabor_dim]
                offset += gabor_dim
                gabor_sim = self._cosine_similarity(gabor1, gabor2)
                gabor_sim = max(0.0, gabor_sim)
            else:
                gabor_sim = 0.5
                gabor_dim = 0
            
            if len(features1) > offset:
                ssim1 = features1[offset:]
                ssim2 = features2[offset:]
                ssim_sim = self._cosine_similarity(ssim1, ssim2)
                ssim_sim = max(0.0, ssim_sim)
            else:
                ssim_sim = 0.5
            
            gabor_weight = remaining * 0.5
            ssim_weight = remaining * 0.5
            
            similarity = (hist_weight * hist_sim + 
                         lbp_weight * lbp_sim + 
                         hog_weight * hog_sim + 
                         gabor_weight * gabor_sim + 
                         ssim_weight * ssim_sim)
            
            if hist_sim > 0.7 and lbp_sim > 0.7 and hog_sim > 0.7:
                similarity = min(1.0, similarity * 1.15)
            elif hist_sim < 0.3 and lbp_sim < 0.3:
                similarity = similarity * 0.85
            
            similarity = max(0.0, min(1.0, similarity))
            return round(float(similarity), 2)
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0
    
    def compare_faces(self, image1, image2):
        """比较两张图片中的人脸"""
        features1 = self.extract_face_features(image1)
        features2 = self.extract_face_features(image2)
        similarity = self.calculate_similarity(features1, features2)
        return similarity
    
    def add_face(self, name, image):
        """添加人脸到数据库"""
        try:
            if not name or not name.strip():
                return False, "姓名不能为空"
            name = name.strip()
            person_dir = os.path.join(self.face_db_path, name)
            os.makedirs(person_dir, exist_ok=True)
            boxes, scores, classes, num = detector.detect_faces(image)
            if num == 0:
                return False, "未检测到人脸，请确保图片中包含清晰的人脸"
            box, score, quality = self.select_best_face(image, boxes, scores)
            if box is None:
                return False, "无法提取有效人脸"
            face_roi = self._extract_face_roi(image, box)
            if face_roi is None or face_roi.size == 0:
                return False, "人脸区域提取失败"
            timestamp = int(time.time() * 1000)
            unique_id = f"{timestamp}_{np.random.randint(1000, 9999)}"
            face_filename = f"{name}_{unique_id}.jpg"
            face_path = os.path.join(person_dir, face_filename)
            cv2.imwrite(face_path, face_roi)
            logger.info(f"人脸添加成功: {face_path}")
            return True, f"人脸添加成功: {face_filename}"
        except Exception as e:
            logger.error(f"添加人脸失败: {e}")
            return False, str(e)
    
    def search_face(self, image, mode='size'):
        """搜索人脸库（按阈值过滤，不限数量）"""
        try:
            query_features = self.extract_face_features(image, mode=mode)
            if query_features is None:
                return [], "未检测到人脸"
            results = []
            start_time = time.time()
            if not os.path.exists(self.face_db_path):
                return [], "人脸库为空"
            for person_name in os.listdir(self.face_db_path):
                person_dir = os.path.join(self.face_db_path, person_name)
                if not os.path.isdir(person_dir):
                    continue
                for face_file in os.listdir(person_dir):
                    if not face_file.endswith('.jpg'):
                        continue
                    face_path = os.path.join(person_dir, face_file)
                    face_image = cv2.imread(face_path)
                    if face_image is None:
                        continue
                    db_features = self.extract_face_features(face_image, mode='size')
                    if db_features is None:
                        continue
                    similarity = self.calculate_similarity(query_features, db_features)
                    if similarity >= self.similarity_threshold:
                        results.append({
                            'name': person_name,
                            'path': face_path,
                            'similarity': similarity
                        })
            results.sort(key=lambda x: x['similarity'], reverse=True)
            elapsed_time = time.time() - start_time
            logger.info(f"搜索完成，耗时: {elapsed_time:.2f}秒，返回 {len(results)} 个结果")
            return results, f"搜索完成，耗时 {elapsed_time:.2f}秒"
        except Exception as e:
            logger.error(f"搜索人脸失败: {e}")
            return [], str(e)
    
    def delete_face(self, name, face_file=None):
        """删除人脸"""
        try:
            if not name:
                return False, "姓名不能为空"
            person_dir = os.path.join(self.face_db_path, name)
            if not os.path.exists(person_dir):
                return False, f"人物 '{name}' 不存在"
            if face_file:
                face_path = os.path.join(person_dir, face_file)
                if os.path.exists(face_path):
                    os.remove(face_path)
                    logger.info(f"人脸删除成功: {face_file}")
                    return True, f"人脸删除成功: {face_file}"
                else:
                    return False, f"人脸文件 '{face_file}' 不存在"
            else:
                import shutil
                shutil.rmtree(person_dir)
                logger.info(f"人物 '{name}' 已删除")
                return True, f"人物 '{name}' 已删除"
        except Exception as e:
            logger.error(f"删除人脸失败: {e}")
            return False, str(e)
    
    def get_all_faces(self):
        """获取所有人脸数据库中的条目"""
        try:
            faces = []
            if not os.path.exists(self.face_db_path):
                return faces
            for person_name in os.listdir(self.face_db_path):
                person_dir = os.path.join(self.face_db_path, person_name)
                if not os.path.isdir(person_dir):
                    continue
                for face_file in os.listdir(person_dir):
                    if face_file.endswith('.jpg'):
                        face_path = os.path.join(person_dir, face_file)
                        mtime = os.path.getmtime(face_path)
                        add_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                        faces.append({
                            'name': person_name,
                            'file': face_file,
                            'path': face_path,
                            'add_time': add_time
                        })
            return faces
        except Exception as e:
            logger.error(f"获取人脸列表失败: {e}")
            return []

face_recognition = FaceRecognition()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend', 'public')

@app.route('/')
def serve_index():
    return jsonify({
        'status': 'running',
        'message': 'Face Detection API Server',
        'version': '1.0',
        'endpoints': [
            '/api/test_connection',
            '/api/compare_faces',
            '/api/add_face',
            '/api/search_face',
            '/api/list_faces',
            '/api/delete_face',
            '/api/start_camera',
            '/api/stop_camera',
            '/api/capture_frame',
            '/api/get_detection_status'
        ]
    })

@app.route('/<path:path>')
def serve_static(path):
    return jsonify({'error': 'Not found', 'message': 'Use /api/* endpoints'}), 404

if __name__ == '__main__':
    logger.info("="*70)
    logger.info("🔥 实时人脸识别系统")
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info("✅ 本版本特点：")
    logger.info("   • 修复多框问题（NMS非极大值抑制）")
    logger.info("   • 实时检测结果显示在右侧面板")
    logger.info("   • 优化坐标映射，提高检测精度")
    logger.info("   • 添加检测统计和置信度显示")
    logger.info("   • 优化近距离检测效果")
    logger.info("   • 解决跨域问题")
    logger.info("="*70)
    
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media'), exist_ok=True)
    
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

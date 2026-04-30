var isDetecting = false;
var logMessages = [];
var uploadedImage = null;
var uploadedImageData = null;
var cameraStream = null;
var cameraVideoElement = null;
var detectionIntervalId = null;
var faceDetectedInCamera = false;

function showToast(message, type) {
    type = type || 'info';
    var toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(function() { toast.remove(); }, 3500);
}

function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showInputModal(title, placeholder, confirmCallback) {
    document.getElementById('inputModalTitle').textContent = title;
    document.getElementById('inputModalField').placeholder = placeholder;
    document.getElementById('inputModalField').value = '';
    document.getElementById('inputModal').style.display = 'flex';
    document.getElementById('inputModalConfirm').onclick = function() {
        var value = document.getElementById('inputModalField').value;
        closeInputModal();
        if (confirmCallback) confirmCallback(value);
    };
    setTimeout(function() { document.getElementById('inputModalField').focus(); }, 100);
}

function closeInputModal() {
    document.getElementById('inputModal').style.display = 'none';
}

function addLog(message) {
    var timestamp = new Date().toLocaleTimeString();
    var logEntry = '[' + timestamp + '] ' + message + '<br>';
    logMessages.push(logEntry);
    if (logMessages.length > 30) logMessages = logMessages.slice(-30);
    var panel = document.getElementById('logPanel');
    if (panel) {
        panel.innerHTML = '系统日志：<br>' + logMessages.join('');
        panel.scrollTop = panel.scrollHeight;
    }
}

function showCameraError(show, message) {
    var overlay = document.getElementById('cameraErrorOverlay');
    var hint = document.getElementById('cameraErrorHint');
    if (overlay) overlay.style.display = show ? 'flex' : 'none';
    if (hint && message) hint.textContent = message;
}

async function startCamera() {
    try {
        addLog('正在请求摄像头权限...');
        
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        
        cameraVideoElement = document.getElementById('cameraVideo');
        cameraVideoElement.srcObject = cameraStream;
        cameraVideoElement.style.display = 'block';
        
        cameraVideoElement.onloadedmetadata = function() {
            cameraVideoElement.play();
            isDetecting = true;
            faceDetectedInCamera = false;
            
            document.getElementById('status').textContent = '检测已启动';
            document.getElementById('status').className = 'status running';
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('startBtn').disabled = true;
            
            showCameraError(false);
            addLog('摄像头已启动');
            showToast('摄像头已启动', 'success');
            
            document.getElementById('detectionInfo').innerHTML = 
                '<p>检测已启动</p><p>正在识别人脸...</p>';
            
            if (detectionIntervalId) clearInterval(detectionIntervalId);
            detectionIntervalId = setInterval(runFaceDetection, 1000);
        };
        
    } catch (err) {
        addLog('摄像头权限被拒绝或无法访问');
        showCameraError(true, '请允许浏览器访问摄像头，或检查摄像头是否被其他程序占用');
        showToast('无法访问摄像头: ' + (err.message || '权限被拒绝'), 'error');
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(function(track) { track.stop(); });
        cameraStream = null;
    }
    
    if (cameraVideoElement) {
        cameraVideoElement.srcObject = null;
        cameraVideoElement.style.display = 'none';
    }
    
    if (detectionIntervalId) {
        clearInterval(detectionIntervalId);
        detectionIntervalId = null;
    }
    
    isDetecting = false;
    faceDetectedInCamera = false;
    
    document.getElementById('status').textContent = '检测已停止';
    document.getElementById('status').className = 'status stopped';
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('startBtn').disabled = false;
    document.getElementById('detectionInfo').innerHTML = 
        '<p>检测已停止</p><p>点击"开始检测"重新启动</p>';
    
    addLog('摄像头已停止');
    showToast('检测已停止', 'info');
}

function captureFrame() {
    if (!cameraVideoElement || !cameraVideoElement.readyState >= 2) {
        return null;
    }
    
    var canvas = document.createElement('canvas');
    canvas.width = cameraVideoElement.videoWidth || 640;
    canvas.height = cameraVideoElement.videoHeight || 480;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(cameraVideoElement, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
}

async function runFaceDetection() {
    if (!isDetecting || !cameraStream) return;
    
    try {
        var imageData = captureFrame();
        if (!imageData) return;
        
        var data = await ApiClient.searchFace(imageData, 'confidence');
        
        if (data.status === 'success') {
            var count = (data.results || []).length;
            if (count > 0) {
                faceDetectedInCamera = true;
                var maxScore = Math.max.apply(null, data.results.map(function(r) { return r.similarity; }));
                document.getElementById('currentConfidence').textContent = maxScore.toFixed(2) + '%';
                document.getElementById('totalFaces').textContent =
                    parseInt(document.getElementById('totalFaces').textContent || 0) + 1;
                document.getElementById('detectionInfo').innerHTML = 
                    '<p>检测到 <strong>' + count + '</strong> 个人脸</p>' +
                    '<p>最高匹配: <strong>' + maxScore.toFixed(2) + '%</strong></p>' +
                    '<p>匹配人员: ' + data.results.map(function(r) { return r.name; }).join(', ') + '</p>';
            } else {
                faceDetectedInCamera = false;
                document.getElementById('detectionInfo').innerHTML = 
                    '<p>检测中...</p><p>未检测到已知人脸</p>';
            }
        }
    } catch (err) {
        console.warn('实时检测更新失败:', err);
    }
}

async function testConnection() {
    try {
        addLog('正在测试连接...');
        showLoading();
        var data = await ApiClient.getTestConnection();
        hideLoading();
        addLog('连接测试成功');
        showToast('连接测试成功', 'success');
        document.getElementById('dirPath').textContent = data.current_dir;
        document.getElementById('pyVersion').textContent = data.python_version;
        document.getElementById('tfVersion').textContent = data.tensorflow_version;
        document.getElementById('modelStatus').textContent = data.has_model ? '已找到' : '未找到';
        updateDetectionDisplay(data.last_detection);
        setTimeout(function() { listFaces(true); }, 500);
    } catch (error) {
        hideLoading();
        addLog('网络错误: ' + error.message);
        showToast('网络错误: ' + error.message + ' - 请确保后端服务在运行', 'error');
    }
}

function showSystemInfo() {
    testConnection();
}

function updateDetectionDisplay(detectionData) {
    if (!detectionData) return;
    var totalEl = document.getElementById('totalFaces');
    var confEl = document.getElementById('currentConfidence');
    if (totalEl) totalEl.textContent = detectionData.count || 0;
    if (confEl && detectionData.scores && detectionData.scores.length > 0) {
        confEl.textContent = Math.max.apply(null, detectionData.scores).toFixed(2) + '%';
    }
}

function compressImage(imageData, maxWidth, quality) {
    maxWidth = maxWidth || 800;
    quality = quality || 0.8;
    return new Promise(function(resolve) {
        var img = new Image();
        img.onload = function() {
            var canvas = document.createElement('canvas');
            var w = img.width;
            var h = img.height;
            if (w > maxWidth) {
                h = Math.round(h * maxWidth / w);
                w = maxWidth;
            }
            canvas.width = w;
            canvas.height = h;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, w, h);
            resolve(canvas.toDataURL('image/jpeg', quality));
        };
        img.src = imageData;
    });
}

function handleImageUpload(event) {
    var file = event.target.files[0];
    if (!file) return;
    var validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (validTypes.indexOf(file.type) === -1) {
        showToast('仅支持 JPEG/PNG 格式', 'error');
        return;
    }
    addLog('上传图片: ' + file.name + ' (' + (file.size / 1024 / 1024).toFixed(2) + ' MB)');
    var reader = new FileReader();
    reader.onload = function(e) {
        var imgEl = document.getElementById('uploadedImage');
        if (imgEl) {
            imgEl.src = e.target.result;
            imgEl.style.display = 'block';
        }
        var placeholder = document.getElementById('uploadPlaceholder');
        if (placeholder) placeholder.style.display = 'none';
        var removeBtn = document.getElementById('removeImageBtn');
        if (removeBtn) removeBtn.style.display = 'block';
        uploadedImageData = e.target.result;
        addLog('图片上传成功');
        showToast('图片上传成功', 'success');
    };
    reader.readAsDataURL(file);
}

function removeUploadedImage() {
    var imgEl = document.getElementById('uploadedImage');
    if (imgEl) { imgEl.src = ''; imgEl.style.display = 'none'; }
    var placeholder = document.getElementById('uploadPlaceholder');
    if (placeholder) placeholder.style.display = 'flex';
    var removeBtn = document.getElementById('removeImageBtn');
    if (removeBtn) removeBtn.style.display = 'none';
    uploadedImageData = null;
    document.getElementById('imageUpload').value = '';
    addLog('已移除上传图片');
}

async function compareFaces() {
    var image1Data = null;
    var image2Data = uploadedImageData;
    
    if (!image2Data) {
        showToast('请先上传图片', 'error');
        return;
    }
    
    if (isDetecting) {
        image1Data = captureFrame();
        if (!image1Data) {
            showToast('无法从摄像头捕获画面', 'error');
            return;
        }
    } else {
        showToast('请先启动摄像头检测', 'error');
        return;
    }
    
    try {
        addLog('正在进行人脸比对...');
        showLoading();
        var data = await ApiClient.compareFaces(
            await compressImage(image1Data),
            await compressImage(image2Data),
            'confidence',
            'size'
        );
        hideLoading();
        addLog('比对完成: ' + data.message);
        var similarityContainer = document.getElementById('similarityContainer');
        var similarityProgress = document.getElementById('similarityProgress');
        var similarityText = document.getElementById('similarityText');
        var simPlaceholder = document.getElementById('similarityPlaceholder');
        if (similarityContainer) similarityContainer.style.display = 'block';
        if (simPlaceholder) simPlaceholder.style.display = 'none';
        if (similarityProgress) similarityProgress.style.width = (data.similarity || 0) + '%';
        if (similarityText) similarityText.textContent = '相似度: ' + (data.similarity || 0) + '%';
        var badgeEl = document.getElementById('similarityBadge');
        if (badgeEl) {
            var sim = data.similarity || 0;
            var badgeClass = sim >= 70 ? 'high' : (sim >= 50 ? 'medium' : 'low');
            var badgeText = sim >= 70 ? '高度相似' : (sim >= 50 ? '中度相似' : '低度相似');
            badgeEl.className = 'similarity-badge ' + badgeClass;
            badgeEl.textContent = badgeText;
            badgeEl.style.display = 'inline-block';
        }
        showToast('比对完成，相似度 ' + (data.similarity || 0) + '%', 'info');
    } catch (error) {
        hideLoading();
        addLog('比对错误: ' + error.message);
        showToast('比对失败: ' + error.message, 'error');
    }
}

async function addFace(name, source) {
    var imageData = null;
    if (source === 'upload') {
        imageData = uploadedImageData;
    } else if (source === 'camera') {
        imageData = captureFrame();
    }
    if (!imageData) {
        showToast('无法获取人脸图像', 'error');
        return;
    }
    try {
        addLog('正在添加人脸: ' + name + '...');
        showLoading();
        var data = await ApiClient.addFace(name, await compressImage(imageData));
        hideLoading();
        if (data.status === 'success') {
            addLog(data.message);
            showToast(data.message, 'success');
            listFaces();
        } else {
            addLog('添加失败: ' + data.message);
            showToast('添加失败: ' + data.message, 'error');
        }
    } catch (error) {
        hideLoading();
        addLog('添加错误: ' + error.message);
        showToast('添加失败: ' + error.message, 'error');
    }
}

function showAddFaceModal() {
    var hasCamera = isDetecting && faceDetectedInCamera;
    var hasUpload = !!uploadedImageData;
    if (!hasCamera && !hasUpload) {
        if (isDetecting) {
            showToast('摄像头未检测到人脸，请正对摄像头或上传包含人脸的图片', 'error');
        } else {
            showToast('没有检测到人脸，请先启动摄像头或上传包含人脸的图片', 'error');
        }
        return;
    }
    if (hasCamera && hasUpload) {
        showSourceSelectModal(
            '添加人脸 - 选择来源',
            '摄像头和上传图片均可用，请选择使用哪个来源：',
            function() {
                showInputModal('添加人脸（摄像头）', '请输入人物姓名', function(name) {
                    if (name && name.trim()) addFace(name.trim(), 'camera');
                });
            },
            function() {
                showInputModal('添加人脸（上传图片）', '请输入人物姓名', function(name) {
                    if (name && name.trim()) addFace(name.trim(), 'upload');
                });
            }
        );
    } else if (hasUpload) {
        showInputModal('添加人脸', '请输入人物姓名', function(name) {
            if (name && name.trim()) addFace(name.trim(), 'upload');
        });
    } else {
        showInputModal('添加人脸', '请输入人物姓名', function(name) {
            if (name && name.trim()) addFace(name.trim(), 'camera');
        });
    }
}

async function searchFace(source) {
    var imageData = null;
    if (source === 'upload') {
        imageData = uploadedImageData;
    } else if (source === 'camera') {
        imageData = captureFrame();
    }
    if (!imageData) {
        showToast('无法获取人脸图像', 'error');
        return;
    }
    try {
        addLog('正在搜索人脸...');
        showLoading();
        var searchMode = source === 'camera' ? 'confidence' : 'size';
        var data = await ApiClient.searchFace(await compressImage(imageData), searchMode);
        hideLoading();
        addLog('搜索完成: ' + data.message);
        var searchResultsContainer = document.getElementById('searchResultsContainer');
        var searchResults = document.getElementById('searchResults');
        var searchPlaceholder = document.getElementById('searchPlaceholder');
        if (searchResultsContainer) searchResultsContainer.style.display = 'block';
        if (searchPlaceholder) searchPlaceholder.style.display = 'none';
        if (searchResults) {
            if (!data.results || data.results.length === 0) {
                searchResults.innerHTML = '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;min-height:120px;color:#94a3b8;"><div style="font-size:36px;margin-bottom:8px;">&#128269;</div><div style="font-size:13px;">未找到匹配的人脸</div><div style="font-size:11px;margin-top:4px;">匹配度低于25%的结果已过滤</div></div>';
                showToast('未找到匹配的人脸', 'info');
            } else {
                searchResults.innerHTML = '';
                data.results.forEach(function(result) {
                    var sim = result.similarity || 0;
                    var matchClass = sim >= 70 ? 'high' : (sim >= 50 ? 'medium' : 'low');
                    var matchText = sim >= 70 ? '高度匹配' : (sim >= 50 ? '中度匹配' : '低度匹配');
                    var resultItem = document.createElement('div');
                    resultItem.className = 'result-item match-' + matchClass;
                    resultItem.innerHTML = 
                        '<img src="' + (result.face_image || '') + '" alt="' + result.name + '" data-face-src="' + (result.face_image || '') + '" data-face-name="' + result.name + '">' +
                        '<div class="result-name">' + result.name + '</div>' +
                        '<div class="result-similarity ' + matchClass + '">' + sim + '%</div>' +
                        '<div class="match-label ' + matchClass + '">' + matchText + '</div>';
                    searchResults.appendChild(resultItem);
                });
                showToast('找到 ' + data.results.length + ' 个匹配结果', 'success');
            }
        }
        updateScrollIndicators('searchResultsScroll');
    } catch (error) {
        hideLoading();
        addLog('搜索错误: ' + error.message);
        showToast('搜索失败: ' + error.message, 'error');
    }
}

function onSearchFaceClick() {
    var hasCamera = isDetecting;
    var hasUpload = !!uploadedImageData;
    if (!hasCamera && !hasUpload) {
        showToast('请先启动摄像头或上传图片', 'error');
        return;
    }
    if (hasCamera && hasUpload) {
        showSourceSelectModal(
            '搜索人脸 - 选择来源',
            '摄像头和上传图片均可用，请选择使用哪个来源进行搜索：',
            function() { searchFace('camera'); },
            function() { searchFace('upload'); }
        );
    } else if (hasUpload) {
        searchFace('upload');
    } else {
        searchFace('camera');
    }
}

async function listFaces(silent) {
    try {
        if (!silent) { addLog('正在获取人脸列表...'); showLoading(); }
        var data = await ApiClient.listFaces();
        if (!silent) hideLoading();
        if (!silent) addLog('获取到 ' + (data.total || 0) + ' 个人脸');
        var faceListContainer = document.getElementById('faceListContainer');
        var faceList = document.getElementById('faceList');
        var facePlaceholder = document.getElementById('facePlaceholder');
        if (faceListContainer) faceListContainer.style.display = 'block';
        if (facePlaceholder) facePlaceholder.style.display = 'none';
        if (faceList) {
            if (!data.faces || data.faces.length === 0) {
                faceList.innerHTML = '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;min-height:140px;color:#94a3b8;"><div style="font-size:36px;margin-bottom:8px;">&#128100;</div><div style="font-size:13px;">人脸库为空</div><div style="font-size:11px;margin-top:4px;">点击"添加人脸"开始使用</div></div>';
            } else {
                faceList.innerHTML = '';
                data.faces.forEach(function(face) {
                    var faceItem = document.createElement('div');
                    faceItem.className = 'face-list-item';
                    var addTime = face.add_time || '';
                    faceItem.innerHTML = 
                        '<img src="' + (face.face_image || '') + '" alt="' + face.name + '" data-face-src="' + (face.face_image || '') + '" data-face-name="' + face.name + '">' +
                        '<div class="face-list-item-name">' + face.name + '</div>' +
                        (addTime ? '<div class="face-list-item-time">' + addTime + '</div>' : '') +
                        '<button class="face-list-item-delete" onclick="deleteFace(\'' + face.name + '\', \'' + face.file + '\')">删除</button>';
                    faceList.appendChild(faceItem);
                });
            }
        }
        updateScrollIndicators('faceListScroll');
        if (!silent) showToast('共 ' + (data.total || 0) + ' 个人脸', 'success');
    } catch (error) {
        if (!silent) { hideLoading(); addLog('获取错误: ' + error.message); showToast('获取失败: ' + error.message, 'error'); }
    }
}

async function deleteFace(name, file) {
    if (!confirm('确定要删除 "' + name + '" 的这张人脸吗？')) return;
    try {
        addLog('正在删除人脸: ' + name + '...');
        showLoading();
        var data = await ApiClient.deleteFace(name, file);
        hideLoading();
        if (data.status === 'success') {
            addLog(data.message);
            showToast(data.message, 'success');
            listFaces();
        } else {
            addLog('删除失败: ' + data.message);
            showToast('删除失败: ' + data.message, 'error');
        }
    } catch (error) {
        hideLoading();
        addLog('删除错误: ' + error.message);
        showToast('删除失败: ' + error.message, 'error');
    }
}

function showDeleteFaceModal() {
    showInputModal('删除人脸', '请输入要删除的人物姓名', function(name) {
        if (name && name.trim()) {
            if (!confirm('确定要删除 "' + name.trim() + '" 的所有人脸吗？此操作不可恢复！')) return;
            ApiClient.deleteFace(name.trim()).then(function(data) {
                if (data.status === 'success') {
                    addLog(data.message);
                    showToast(data.message, 'success');
                    listFaces();
                } else {
                    addLog('删除失败: ' + data.message);
                    showToast('删除失败: ' + data.message, 'error');
                }
            }).catch(function(error) {
                showToast('删除失败: ' + error.message, 'error');
            });
        }
    });
}

function showSourceSelectModal(title, desc, cameraCallback, uploadCallback) {
    document.getElementById('sourceSelectTitle').textContent = title;
    document.getElementById('sourceSelectDesc').textContent = desc;
    document.getElementById('sourceSelectModal').style.display = 'flex';
    document.getElementById('sourceSelectCamera').onclick = function() {
        document.getElementById('sourceSelectModal').style.display = 'none';
        if (cameraCallback) cameraCallback();
    };
    document.getElementById('sourceSelectUpload').onclick = function() {
        document.getElementById('sourceSelectModal').style.display = 'none';
        if (uploadCallback) uploadCallback();
    };
}

function scrollContainer(trackId, direction) {
    var track = document.getElementById(trackId);
    if (!track) return;
    track.scrollBy({ left: direction * 200, behavior: 'smooth' });
}

function updateScrollIndicators(trackId) {
    var track = document.getElementById(trackId);
    if (!track) return;
    var wrapper = track.closest('.scroll-wrapper');
    if (!wrapper) return;
    var leftArrow = wrapper.querySelector('.scroll-left');
    var rightArrow = wrapper.querySelector('.scroll-right');
    var sl = track.scrollLeft;
    var maxScroll = track.scrollWidth - track.clientWidth;
    if (maxScroll <= 0) {
        track.classList.add('at-start', 'at-end');
        if (leftArrow) leftArrow.classList.remove('visible');
        if (rightArrow) rightArrow.classList.remove('visible');
        return;
    }
    if (sl <= 2) track.classList.add('at-start');
    else track.classList.remove('at-start');
    if (sl >= maxScroll - 2) track.classList.add('at-end');
    else track.classList.remove('at-end');
    if (leftArrow) leftArrow.classList.toggle('visible', sl > 2);
    if (rightArrow) rightArrow.classList.toggle('visible', sl < maxScroll - 2);
}

function initDragScroll(track) {
    if (!track) return;
    var isDown = false, startX = 0, scrollLeft = 0;
    track.addEventListener('mousedown', function(e) {
        if (e.target.tagName === 'BUTTON' || e.target.tagName === 'IMG') return;
        isDown = true;
        track.style.cursor = 'grabbing';
        startX = e.pageX - track.offsetLeft;
        scrollLeft = track.scrollLeft;
    });
    track.addEventListener('mouseleave', function() { isDown = false; track.style.cursor = ''; });
    track.addEventListener('mouseup', function() { isDown = false; track.style.cursor = ''; });
    track.addEventListener('mousemove', function(e) {
        if (!isDown) return;
        e.preventDefault();
        var x = e.pageX - track.offsetLeft;
        track.scrollLeft = scrollLeft - (x - startX) * 1.5;
    });
    track.addEventListener('scroll', function() { updateScrollIndicators(track.id); });
}

var _previewTimer = null;

function showFacePreview(imgSrc, name, anchorRect) {
    var tooltip = document.getElementById('facePreviewTooltip');
    var previewImg = document.getElementById('facePreviewImg');
    var previewName = document.getElementById('facePreviewName');
    if (!tooltip || !previewImg) return;
    previewImg.src = imgSrc;
    if (previewName) previewName.textContent = name || '';
    var gap = 12, tooltipW = 256, tooltipH = 280;
    var vw = window.innerWidth, vh = window.innerHeight;
    var left, top;
    if (anchorRect.right + gap + tooltipW < vw) left = anchorRect.right + gap;
    else left = anchorRect.left - gap - tooltipW;
    if (left < 8) left = 8;
    top = anchorRect.top + (anchorRect.height / 2) - (tooltipH / 2);
    if (top < 8) top = 8;
    if (top + tooltipH > vh - 8) top = vh - tooltipH - 8;
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
    tooltip.classList.add('active');
}

function hideFacePreview() {
    var tooltip = document.getElementById('facePreviewTooltip');
    if (tooltip) tooltip.classList.remove('active');
}

function initFacePreviewHandlers() {
    document.addEventListener('mouseenter', function(e) {
        var img = e.target;
        if (img.tagName === 'IMG' && img.dataset.faceSrc) {
            if (_previewTimer) clearTimeout(_previewTimer);
            _previewTimer = setTimeout(function() {
                var rect = img.getBoundingClientRect();
                showFacePreview(img.dataset.faceSrc, img.dataset.faceName || '', rect);
            }, 150);
        }
    }, true);
    document.addEventListener('mouseleave', function(e) {
        var img = e.target;
        if (img.tagName === 'IMG' && img.dataset.faceSrc) {
            if (_previewTimer) clearTimeout(_previewTimer);
            hideFacePreview();
        }
    }, true);
    var longPressTimer = null;
    document.addEventListener('touchstart', function(e) {
        var img = e.target;
        if (img.tagName === 'IMG' && img.dataset.faceSrc) {
            longPressTimer = setTimeout(function() {
                var rect = img.getBoundingClientRect();
                showFacePreview(img.dataset.faceSrc, img.dataset.faceName || '', rect);
            }, 500);
        }
    }, { passive: true });
    document.addEventListener('touchend', function() {
        if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
        hideFacePreview();
    });
    document.addEventListener('touchmove', function() {
        if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
        hideFacePreview();
    });
}

window.addEventListener('DOMContentLoaded', function() {
    uploadedImage = document.getElementById('uploadedImage');
    addLog('系统初始化完成');
    setTimeout(testConnection, 800);
    var searchScroll = document.getElementById('searchResultsScroll');
    var faceScroll = document.getElementById('faceListScroll');
    initDragScroll(searchScroll);
    initDragScroll(faceScroll);
    initFacePreviewHandlers();
});

document.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        var modal = document.getElementById('inputModal');
        if (modal && modal.style.display !== 'none') {
            document.getElementById('inputModalConfirm').click();
        }
    }
});

var uploadWrapper = document.getElementById('uploadAreaWrapper');
if (uploadWrapper) {
    uploadWrapper.addEventListener('dragover', function(e) {
        e.preventDefault();
        var ph = document.getElementById('uploadPlaceholder');
        if (ph) { ph.style.borderColor = '#3b82f6'; ph.style.background = '#1e293b'; }
    });
    uploadWrapper.addEventListener('dragleave', function(e) {
        e.preventDefault();
        var ph = document.getElementById('uploadPlaceholder');
        if (ph) { ph.style.borderColor = '#334155'; ph.style.background = '#0f172a'; }
    });
    uploadWrapper.addEventListener('drop', function(e) {
        e.preventDefault();
        var ph = document.getElementById('uploadPlaceholder');
        if (ph) { ph.style.borderColor = '#334155'; ph.style.background = '#0f172a'; }
        var files = e.dataTransfer.files;
        if (files.length > 0) {
            document.getElementById('imageUpload').files = files;
            handleImageUpload({ target: { files: files } });
        }
    });
}

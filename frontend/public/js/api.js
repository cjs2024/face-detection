var _apiBaseUrl = null;

function getApiBaseUrl() {
    if (_apiBaseUrl) return _apiBaseUrl;
    
    if (typeof window !== 'undefined') {
        if (window.__API_BASE_URL__) {
            _apiBaseUrl = window.__API_BASE_URL__;
        } else {
            var hostname = window.location.hostname;
            if (hostname === 'localhost' || hostname === '127.0.0.1') {
                _apiBaseUrl = 'http://localhost:10000/api';
            } else {
                _apiBaseUrl = 'https://face-detection-api.onrender.com/api';
            }
        }
    } else {
        _apiBaseUrl = 'https://face-detection-api.onrender.com/api';
    }
    return _apiBaseUrl;
}

class ApiClient {
    constructor() {
        this.baseUrl = getApiBaseUrl();
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultOptions = { mode: 'cors' };
        const config = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, config);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    async getTestConnection() {
        return this.request('/test_connection');
    }

    async startCamera() {
        return this.request('/start_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
    }

    async stopCamera() {
        return this.request('/stop_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
    }

    async getDetectionStatus() {
        return this.request('/get_detection_status');
    }

    async compareFaces(image1Data, image2Data, mode1, mode2) {
        var params = { image1: image1Data, image2: image2Data };
        if (mode1) params.mode1 = mode1;
        if (mode2) params.mode2 = mode2;
        return this.request('/compare_faces', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams(params),
        });
    }

    async addFace(name, imageData) {
        return this.request('/add_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ name: name, image: imageData }),
        });
    }

    async searchFace(imageData, mode) {
        var params = { image: imageData };
        if (mode) params.mode = mode;
        return this.request('/search_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams(params),
        });
    }

    async deleteFace(name, faceFile) {
        var params = { name: name };
        if (faceFile) params.face_file = faceFile;
        return this.request('/delete_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams(params),
        });
    }

    async listFaces() {
        return this.request('/list_faces');
    }

    async captureFrame() {
        return this.request('/capture_frame');
    }
}

window.ApiClient = ApiClient;

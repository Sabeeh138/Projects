import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export async function getMetrics() {
    return axios.get(`${API_BASE}/metrics`).then(r => r.data);
}

export async function getFeatureNames() {
    return axios.get(`${API_BASE}/feature-names`).then(r => r.data);
}

export async function predict(payload) {
    return axios.post(`${API_BASE}/predict`, payload).then(r => r.data);
}

export async function recommend(payload) {
    return axios.post(`${API_BASE}/recommend`, payload).then(r => r.data);
}

export async function limeExplain(payload) {
    return axios.post(`${API_BASE}/lime`, payload).then(r => r.data);
}

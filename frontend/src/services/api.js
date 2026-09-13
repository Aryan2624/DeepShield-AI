import axios from "axios";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

export async function getHealthStatus() {
  const response = await api.get("/api/health");
  return response.data;
}

export async function analyzeUrl(url) {
  const response = await api.post("/api/url/analyze", {
    url,
  });

  return response.data;
}

export default api;
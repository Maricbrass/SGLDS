import axios from 'axios'

const defaultApiBaseUrl = typeof window === 'undefined'
  ? 'http://localhost:8000/api/v1'
  : `${window.location.protocol}//${window.location.hostname}:8000/api/v1`

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined) || defaultApiBaseUrl

const api = axios.create({
  baseURL: apiBaseUrl,
  timeout: 120000,
})

export default api

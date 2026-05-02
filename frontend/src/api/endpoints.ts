import api from "./client";
import type {
  AnalyzeHistoryResponse,
  AnalyzeRunResponse,
  AnalyzeStartResponse,
  ConfigResponse,
  EuclidImagesResponse,
  EuclidSearchResponse,
  FetchCutoutResponse,
  GalleryResponse,
  ResultsStats,
  SystemStats,
  TrainingHistoryResponse,
} from "./types";

export async function searchEuclid(params: {
  target_name?: string;
  ra?: number;
  dec?: number;
  radius_arcsec?: number;
}) {
  const response = await api.get<EuclidSearchResponse>("/euclid/search", { params });
  return response.data;
}

export async function fetchEuclidCutout(payload: {
  s3_url: string;
  euclid_id?: string;
  cutout_size_arcmin?: number;
  target_ra?: number;
  target_dec?: number;
}) {
  const response = await api.post<FetchCutoutResponse>("/euclid/fetch", payload, {
    timeout: 300000,
  });
  return response.data;
}

export async function listCachedImages(limit = 50) {
  const response = await api.get<EuclidImagesResponse>("/euclid/images", { params: { limit } });
  return response.data;
}

export async function startAnalysis(imageId: number, force = false) {
  const response = await api.post<AnalyzeStartResponse>(`/analyze/image/${imageId}`, null, {
    params: { force },
  });
  return response.data;
}

export async function getRun(runId: number) {
  const response = await api.get<AnalyzeRunResponse>(`/analyze/runs/${runId}`);
  return response.data;
}

export async function getRunHistory(imageId: number, limit = 10) {
  const response = await api.get<AnalyzeHistoryResponse>(`/analyze/image/${imageId}/history`, {
    params: { limit },
  });
  return response.data;
}

export async function getSystemStats() {
  const response = await api.get<SystemStats>("/stats");
  return response.data;
}

export async function getResultsStats() {
  const response = await api.get<ResultsStats>("/results/stats");
  return response.data;
}

export async function getGallery(params: { min_confidence?: number; limit?: number; skip?: number }) {
  const response = await api.get<GalleryResponse>("/results/gallery", { params });
  return response.data;
}

export async function getTrainingHistory(limit = 10) {
  const response = await api.get<TrainingHistoryResponse>("/training/history", {
    params: { limit },
  });
  return response.data;
}

export async function getConfig() {
  const response = await api.get<ConfigResponse>("/config");
  return response.data;
}

export async function getTrainingRunMetrics(runId: number) {
  const response = await api.get(`/training/${runId}/metrics`);
  return response.data;
}

export async function getEvaluation(runId: number) {
  const response = await api.get(`/evaluation/${runId}`);
  return response.data;
}

export async function updateConfig(payload: Partial<ConfigResponse["config"]>) {
  const response = await api.put<{ status: string; config: ConfigResponse["config"] }>("/config", payload);
  return response.data;
}

export function toErrorMessage(error: unknown): string {
  if (typeof error === "object" && error !== null) {
    const maybeAxios = error as { response?: { data?: { detail?: string } } };
    if (maybeAxios.response?.data?.detail) {
      return maybeAxios.response.data.detail;
    }
  }
  return "Request failed";
}

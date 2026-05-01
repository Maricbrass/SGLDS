export type ApiStatus = "pending" | "queued" | "running" | "completed" | "failed";

export interface EuclidSearchResult {
  euclid_id?: string;
  filename?: string;
  filter?: string;
  instrument?: string;
  s3_url: string;
}

export interface EuclidSearchResponse {
  count: number;
  results: EuclidSearchResult[];
}

export interface EuclidImage {
  id: number;
  euclid_id?: string;
  source?: string;
  s3_url?: string;
  local_path?: string;
  fetch_date?: string;
}

export interface EuclidImagesResponse {
  count: number;
  images: EuclidImage[];
}

export interface FetchCutoutResponse {
  image_id: number;
  local_path: string;
}

export interface AnalyzeStartResponse {
  run_id: number;
  status: ApiStatus;
  cached: boolean;
  consensus_result?: {
    final_confidence?: number;
    final_prediction?: number;
  };
}

export interface AnalyzeRunResponse {
  id: number;
  image_id: number;
  status: ApiStatus;
  stage_1_result?: Record<string, unknown>;
  stage_2_results?: Record<string, unknown>;
  stage_3_results?: Record<string, unknown>;
  consensus_result?: {
    final_confidence?: number;
    final_prediction?: number;
  };
  analysis_time_seconds?: number;
  error_message?: string;
  heatmap_path?: string;
}

export interface AnalyzeHistoryItem {
  id: number;
  status: ApiStatus;
  run_timestamp?: string;
  consensus_result?: {
    final_confidence?: number;
    final_prediction?: number;
  };
}

export interface AnalyzeHistoryResponse {
  count: number;
  runs: AnalyzeHistoryItem[];
}

export interface SystemStats {
  total_images?: number;
  total_analyzed?: number;
  completed_runs?: number;
  failed_runs?: number;
  lenses_found?: number;
  avg_analysis_time_seconds?: number;
}

export interface ResultsStats {
  total_analyzed: number;
  total_runs: number;
  completed: number;
  failed: number;
  lenses_found: number;
  non_lenses: number;
}

export interface GalleryItem {
  run_id: number;
  image_id: number;
  euclid_id?: string;
  confidence: number;
  prediction: number;
  analysis_time_seconds?: number;
  timestamp?: string;
  heatmap_url?: string;
}

export interface GalleryResponse {
  count: number;
  results: GalleryItem[];
}

export interface TrainingRunItem {
  id: number;
  config_name?: string;
  model_name?: string;
  start_time?: string;
  end_time?: string;
  total_epochs?: number;
  best_epoch?: number;
  best_val_auc?: number;
  status?: string;
}

export interface TrainingHistoryResponse {
  count: number;
  runs: TrainingRunItem[];
}

export interface ConfigResponse {
  config: {
    inference: {
      stage_1_threshold: number;
      stage_2_threshold: number;
      stage_3_threshold: number;
      chunk_size_pixels: number;
      sub_chunk_size_pixels: number;
      overlap_percent: number;
    };
    model: {
      current_model: string;
      available_models: string[];
      device: string;
    };
    data: {
      cache_dir: string;
      max_cache_size_gb: number;
      cutout_size_arcmin: number;
    };
    export: {
      include_heatmaps: boolean;
      csv_fields?: string[];
      pdf_report_template?: string;
    };
  };
  version: string;
}

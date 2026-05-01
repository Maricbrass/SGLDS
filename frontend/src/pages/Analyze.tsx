import { useEffect, useMemo, useState } from "react";
import {
  getRun,
  getRunHistory,
  startAnalysis,
  toErrorMessage,
} from "../api/endpoints";
import type { AnalyzeHistoryItem, AnalyzeRunResponse } from "../api/types";
import HeatmapOverlay from "../components/HeatmapOverlay";
import RunStatusBadge from "../components/RunStatusBadge";

export default function AnalyzePage() {
  const [imageId, setImageId] = useState<string>("");
  const [runId, setRunId] = useState<number | null>(null);
  const [run, setRun] = useState<AnalyzeRunResponse | null>(null);
  const [history, setHistory] = useState<AnalyzeHistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const canPoll = useMemo(() => {
    if (!run) {
      return false;
    }
    return run.status === "queued" || run.status === "running" || run.status === "pending";
  }, [run]);

  async function refreshRun(currentRunId: number) {
    const data = await getRun(currentRunId);
    setRun(data);
    return data;
  }

  async function loadHistory(image: number) {
    const data = await getRunHistory(image, 8);
    setHistory(data.runs);
  }

  async function onStart() {
    const numericImageId = Number(imageId);
    if (!numericImageId) {
      setError("Image ID is required");
      return;
    }

    setError(null);
    setMessage(null);
    setLoading(true);

    try {
      const startResponse = await startAnalysis(numericImageId);
      setRunId(startResponse.run_id);
      setMessage(startResponse.cached ? "Used cached analysis" : "Analysis queued");
      const currentRun = await refreshRun(startResponse.run_id);
      if (currentRun.status === "completed") {
        await loadHistory(numericImageId);
      }
    } catch (err) {
      setError(toErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  async function onRefresh() {
    if (!runId) {
      return;
    }
    try {
      const currentRun = await refreshRun(runId);
      if (currentRun.status === "completed" && run?.image_id) {
        await loadHistory(run.image_id);
      }
    } catch (err) {
      setError(toErrorMessage(err));
    }
  }

  useEffect(() => {
    if (!runId || !canPoll) {
      return;
    }

    const interval = window.setInterval(() => {
      void refreshRun(runId).catch(() => {
        // Polling errors are shown on explicit refresh/start.
      });
    }, 2000);

    return () => window.clearInterval(interval);
  }, [runId, canPoll]);

  const confidence = run?.consensus_result?.final_confidence;
  const prediction = run?.consensus_result?.final_prediction;

  return (
    <section className="page-grid two-column">
      <div className="panel glass">
        <h2>Inference Studio</h2>
        <p className="panel-subtitle">Queue image analyses, monitor status, and inspect stage outputs.</p>

        <div className="form-stack">
          <label className="field">
            <span>Image ID</span>
            <input value={imageId} onChange={(e) => setImageId(e.target.value)} placeholder="123" />
          </label>

          <div className="button-row">
            <button className="primary-btn" type="button" onClick={onStart} disabled={loading}>
              {loading ? "Submitting..." : "Start Analysis"}
            </button>
            <button className="secondary-btn" type="button" onClick={onRefresh} disabled={!runId}>
              Refresh Run
            </button>
          </div>
        </div>

        {message ? <p className="inline-note success">{message}</p> : null}
        {error ? <p className="inline-note error">{error}</p> : null}

        {run ? (
          <div className="detail-list">
            <div><strong>Run ID:</strong> {run.id}</div>
            <div>
              <strong>Status:</strong> <RunStatusBadge status={run.status} />
            </div>
            <div><strong>Prediction:</strong> {prediction === 1 ? "Lens" : prediction === 0 ? "Non-lens" : "N/A"}</div>
            <div><strong>Confidence:</strong> {typeof confidence === "number" ? confidence.toFixed(4) : "N/A"}</div>
            <div><strong>Elapsed:</strong> {run.analysis_time_seconds ? `${run.analysis_time_seconds.toFixed(2)} sec` : "N/A"}</div>
            {run.error_message ? <div><strong>Error:</strong> {run.error_message}</div> : null}
          </div>
        ) : (
          <p className="empty-state">No run selected yet.</p>
        )}

        <h3>Stage Snapshot</h3>
        <pre className="json-preview">{JSON.stringify({
          stage_1: run?.stage_1_result,
          stage_2: run?.stage_2_results,
          stage_3: run?.stage_3_results,
        }, null, 2)}</pre>
      </div>

      <div className="panel">
        <h2>Visualization</h2>
        <p className="panel-subtitle">Heatmap is served from backend once a run completes.</p>

        <HeatmapOverlay
          heatmapUrl={run && run.status === "completed" && run.heatmap_path ? `/api/v1/analyze/runs/${run.id}/heatmap` : undefined}
        />

        <h3>Recent Runs For Image</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>Confidence</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr key={item.id}>
                  <td>{item.id}</td>
                  <td><RunStatusBadge status={item.status} /></td>
                  <td>{typeof item.consensus_result?.final_confidence === "number" ? item.consensus_result.final_confidence.toFixed(3) : "-"}</td>
                  <td>{item.run_timestamp ? new Date(item.run_timestamp).toLocaleString() : "-"}</td>
                </tr>
              ))}
              {history.length === 0 ? (
                <tr>
                  <td colSpan={4} className="table-empty">No history for this image yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

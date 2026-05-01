import { useEffect, useMemo, useState } from "react";
import { getResultsStats, getSystemStats, getTrainingHistory, toErrorMessage } from "../api/endpoints";
import type { ResultsStats, SystemStats, TrainingRunItem } from "../api/types";
import StatCard from "../components/StatCard";
import RunStatusBadge from "../components/RunStatusBadge";

export default function DashboardPage() {
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [resultsStats, setResultsStats] = useState<ResultsStats | null>(null);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRunItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [system, results, training] = await Promise.all([
          getSystemStats(),
          getResultsStats(),
          getTrainingHistory(6),
        ]);
        setSystemStats(system);
        setResultsStats(results);
        setTrainingRuns(training.runs);
      } catch (err) {
        setError(toErrorMessage(err));
      } finally {
        setLoading(false);
      }
    }

    void load();
  }, []);

  const avgRunTime = useMemo(() => {
    if (typeof systemStats?.avg_analysis_time_seconds !== "number") {
      return "N/A";
    }
    return `${systemStats.avg_analysis_time_seconds.toFixed(2)} sec`;
  }, [systemStats]);

  return (
    <section className="page-grid">
      <div className="panel glass">
        <h2>Mission Control</h2>
        <p className="panel-subtitle">Team-shared snapshot of ingestion, inference throughput, and model quality.</p>

        {loading ? <p className="inline-note">Loading dashboard metrics...</p> : null}
        {error ? <p className="inline-note error">{error}</p> : null}

        <div className="stats-grid">
          <StatCard label="Cached Images" value={systemStats?.total_images ?? "-"} hint="Cutouts available for analysis" />
          <StatCard label="Completed Runs" value={resultsStats?.completed ?? systemStats?.completed_runs ?? "-"} hint="Successful inference runs" />
          <StatCard label="Lenses Found" value={resultsStats?.lenses_found ?? systemStats?.lenses_found ?? "-"} hint="Predicted positives" />
          <StatCard label="Avg Analysis Time" value={avgRunTime} hint="End-to-end run duration" />
        </div>
      </div>

      <div className="panel">
        <h2>Recent Training Runs</h2>
        <p className="panel-subtitle">Latest model experiments and validation performance.</p>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Model</th>
                <th>Config</th>
                <th>Best AUC</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {trainingRuns.map((run) => (
                <tr key={run.id}>
                  <td>{run.id}</td>
                  <td>{run.model_name || "-"}</td>
                  <td>{run.config_name || "-"}</td>
                  <td>{typeof run.best_val_auc === "number" ? run.best_val_auc.toFixed(4) : "-"}</td>
                  <td><RunStatusBadge status={run.status || "pending"} /></td>
                </tr>
              ))}
              {trainingRuns.length === 0 ? (
                <tr>
                  <td colSpan={5} className="table-empty">No training runs recorded yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

import { useEffect, useState } from "react";
import StatCard from "../components/StatCard";
import { getTrainingHistory, getTrainingRunMetrics } from "../api/endpoints";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function TrainingPage() {
  const [runs, setRuns] = useState<any[]>([]);
  const [selectedRun, setSelectedRun] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [status, setStatus] = useState("Idle");
  
  useEffect(() => {
    getTrainingHistory(5).then((data) => {
      setRuns(data.runs || []);
      if (data.runs && data.runs.length > 0) {
        setSelectedRun(data.runs[0].id);
        setStatus(data.runs[0].status || "Completed");
      }
    });
  }, []);

  useEffect(() => {
    if (selectedRun) {
      getTrainingRunMetrics(selectedRun).then((data) => {
        // Transform metrics_history object to array for recharts
        const hist = data.metrics_history || {};
        const epochs = Object.keys(hist).map(Number).sort((a, b) => a - b);
        const chartData = epochs.map(epoch => ({
          epoch: epoch + 1,
          ...hist[epoch]
        }));
        setMetrics(chartData);
      }).catch(console.error);
    }
  }, [selectedRun]);

  const activeRun = runs.find(r => r.id === selectedRun) || {};

  return (
    <div className="page-grid">
      <div className="panel glass">
        <h2>Training Monitor</h2>
        <p className="panel-subtitle">Real-time metrics for Swin Transformer</p>

        <div style={{ marginTop: "1rem" }}>
          <select 
            value={selectedRun || ""} 
            onChange={(e) => setSelectedRun(Number(e.target.value))}
            style={{ maxWidth: "300px" }}
          >
            {runs.map(r => (
              <option key={r.id} value={r.id}>
                Run #{r.id} - {r.model_name || "Model"}
              </option>
            ))}
          </select>
        </div>
        
        <div className="stats-grid" style={{ marginTop: "2rem" }}>
          <StatCard label="Status" value={activeRun.status || "Idle"} />
          <StatCard label="Total Epochs" value={activeRun.total_epochs || "0"} />
          <StatCard label="Best Epoch" value={activeRun.best_epoch !== null ? String(activeRun.best_epoch + 1) : "N/A"} />
          <StatCard label="Best Val AUC" value={activeRun.best_val_auc ? activeRun.best_val_auc.toFixed(3) : "N/A"} />
        </div>

        <div style={{ marginTop: "2rem", height: "400px", background: "rgba(0,0,0,0.2)", borderRadius: "12px", padding: "1rem" }}>
          {metrics.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.5)" />
                <YAxis yAxisId="left" stroke="rgba(0, 229, 255, 0.8)" />
                <YAxis yAxisId="right" orientation="right" stroke="rgba(178, 0, 255, 0.8)" />
                <Tooltip 
                  contentStyle={{ backgroundColor: "rgba(10, 15, 30, 0.9)", border: "1px solid rgba(0, 229, 255, 0.5)", borderRadius: "8px" }}
                  itemStyle={{ color: "#fff" }}
                />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="train_loss" stroke="#00e5ff" name="Train Loss" strokeWidth={2} dot={false} />
                <Line yAxisId="left" type="monotone" dataKey="val_loss" stroke="#ff00e5" name="Val Loss" strokeWidth={2} dot={false} />
                <Line yAxisId="right" type="monotone" dataKey="val_auc" stroke="#00ffb2" name="Val AUC" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <p style={{ color: "var(--text-muted)" }}>No metrics data available for this run</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import { getTrainingHistory, getEvaluation } from "../api/endpoints";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function EvaluationPage() {
  const [runs, setRuns] = useState<any[]>([]);
  const [selectedRun, setSelectedRun] = useState<number | null>(null);
  const [evalData, setEvalData] = useState<any>(null);

  useEffect(() => {
    getTrainingHistory(5).then((data) => {
      setRuns(data.runs || []);
      if (data.runs && data.runs.length > 0) {
        setSelectedRun(data.runs[0].id);
      }
    });
  }, []);

  useEffect(() => {
    if (selectedRun) {
      getEvaluation(selectedRun).then((data) => {
        if (data.evaluations && data.evaluations.length > 0) {
          setEvalData(data.evaluations[0]);
        } else {
          setEvalData(null);
        }
      }).catch(() => setEvalData(null));
    }
  }, [selectedRun]);

  const activeRun = runs.find(r => r.id === selectedRun) || {};

  // Parse confusion matrix (expected format [[TN, FP], [FN, TP]])
  let cm = [[0, 0], [0, 0]];
  if (evalData && evalData.confusion_matrix) {
    cm = evalData.confusion_matrix;
  }
  const [tn, fp] = cm[0];
  const [fn, tp] = cm[1];

  // Format ROC data for Recharts
  const rocData = useMemo(() => {
    if (!evalData || !evalData.metrics_json?.roc_curve) return [];
    const { fpr, tpr } = evalData.metrics_json.roc_curve;
    return fpr.map((f: number, i: number) => ({
      fpr: f,
      tpr: tpr[i],
    }));
  }, [evalData]);

  return (
    <div className="page-grid two-column">
      <div className="panel glass">
        <h2>Evaluation Reports</h2>
        <p className="panel-subtitle">ROC curves, Confusion Matrices, and Metrics</p>
        
        <div style={{ marginTop: "1rem" }}>
          <select 
            value={selectedRun || ""} 
            onChange={(e) => setSelectedRun(Number(e.target.value))}
            style={{ maxWidth: "300px" }}
          >
            {runs.map(r => (
              <option key={r.id} value={r.id}>
                Run #{r.id} - {r.model_name || "Model"} (Best AUC: {r.best_val_auc ? r.best_val_auc.toFixed(3) : "N/A"})
              </option>
            ))}
          </select>
        </div>

        <div style={{ marginTop: "2rem", height: "350px", background: "rgba(0,0,0,0.3)", borderRadius: "16px", padding: "1.5rem", border: "1px solid var(--line)" }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rocData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
              <XAxis 
                dataKey="fpr" 
                label={{ value: "False Positive Rate", position: "insideBottom", offset: -5, fill: "var(--text-muted)", fontSize: 12 }} 
                tick={{ fill: "var(--text-muted)", fontSize: 10 }}
                stroke="rgba(255,255,255,0.2)"
              />
              <YAxis 
                label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }} 
                tick={{ fill: "var(--text-muted)", fontSize: 10 }}
                stroke="rgba(255,255,255,0.2)"
              />
              <Tooltip 
                contentStyle={{ background: "#0a0f1e", border: "1px solid var(--accent)", borderRadius: "8px" }}
                itemStyle={{ color: "var(--accent)" }}
                labelFormatter={(val) => `FPR: ${val.toFixed(3)}`}
              />
              <Line 
                type="monotone" 
                dataKey="tpr" 
                stroke="var(--accent)" 
                strokeWidth={3} 
                dot={{ r: 4, fill: "var(--accent)" }} 
                activeDot={{ r: 6, stroke: "#fff", strokeWidth: 2 }}
                name="ROC Curve"
              />
              <Line 
                data={[{fpr: 0, tpr: 0}, {fpr: 1, tpr: 1}]} 
                dataKey="tpr" 
                stroke="rgba(255,255,255,0.2)" 
                strokeDasharray="5 5" 
                dot={false}
                activeDot={false}
                name="Random"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel glass">
        <h3>Metrics Overview</h3>
        <div className="detail-list" style={{ marginTop: "1rem" }}>
          <div className="metric-row">
            <span>ROC AUC</span>
            <strong>{evalData && evalData.roc_auc ? evalData.roc_auc.toFixed(4) : "N/A"}</strong>
          </div>
          <div className="metric-row">
            <span>Precision</span>
            <strong>{evalData && evalData.precision ? (evalData.precision * 100).toFixed(1) + "%" : "N/A"}</strong>
          </div>
          <div className="metric-row">
            <span>Recall / TPR</span>
            <strong>{evalData && evalData.recall ? (evalData.recall * 100).toFixed(1) + "%" : "N/A"}</strong>
          </div>
          <div className="metric-row">
            <span>F1 Score</span>
            <strong>{evalData && evalData.f1_score ? evalData.f1_score.toFixed(3) : "N/A"}</strong>
          </div>
        </div>

        <h3 style={{ marginTop: "2rem" }}>Confusion Matrix</h3>
        {!evalData ? (
          <p className="empty-state" style={{ marginTop: "1rem" }}>No evaluation metrics found for this run. Please ensure evaluation has been performed.</p>
        ) : (
          <div style={{ marginTop: "1rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
              <div style={{ background: "rgba(255, 255, 255, 0.05)", padding: "1rem", borderRadius: "12px", textAlign: "center" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>True Negatives</span>
                <div style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#00ffb2" }}>{tn}</div>
              </div>
              <div style={{ background: "rgba(255, 255, 255, 0.05)", padding: "1rem", borderRadius: "12px", textAlign: "center" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>False Positives</span>
                <div style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#ff00e5" }}>{fp}</div>
              </div>
              <div style={{ background: "rgba(255, 255, 255, 0.05)", padding: "1rem", borderRadius: "12px", textAlign: "center" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>False Negatives</span>
                <div style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#ff00e5" }}>{fn}</div>
              </div>
              <div style={{ background: "rgba(255, 255, 255, 0.05)", padding: "1rem", borderRadius: "12px", textAlign: "center" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>True Positives</span>
                <div style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#00e5ff" }}>{tp}</div>
              </div>
          </div>
        )}
      </div>
    </div>
  );
}

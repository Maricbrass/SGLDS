import { useEffect, useState } from "react";
import { getGallery, toErrorMessage } from "../api/endpoints";
import type { GalleryItem } from "../api/types";

export default function GalleryPage() {
  const [items, setItems] = useState<GalleryItem[]>([]);
  const [minConfidence, setMinConfidence] = useState("0.7");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadGallery() {
    setLoading(true);
    setError(null);
    try {
      const data = await getGallery({ min_confidence: Number(minConfidence), limit: 60, skip: 0 });
      setItems(data.results);
    } catch (err) {
      setError(toErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadGallery();
  }, []);

  return (
    <section className="page-grid">
      <div className="panel glass">
        <h2>Results Gallery</h2>
        <p className="panel-subtitle">Filter completed analyses by confidence and inspect generated heatmaps.</p>

        <div className="button-row align-bottom">
          <label className="field compact">
            <span>Min confidence</span>
            <input value={minConfidence} onChange={(e) => setMinConfidence(e.target.value)} />
          </label>
          <button className="primary-btn" type="button" onClick={loadGallery} disabled={loading}>
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>

        {error ? <p className="inline-note error">{error}</p> : null}
      </div>

      <div className="gallery-grid">
        {items.map((item) => (
          <article className="gallery-card" key={item.run_id}>
            <header>
              <h3>{item.euclid_id || `Image ${item.image_id}`}</h3>
              <span className={item.prediction === 1 ? "pill positive" : "pill negative"}>
                {item.prediction === 1 ? "Lens" : "Non-lens"}
              </span>
            </header>

            <div className="metric-row">
              <span>Confidence</span>
              <strong>{item.confidence.toFixed(3)}</strong>
            </div>
            <div className="metric-row">
              <span>Run ID</span>
              <strong>{item.run_id}</strong>
            </div>

            <div className="gallery-actions">
              <a className="secondary-btn as-link" href={`/api/v1/analyze/runs/${item.run_id}/heatmap`} target="_blank" rel="noreferrer">
                Open Heatmap
              </a>
              <a className="ghost-link" href={`/api/v1/results/report/${item.run_id}`} target="_blank" rel="noreferrer">
                View Report JSON
              </a>
            </div>
          </article>
        ))}
        {items.length === 0 && !loading ? <p className="empty-state">No results match this filter yet.</p> : null}
      </div>
    </section>
  );
}

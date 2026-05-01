import { useEffect, useState } from "react";
import { getConfig, toErrorMessage, updateConfig } from "../api/endpoints";
import type { ConfigResponse } from "../api/types";

export default function SettingsPage() {
  const [config, setConfig] = useState<ConfigResponse["config"] | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const response = await getConfig();
        setConfig(response.config);
      } catch (err) {
        setError(toErrorMessage(err));
      }
    }
    void load();
  }, []);

  function updateInferenceField<K extends keyof ConfigResponse["config"]["inference"]>(
    key: K,
    value: number,
  ) {
    setConfig((current) => {
      if (!current) {
        return current;
      }
      return {
        ...current,
        inference: {
          ...current.inference,
          [key]: value,
        },
      };
    });
  }

  async function onSave() {
    if (!config) {
      return;
    }

    setSaving(true);
    setError(null);
    setMessage(null);

    try {
      const response = await updateConfig({
        inference: config.inference,
        model: { current_model: config.model.current_model },
      });
      setConfig(response.config);
      setMessage("Configuration saved");
    } catch (err) {
      setError(toErrorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  if (!config) {
    return (
      <section className="page-grid">
        <div className="panel glass">
          <h2>Settings</h2>
          <p className="inline-note">Loading config...</p>
          {error ? <p className="inline-note error">{error}</p> : null}
        </div>
      </section>
    );
  }

  return (
    <section className="page-grid two-column">
      <div className="panel glass">
        <h2>Inference Parameters</h2>
        <p className="panel-subtitle">Tune stage thresholds and chunking settings used by the pipeline.</p>

        <div className="form-stack">
          <label className="field">
            <span>Stage 1 Threshold</span>
            <input
              type="number"
              step="0.01"
              value={config.inference.stage_1_threshold}
              onChange={(e) => updateInferenceField("stage_1_threshold", Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>Stage 2 Threshold</span>
            <input
              type="number"
              step="0.01"
              value={config.inference.stage_2_threshold}
              onChange={(e) => updateInferenceField("stage_2_threshold", Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>Stage 3 Threshold</span>
            <input
              type="number"
              step="0.01"
              value={config.inference.stage_3_threshold}
              onChange={(e) => updateInferenceField("stage_3_threshold", Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>Chunk Size (pixels)</span>
            <input
              type="number"
              value={config.inference.chunk_size_pixels}
              onChange={(e) => updateInferenceField("chunk_size_pixels", Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>Sub-chunk Size (pixels)</span>
            <input
              type="number"
              value={config.inference.sub_chunk_size_pixels}
              onChange={(e) => updateInferenceField("sub_chunk_size_pixels", Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>Overlap (%)</span>
            <input
              type="number"
              value={config.inference.overlap_percent}
              onChange={(e) => updateInferenceField("overlap_percent", Number(e.target.value))}
            />
          </label>
        </div>
      </div>

      <div className="panel">
        <h2>Model and Data</h2>
        <p className="panel-subtitle">Current model selection and data cache profile.</p>

        <div className="detail-list">
          <div><strong>Current model:</strong> {config.model.current_model}</div>
          <div><strong>Available:</strong> {config.model.available_models.join(", ")}</div>
          <div><strong>Device:</strong> {config.model.device}</div>
          <div><strong>Cache directory:</strong> {config.data.cache_dir}</div>
          <div><strong>Max cache size:</strong> {config.data.max_cache_size_gb} GB</div>
          <div><strong>Default cutout size:</strong> {config.data.cutout_size_arcmin} arcmin</div>
        </div>

        <div className="button-row">
          <button className="primary-btn" type="button" onClick={onSave} disabled={saving}>
            {saving ? "Saving..." : "Save Changes"}
          </button>
        </div>

        {message ? <p className="inline-note success">{message}</p> : null}
        {error ? <p className="inline-note error">{error}</p> : null}
      </div>
    </section>
  );
}

import { type FormEvent, useEffect, useState } from "react";
import {
  fetchEuclidCutout,
  listCachedImages,
  searchEuclid,
  toErrorMessage,
} from "../api/endpoints";
import type { EuclidImage, EuclidSearchResult } from "../api/types";

type SearchMode = "target" | "coords";

export default function SearchPage() {
  const [mode, setMode] = useState<SearchMode>("target");
  const [targetName, setTargetName] = useState("");
  const [ra, setRa] = useState("273.0173");
  const [dec, setDec] = useState("68.1076");
  const [radius, setRadius] = useState("10");
  const [results, setResults] = useState<EuclidSearchResult[]>([]);
  const [cachedImages, setCachedImages] = useState<EuclidImage[]>([]);
  const [loading, setLoading] = useState(false);
  const [fetchingS3, setFetchingS3] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function loadCachedImages() {
    try {
      const data = await listCachedImages(12);
      setCachedImages(data.images);
    } catch (err) {
      setError(toErrorMessage(err));
    }
  }

  useEffect(() => {
    void loadCachedImages();
  }, []);

  async function onSearch(event?: FormEvent) {
    event?.preventDefault();
    setLoading(true);
    setError(null);
    setMessage(null);

    try {
      if (mode === "target") {
        const data = await searchEuclid({ target_name: targetName.trim(), radius_arcsec: Number(radius) });
        setResults(data.results);
        setMessage(`Found ${data.count} entries`);
        return;
      }

      const data = await searchEuclid({
        ra: Number(ra),
        dec: Number(dec),
        radius_arcsec: Number(radius),
      });
      setResults(data.results);
      setMessage(`Found ${data.count} entries`);
    } catch (err) {
      setError(toErrorMessage(err));
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  async function onFetchCutout(item: EuclidSearchResult) {
    setFetchingS3(item.s3_url);
    setError(null);
    setMessage(null);
    try {
      const response = await fetchEuclidCutout({
        s3_url: item.s3_url,
        euclid_id: item.euclid_id,
        cutout_size_arcmin: 1.0,
      });
      setMessage(`Fetched cutout as image ${response.image_id}`);
      await loadCachedImages();
    } catch (err) {
      setError(toErrorMessage(err));
    } finally {
      setFetchingS3(null);
    }
  }

  return (
    <section className="page-grid two-column">
      <div className="panel glass">
        <h2>Euclid Search</h2>
        <p className="panel-subtitle">Find targets by object name or sky coordinates, then fetch cutouts into cache.</p>

        <div className="segmented-control" role="tablist" aria-label="Search mode">
          <button
            type="button"
            className={mode === "target" ? "is-active" : ""}
            onClick={() => setMode("target")}
          >
            Target Name
          </button>
          <button
            type="button"
            className={mode === "coords" ? "is-active" : ""}
            onClick={() => setMode("coords")}
          >
            RA / DEC
          </button>
        </div>

        <form onSubmit={onSearch} className="form-stack">
          {mode === "target" ? (
            <label className="field">
              <span>Target</span>
              <input
                value={targetName}
                onChange={(e) => setTargetName(e.target.value)}
                placeholder="TYC 4429-1677-1"
              />
            </label>
          ) : (
            <div className="field-row">
              <label className="field">
                <span>RA</span>
                <input value={ra} onChange={(e) => setRa(e.target.value)} placeholder="273.0173" />
              </label>
              <label className="field">
                <span>DEC</span>
                <input value={dec} onChange={(e) => setDec(e.target.value)} placeholder="68.1076" />
              </label>
            </div>
          )}

          <label className="field">
            <span>Radius (arcsec)</span>
            <input value={radius} onChange={(e) => setRadius(e.target.value)} />
          </label>

          <button className="primary-btn" type="submit" disabled={loading}>
            {loading ? "Searching..." : "Search Euclid"}
          </button>
        </form>

        {message ? <p className="inline-note success">{message}</p> : null}
        {error ? <p className="inline-note error">{error}</p> : null}

        <div className="result-grid">
          {results.map((item) => (
            <article className="result-card" key={item.s3_url}>
              <h3>{item.euclid_id || item.filename || "Euclid image"}</h3>
              <p>{item.instrument || "N/A"} | {item.filter || "N/A"}</p>
              <code className="line-clamp">{item.s3_url}</code>
              <button
                className="secondary-btn"
                type="button"
                onClick={() => onFetchCutout(item)}
                disabled={fetchingS3 === item.s3_url}
              >
                {fetchingS3 === item.s3_url ? "Fetching..." : "Fetch Cutout"}
              </button>
            </article>
          ))}
          {results.length === 0 ? <p className="empty-state">No search results yet.</p> : null}
        </div>
      </div>

      <div className="panel">
        <h2>Cached Images</h2>
        <p className="panel-subtitle">Recently fetched cutouts stored on the backend.</p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Euclid ID</th>
                <th>Source</th>
                <th>Fetched</th>
              </tr>
            </thead>
            <tbody>
              {cachedImages.map((image) => (
                <tr key={image.id}>
                  <td>{image.id}</td>
                  <td>{image.euclid_id || "-"}</td>
                  <td>{image.source || "euclid"}</td>
                  <td>{image.fetch_date ? new Date(image.fetch_date).toLocaleString() : "-"}</td>
                </tr>
              ))}
              {cachedImages.length === 0 ? (
                <tr>
                  <td colSpan={4} className="table-empty">No cached images yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

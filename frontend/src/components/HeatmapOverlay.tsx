import { useState } from "react";

type Props = {
  imageUrl?: string;
  heatmapUrl?: string;
};

export default function HeatmapOverlay({ imageUrl, heatmapUrl }: Props) {
  const [heatmapLoaded, setHeatmapLoaded] = useState(false);
  const [heatmapError, setHeatmapError] = useState(false);

  return (
    <div className="heatmap-stage">
      {imageUrl ? <img src={imageUrl} alt="science frame" className="base-layer" /> : null}
      {heatmapUrl && !heatmapError ? (
        <img
          src={heatmapUrl}
          alt="inference heatmap"
          className="heatmap-layer"
          onLoad={() => setHeatmapLoaded(true)}
          onError={() => setHeatmapError(true)}
        />
      ) : null}

      {!imageUrl && !heatmapUrl ? <div className="overlay-empty">Run analysis to view heatmap output.</div> : null}
      {heatmapUrl && !heatmapLoaded && !heatmapError ? <div className="overlay-empty">Loading heatmap...</div> : null}
      {heatmapError ? <div className="overlay-empty">Heatmap is not available yet.</div> : null}
    </div>
  );
}

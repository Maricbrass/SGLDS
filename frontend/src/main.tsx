import { createRoot } from "react-dom/client";
import { useEffect, useState } from "react";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import AnalyzePage from "./pages/Analyze";
import DashboardPage from "./pages/Dashboard";
import GalleryPage from "./pages/Gallery";
import SearchPage from "./pages/Search";
import SettingsPage from "./pages/Settings";
import "./styles.css";

function App() {
  const [dark, setDark] = useState<boolean>(() => {
    try {
      return localStorage.getItem("sglds:theme") === "dark";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("sglds:theme", dark ? "dark" : "light");
    } catch {}
    if (dark) document.body.classList.add("theme-dark", "starry-bg");
    else document.body.classList.remove("theme-dark", "starry-bg");
  }, [dark]);

  return (
    <BrowserRouter>
      <div className="app-shell">
        <header className="app-header">
          <div>
            <h1>SGLDS Team Dashboard</h1>
            <p>Strong Gravitational Lens Detection in Euclid Q1</p>
          </div>
          <div style={{ display: "flex", gap: "0.6rem", alignItems: "center" }}>
            <button
              className="pill"
              onClick={() => setDark((d) => !d)}
              aria-pressed={dark}
              title={dark ? "Switch to light theme" : "Switch to dark theme"}
            >
              {dark ? "🌟 Dark" : "✨ Light"}
            </button>
            <span className="pill neutral">Phase 1B+</span>
          </div>
        </header>

        <nav className="top-nav" aria-label="Primary">
          <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>Dashboard</NavLink>
          <NavLink to="/search" className={({ isActive }) => (isActive ? "active" : "")}>Search</NavLink>
          <NavLink to="/analyze" className={({ isActive }) => (isActive ? "active" : "")}>Analyze</NavLink>
          <NavLink to="/gallery" className={({ isActive }) => (isActive ? "active" : "")}>Gallery</NavLink>
          <NavLink to="/settings" className={({ isActive }) => (isActive ? "active" : "")}>Settings</NavLink>
        </nav>

        <main className="app-content">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/gallery" element={<GalleryPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

createRoot(document.getElementById("root")!).render(<App />);

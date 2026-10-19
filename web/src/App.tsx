import { useCallback, useEffect, useState } from "react";
import { api, Project } from "./api";
import ImportScreen from "./screens/ImportScreen";
import CalibrateScreen from "./screens/CalibrateScreen";
import ProcessScreen from "./screens/ProcessScreen";
import ResultsScreen from "./screens/ResultsScreen";
import JobsIndicator from "./components/JobsIndicator";

export type Screen = "import" | "calibrate" | "process" | "results";
const SCREENS: { key: Screen; label: string }[] = [
  { key: "import", label: "Import" },
  { key: "calibrate", label: "Calibrate" },
  { key: "process", label: "Process" },
  { key: "results", label: "Results" },
];

/** Which steps already have their output, so the nav reads as a progression. */
function stepDone(p: Project | null, s: Screen): boolean {
  if (!p) return false;
  const c = p.config.calibration;
  switch (s) {
    case "import": return true;
    case "calibrate": return !!(((c.paths ?? []).some((pp) => pp.points.length >= 2) || (c.line_a && c.line_b)) && !p.calibration_result.error && p.calibration_result.fit_ok !== false);
    case "process": return !!p.status.results_json;
    case "results": return !!p.status.render_mp4;
  }
}

const BrandMark = () => (
  <span className="brand-mark" aria-hidden="true">
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round">
      <path d="M3 2.5v9M11 2.5v9" strokeOpacity=".55" />
      <path d="M1.5 7h8.5M8 4.8 10.3 7 8 9.2" />
    </svg>
  </span>
);

function parseHash(): { id: string | null; screen: Screen } {
  const m = window.location.hash.match(/^#\/p\/([a-z0-9-]+)(?:\/(\w+))?/);
  if (!m) return { id: null, screen: "import" };
  const s = (m[2] as Screen) || "import";
  return { id: m[1], screen: SCREENS.some((x) => x.key === s) ? s : "import" };
}

export default function App() {
  const [route, setRoute] = useState(parseHash);
  const [project, setProject] = useState<Project | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const on = () => setRoute(parseHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  const reload = useCallback(async () => {
    if (!route.id) return setProject(null);
    try {
      setProject(await api.project(route.id));
      setError(null);
    } catch (e) {
      setError(String(e));
      setProject(null);
    }
  }, [route.id]);

  useEffect(() => { reload(); }, [reload]);

  const go = (screen: Screen, id = route.id) => {
    window.location.hash = id ? `#/p/${id}/${screen}` : "#/";
  };

  const p = project && project.id === route.id ? project : null;
  return (
    <div className="app">
      <header className="topbar">
        <button className="brand" onClick={() => (window.location.hash = "#/")} title="Project library"><BrandMark />speedcam</button>
        <nav aria-label="Workflow steps">
          <ol className="steps">
            {SCREENS.map((s, i) => {
              const active = route.screen === s.key, done = stepDone(p, s.key);
              return (
                <li key={s.key} className={done ? "done" : ""}>
                  <button className={`step${active ? " active" : ""}${done ? " done" : ""}`} aria-current={active ? "step" : undefined}
                    disabled={!p && s.key !== "import"} onClick={() => go(s.key)}>
                    <span className="step-num">{done && !active ? "✓" : i + 1}</span>{s.label}
                  </button>
                </li>
              );
            })}
          </ol>
        </nav>
        <div className="project-name" title={p?.name}>{p ? p.name : ""}</div>
        <JobsIndicator />
      </header>
      {error && <div className="banner error">{error}</div>}
      <main>
        {route.screen === "import" && (
          <ImportScreen project={p} onProject={(np) => { setProject(np); go("import", np.id); }}
            onChanged={setProject} onNext={() => go("calibrate")} />
        )}
        {p && route.screen === "calibrate" && <CalibrateScreen project={p} onChanged={setProject} onNext={() => go("process")} />}
        {p && route.screen === "process" && <ProcessScreen project={p} onChanged={setProject} onDone={() => go("results")} />}
        {p && route.screen === "results" && <ResultsScreen project={p} onChanged={setProject} />}
      </main>
    </div>
  );
}

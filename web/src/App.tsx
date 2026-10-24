import { useCallback, useEffect, useState } from "react";
import { api, Project } from "./api";
import ImportScreen from "./screens/ImportScreen";
import SetupScreen from "./screens/SetupScreen";
import AnalyzeScreen from "./screens/AnalyzeScreen";
import ReviewScreen from "./screens/ReviewScreen";
import ExportScreen from "./screens/ExportScreen";
import JobsIndicator from "./components/JobsIndicator";
import JobsPage from "./screens/JobsPage";
import Tutorial, { tutorialPending } from "./components/Tutorial";
import { Screen, SCREENS, screenOf } from "./steps";

/** Which steps already have their output, so the nav reads as a progression. */
function stepDone(p: Project | null, s: Screen): boolean {
  if (!p) return false;
  const c = p.config.calibration;
  switch (s) {
    case "import": return true;
    case "setup": {
      const rules = (c.red_lights ?? []).some((r) => r.stop_lines.some((l) => l.points.length >= 2)) || (c.solid_lines ?? []).some((l) => l.points.length >= 2);
      return !!(((c.paths ?? []).some((pp) => pp.points.length >= 2) || (c.line_a && c.line_b) || rules) && !p.calibration_result.error && p.calibration_result.fit_ok !== false);
    }
    case "analyze": return !!p.status.results_json;
    case "review": return !!p.status.results_json;
    case "export": return !!p.status.render_mp4;
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

function parseHash(): { id: string | null; screen: Screen; jobs?: boolean } {
  if (window.location.hash.startsWith("#/jobs")) return { id: null, screen: "import", jobs: true };
  const m = window.location.hash.match(/^#\/p\/([a-z0-9-]+)(?:\/(\w+))?/);
  if (!m) return { id: null, screen: "import" };
  return { id: m[1], screen: screenOf(m[2]) };
}

export default function App() {
  const [route, setRoute] = useState(parseHash);
  const [project, setProject] = useState<Project | null>(null);
  const [tutorial, setTutorial] = useState(tutorialPending);  // opens by itself the first time
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
              const active = !route.jobs && route.screen === s.key, done = stepDone(p, s.key);
              return (
                <li key={s.key} className={done ? "done" : ""}>
                  <button className={`step${active ? " active" : ""}${done ? " done" : ""}`} aria-current={active ? "step" : undefined}
                    disabled={(!p && s.key !== "import") || (s.key === "export" && !!p && !p.status.results_json)}
                    title={s.key === "export" && p && !p.status.results_json ? "Analyse the clip first" : s.title} onClick={() => go(s.key)}>
                    <span className="step-num">{done && !active ? "✓" : i + 1}</span><span className="step-label">{s.label}</span>
                  </button>
                </li>
              );
            })}
          </ol>
        </nav>
        <div className="project-name" title={p?.name}>{p ? p.name : ""}</div>
        <a className={`jobs-link${route.jobs ? " active" : ""}`} href="#/jobs" title="Every job, with the files each one produced">Jobs</a>
        <button className="tutorial-link" onClick={() => setTutorial(true)} title="A short walk through the steps">Tutorial</button>
        <JobsIndicator />
      </header>
      {error && <div className="banner error">{error}</div>}
      <main>
        {route.jobs && <JobsPage />}
        {!route.jobs && route.screen === "import" && (
          <ImportScreen project={p} onProject={(np) => { setProject(np); go("import", np.id); }}
            onChanged={setProject} onNext={() => go("setup")} />
        )}
        {p && route.screen === "setup" && <SetupScreen project={p} onChanged={setProject} onNext={() => go("analyze")} />}
        {p && route.screen === "analyze" && <AnalyzeScreen project={p} onChanged={setProject} onDone={() => go("review")} onSetup={() => go("setup")} />}
        {p && route.screen === "review" && <ReviewScreen project={p} onChanged={setProject} onExport={() => go("export")} onSetup={() => go("setup")} />}
        {p && route.screen === "export" && <ExportScreen project={p} onChanged={setProject} />}
      </main>
      <Tutorial open={tutorial} onClose={() => setTutorial(false)} />
    </div>
  );
}

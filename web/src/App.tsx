import { useCallback, useEffect, useState } from "react";
import { api, Project } from "./api";
import ImportScreen from "./screens/ImportScreen";
import CalibrateScreen from "./screens/CalibrateScreen";
import ProcessScreen from "./screens/ProcessScreen";
import ResultsScreen from "./screens/ResultsScreen";
import JobsIndicator from "./components/JobsIndicator";

export type Screen = "import" | "calibrate" | "process" | "results";
const SCREENS: { key: Screen; label: string }[] = [
  { key: "import", label: "1 · Import" },
  { key: "calibrate", label: "2 · Calibrate" },
  { key: "process", label: "3 · Process" },
  { key: "results", label: "4 · Results" },
];

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
        <div className="brand" onClick={() => (window.location.hash = "#/")}>speedcam</div>
        <nav>
          {SCREENS.map((s) => (
            <button key={s.key} className={route.screen === s.key ? "tab active" : "tab"}
              disabled={!p && s.key !== "import"} onClick={() => go(s.key)}>{s.label}</button>
          ))}
        </nav>
        <div className="project-name">{p ? p.name : ""}</div>
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

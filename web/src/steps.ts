/** The workflow steps, in order. */
export type Screen = "import" | "setup" | "analyze" | "review" | "export";
export const SCREENS: { key: Screen; label: string; title: string }[] = [
  { key: "import", label: "Import", title: "The video, its real frame rate and the browser proxy" },
  { key: "setup", label: "Setup", title: "Scale, where to measure, speed limit and tolerances, red lights and solid lines" },
  { key: "analyze", label: "Analyze", title: "Find the activity, choose what to analyse, run" },
  { key: "review", label: "Review", title: "Speeds, vehicles and violations; fix splits, name vehicles" },
  { key: "export", label: "Export", title: "The annotated video and the data" },
];
/** Step names before the five-step flow (links, the library's "resume"). */
export const SCREEN_ALIASES: Record<string, Screen> = { calibrate: "setup", process: "analyze", results: "review" };
export const screenOf = (s: string | null | undefined): Screen =>
  (SCREEN_ALIASES[s ?? ""] ?? (SCREENS.some((x) => x.key === s) ? s : "import")) as Screen;


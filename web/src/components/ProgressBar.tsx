export default function ProgressBar({ label, done, total, extra }: { label: string; done?: number; total?: number; extra?: string }) {
  const pct = total ? Math.min(100, (100 * (done ?? 0)) / total) : 0;
  return (
    <div className="progress">
      <div className="progress-label">
        <span>{label}</span>
        <span className="muted">{total ? `${done ?? 0} / ${total} · ${pct.toFixed(1)}%` : ""} {extra}</span>
      </div>
      <div className="progress-track"><div className="progress-fill" style={{ width: `${pct}%` }} /></div>
    </div>
  );
}

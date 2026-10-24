import { useEffect, useState } from "react";

/** A number setting saved when it's left (blur or Enter); empty = null where that's allowed. */
export function SettingNum({ value, onCommit, unit, step = 1, min, placeholder }: {
  value: number | null; onCommit: (v: number | null) => void; unit?: string; step?: number; min?: number; placeholder?: string;
}) {
  const [text, setText] = useState(value === null || value === undefined ? "" : String(value));
  useEffect(() => { setText(value === null || value === undefined ? "" : String(value)); }, [value]);
  const commit = () => {
    const raw = text.trim().replace(",", ".");
    const v = raw === "" ? null : Number(raw);
    if (v !== null && (!Number.isFinite(v) || (min !== undefined && v < min))) {
      setText(value === null || value === undefined ? "" : String(value));
      return;
    }
    if (v !== value && (v !== null || placeholder)) onCommit(v);
    else if (v === null) setText(value === null || value === undefined ? "" : String(value));
  };
  return (
    <span className="setting-num">
      <input type="text" inputMode="decimal" value={text} placeholder={placeholder} step={step}
        onChange={(e) => setText(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
      {unit && <span className="unit">{unit}</span>}
    </span>
  );
}

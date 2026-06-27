export function MiniEquityCurve({ className }: { className?: string }) {
  // a V-shaped recovery curve like the in-app preview
  const pts = [4, 18, 30, 22, 10, 24, 38, 30, 16, 6, 20, 34, 28, 40, 52, 46, 58]
  const max = 60
  const w = 220
  const h = 70
  const step = w / (pts.length - 1)
  const line = pts.map((p, i) => `${i * step},${h - (p / max) * h}`).join(" ")
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className={className} preserveAspectRatio="none" aria-hidden>
      <defs>
        <linearGradient id="eqfill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="white" stopOpacity="0.28" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${h} ${line} ${w},${h}`} fill="url(#eqfill)" />
      <polyline
        points={line}
        fill="none"
        stroke="white"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity="0.95"
      />
    </svg>
  )
}

export function GsScoreRadar({ className }: { className?: string }) {
  const cx = 60
  const cy = 56
  const r = 42
  const axes = ["Win %", "Profit", "Avg Win", "Discipline", "R:R"]
  const vals = [0.85, 0.6, 0.72, 0.9, 0.66]
  const angle = (i: number) => (Math.PI * 2 * i) / axes.length - Math.PI / 2
  const point = (i: number, scale: number) =>
    `${cx + Math.cos(angle(i)) * r * scale},${cy + Math.sin(angle(i)) * r * scale}`
  const ring = (scale: number) => axes.map((_, i) => point(i, scale)).join(" ")
  const shape = vals.map((v, i) => point(i, v)).join(" ")
  return (
    <svg viewBox="0 0 120 116" className={className} aria-hidden>
      {[0.35, 0.7, 1].map((s) => (
        <polygon
          key={s}
          points={ring(s)}
          fill="none"
          stroke="currentColor"
          strokeWidth="1"
          opacity="0.18"
        />
      ))}
      {axes.map((_, i) => (
        <line
          key={i}
          x1={cx}
          y1={cy}
          x2={point(i, 1).split(",")[0]}
          y2={point(i, 1).split(",")[1]}
          stroke="currentColor"
          strokeWidth="1"
          opacity="0.15"
        />
      ))}
      <polygon points={shape} fill="var(--brand-purple)" fillOpacity="0.35" stroke="var(--brand-purple)" strokeWidth="2" />
    </svg>
  )
}

/** Tiny donut used for distribution thumbnails */
export function MiniDonut({
  segments,
  className,
}: {
  segments: { value: number; color: string }[]
  className?: string
}) {
  const total = segments.reduce((s, x) => s + x.value, 0)
  const r = 28
  const c = 2 * Math.PI * r
  let offset = 0
  return (
    <svg viewBox="0 0 72 72" className={className} aria-hidden>
      <g transform="translate(36,36) rotate(-90)">
        {segments.map((seg, i) => {
          const len = (seg.value / total) * c
          const el = (
            <circle
              key={i}
              r={r}
              fill="none"
              stroke={seg.color}
              strokeWidth="11"
              strokeDasharray={`${len} ${c - len}`}
              strokeDashoffset={-offset}
            />
          )
          offset += len
          return el
        })}
      </g>
    </svg>
  )
}

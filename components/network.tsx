import { ArrowRight } from "lucide-react"
import { GsMark, Wordmark } from "@/components/wordmark"

const NODES = [
  { id: "AK", x: 50, y: 12, color: "#22d3ee" },
  { id: "LM", x: 24, y: 30, color: "#ec4899" },
  { id: "MR", x: 78, y: 28, color: "#a78bfa" },
  { id: "RC", x: 14, y: 58, color: "#3b82f6" },
  { id: "JL", x: 88, y: 56, color: "#d946ef" },
  { id: "PW", x: 30, y: 82, color: "#a855f7" },
  { id: "TS", x: 72, y: 84, color: "#6366f1" },
]

export function Network() {
  return (
    <section className="relative overflow-hidden py-20 lg:py-28">
      <div className="relative mx-auto max-w-7xl px-5 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="font-display text-3xl font-700 tracking-tight text-balance sm:text-4xl lg:text-5xl">
            Connect, collab, and improve
          </h2>
          <p className="mt-4 text-lg text-muted-foreground text-pretty">
            Build discipline with playbooks, share streaks and recaps, and grow with traders who
            journal the same way you do. Genesis is your hub, not another spreadsheet.
          </p>
          <a
            href="#start"
            className="bg-gradient-brand glow-brand mt-8 inline-flex items-center gap-2 rounded-xl px-6 py-3.5 text-base font-semibold text-white transition-transform hover:scale-[1.02]"
          >
            Get started today
            <ArrowRight className="h-4 w-4" />
          </a>
        </div>

        <div className="relative mx-auto mt-14 aspect-[16/10] max-w-3xl sm:aspect-[16/9]">
          {/* connection lines */}
          <svg className="absolute inset-0 h-full w-full" aria-hidden>
            {NODES.map((n) => (
              <line
                key={n.id}
                x1="50%"
                y1="50%"
                x2={`${n.x}%`}
                y2={`${n.y}%`}
                stroke="var(--brand-purple)"
                strokeOpacity="0.35"
                strokeWidth="1.5"
              />
            ))}
          </svg>

          {/* hub */}
          <div className="absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center">
            <div className="glow-brand flex h-28 w-28 items-center justify-center rounded-full border border-brand-purple/40 bg-gradient-to-br from-brand-purple/30 to-background backdrop-blur">
              <GsMark className="text-4xl" />
            </div>
            <Wordmark className="mt-3 text-xs uppercase tracking-[0.3em] text-muted-foreground" />
          </div>

          {/* nodes */}
          {NODES.map((n) => (
            <div
              key={n.id}
              className="absolute flex h-12 w-12 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full text-sm font-700 text-white shadow-lg ring-2 ring-background"
              style={{ left: `${n.x}%`, top: `${n.y}%`, background: n.color }}
            >
              {n.id}
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

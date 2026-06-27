import { Infinity as InfinityIcon, Sparkles, Diamond, Target, BarChart3 } from "lucide-react"
import { Wordmark } from "@/components/wordmark"

const STAT_CARDS = [
  { value: "100K+", label: "Trades analyzed", icon: BarChart3 },
  { value: "∞", label: "Playbooks", icon: InfinityIcon },
  { value: null, label: "Numerology & astrology", sub: "Linked to your trades", icon: Sparkles },
  { value: null, label: "Patterns & hedge recognition", icon: Diamond },
  { value: null, label: "Resolutions & recaps", icon: Target },
]

const NET_BARS = [40, 52, 44, 60, 50, 68, 62, 78, 72, 86]

export function WhyTraders() {
  return (
    <section id="analytics" className="relative overflow-hidden py-20 lg:py-28">
      <div
        className="pointer-events-none absolute left-1/2 top-0 h-80 w-[40rem] -translate-x-1/2 rounded-full bg-brand-purple/15 blur-[120px]"
        aria-hidden
      />
      <div className="relative mx-auto max-w-7xl px-5 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="font-display text-3xl font-700 tracking-tight text-balance sm:text-4xl lg:text-5xl">
            Why traders choose <Wordmark className="text-gradient-brand" />
          </h2>
          <p className="mt-4 text-lg text-muted-foreground text-pretty">
            Measure more than profit. Consistency compounds when you can see behavior, not just
            P&amp;L.
          </p>
        </div>

        <div className="mt-12 grid grid-cols-2 gap-4 md:grid-cols-5">
          {STAT_CARDS.map((c) => (
            <div
              key={c.label}
              className="rounded-2xl border border-border bg-card/60 p-6 text-center"
            >
              {c.value ? (
                <div className="font-display text-3xl font-700">{c.value}</div>
              ) : (
                <c.icon className="mx-auto h-7 w-7 text-brand-cyan" />
              )}
              <div className="mt-3 text-sm font-medium text-foreground/90">{c.label}</div>
              {c.sub && <div className="mt-1 text-xs text-muted-foreground">{c.sub}</div>}
            </div>
          ))}
        </div>

        <div className="mt-16 grid items-center gap-8 lg:grid-cols-2">
          <div>
            <h3 className="font-display text-3xl font-700 tracking-tight sm:text-4xl">
              Measure more than profit.
            </h3>
            <p className="mt-4 max-w-md text-lg text-muted-foreground text-pretty">
              Equity curve, win rate, profit factor, R:R distribution, sessions, and GS Score radar
              in one dashboard.
            </p>
          </div>

          <div className="rounded-2xl border border-border bg-card/60 p-6">
            <div className="text-sm text-muted-foreground">Net daily P&amp;L</div>
            <div
              className="mt-5 flex h-48 items-end gap-2.5"
              role="img"
              aria-label="Net daily profit and loss bar chart trending up"
            >
              {NET_BARS.map((h, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-t-md bg-gradient-to-t from-brand-cyan/70 to-brand-purple"
                  style={{ height: `${h}%` }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

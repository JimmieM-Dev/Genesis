import { Flame, Moon } from "lucide-react"

const PLAYBOOKS = [
  { name: "London open", meta: "Win rate · P&L · Linked account" },
  { name: "NY continuation", meta: "Win rate · P&L · Linked account" },
  { name: "Asia range", meta: "Win rate · P&L · Linked account" },
]

const WIN_STREAKS = [
  { label: "3 day win", value: "+$1,240" },
  { label: "2 day win", value: "+$890" },
  { label: "1 day win", value: "+$420" },
]

const SESSIONS = [
  { name: "Asia", pnl: "+$41,828", win: "76% win", pos: true },
  { name: "London", pnl: "-$132.76", win: "61% win", pos: false },
  { name: "Sydney", pnl: "-$14,993", win: "70% win", pos: false },
  { name: "New York", pnl: "-$372,173", win: "49% win", pos: false },
]

export function InsightSections() {
  return (
    <section className="relative mx-auto max-w-7xl space-y-20 px-5 py-20 lg:px-8 lg:py-24">
      {/* Consistency compounds */}
      <div className="grid items-center gap-10 lg:grid-cols-2">
        <div className="rounded-2xl border border-border bg-card/50 p-6">
          <div className="grid grid-cols-2 gap-3">
            {PLAYBOOKS.map((p, i) => (
              <div
                key={p.name}
                className={
                  "rounded-xl border border-border bg-background/60 p-4 " +
                  (i === 2 ? "col-span-2 sm:col-span-1" : "")
                }
              >
                <div className="font-600">{p.name}</div>
                <div className="mt-1 text-xs text-muted-foreground">{p.meta}</div>
              </div>
            ))}
          </div>
        </div>
        <div>
          <h3 className="font-display text-3xl font-700 tracking-tight sm:text-4xl">
            Consistency compounds.
          </h3>
          <p className="mt-4 max-w-md text-lg text-muted-foreground text-pretty">
            Playbooks tie to your imported accounts. Score rule adherence, time windows, and
            sessions on real trades.
          </p>
        </div>
      </div>

      {/* Streak intelligence */}
      <div className="grid items-center gap-10 lg:grid-cols-2">
        <div className="order-2 lg:order-1">
          <h3 className="font-display text-3xl font-700 tracking-tight sm:text-4xl">
            Streak intelligence.
          </h3>
          <p className="mt-4 max-w-md text-lg text-muted-foreground text-pretty">
            Day and trade streaks, shareable screenshots, and recaps that show when you are actually
            improving.
          </p>
        </div>
        <div className="order-1 space-y-2.5 rounded-2xl border border-border bg-card/50 p-6 lg:order-2">
          {WIN_STREAKS.map((s) => (
            <div
              key={s.label}
              className="flex items-center justify-between rounded-xl border border-pos/25 bg-pos/10 px-5 py-4"
            >
              <span className="flex items-center gap-2 font-medium text-pos">
                <Flame className="h-4 w-4" />
                {s.label}
              </span>
              <span className="font-700 text-pos">{s.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Performance by session */}
      <div className="rounded-2xl border border-border bg-card/50 p-6 lg:p-8">
        <h3 className="font-display text-xl font-600">Performance by trading session</h3>
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {SESSIONS.map((s) => (
            <div key={s.name} className="rounded-xl border border-border bg-background/60 p-5">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">{s.name}</div>
              <div className={"mt-2 text-2xl font-700 " + (s.pos ? "text-pos" : "text-neg")}>
                {s.pnl}
              </div>
              <div className="mt-3 text-xs text-muted-foreground">{s.win}</div>
            </div>
          ))}
        </div>
        <p className="mt-5 text-xs text-muted-foreground">
          Sessions are bucketed from the trade open time — brokers report in their own time-zone.
        </p>
      </div>

      {/* Lunar cycle */}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="flex items-center gap-4 rounded-2xl border border-pos/30 bg-pos/5 p-6">
          <Moon className="h-9 w-9 shrink-0 text-pos" />
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-pos">
              Best lunar cycle
            </div>
            <div className="mt-1 font-display text-xl font-700">Last Quarter</div>
            <div className="mt-1 text-sm text-muted-foreground">
              16 trades · 43.8% win rate · <span className="text-pos">$52,262</span> · PF 361.03
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4 rounded-2xl border border-brand-purple/40 bg-brand-purple/5 p-6">
          <Moon className="h-9 w-9 shrink-0 text-brand-purple" />
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-neg">
              Worst lunar cycle
            </div>
            <div className="mt-1 font-display text-xl font-700">Waxing Gibbous</div>
            <div className="mt-1 text-sm text-muted-foreground">
              110 trades · 43.6% win rate · <span className="text-neg">-$304,067</span> · PF 0.14
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

import { Check, Link2, Upload } from "lucide-react"
import { GsScoreRadar, MiniEquityCurve } from "@/components/mini-visuals"

const BROKER_BADGES = ["HFM", "IC", "XM", "FX", "JM", "CT", "PS", "MT5", "NT"]

const ACCOUNTS = [
  { name: "All Accounts", primary: true },
  { name: "Demo Data" },
  { name: "My Trades" },
  { name: "Funded — 10K", live: true },
  { name: "NinjaTrader" },
]

const STAT_TILES = [
  { label: "Win Rate", value: "60.96%", tone: "pos" },
  { label: "Day P/L", value: "1.24" },
  { label: "Balance", value: "$32,032" },
  { label: "Expectancy", value: "3.11" },
  { label: "Profit Factor", value: "1.86", tone: "pos" },
  { label: "Profit / Trade", value: "2.10" },
]

export function FeatureShowcase() {
  return (
    <section id="features" className="relative mx-auto max-w-7xl px-5 py-20 lg:px-8 lg:py-28">
      <div className="mx-auto max-w-2xl text-center">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-brand-cyan">
          The Genesis edge
        </p>
        <h2 className="mt-3 font-display text-3xl font-700 tracking-tight text-balance sm:text-4xl lg:text-5xl">
          Everything you import, working for you
        </h2>
        <p className="mt-4 text-lg text-muted-foreground text-pretty">
          Three engines power your journal — connect, consolidate, and let Genesis calculate the
          stats automatically on every trade.
        </p>
      </div>

      <div className="mt-14 grid gap-6 lg:grid-cols-3">
        {/* Card 1 — Automated Journaling */}
        <article className="group relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-brand-magenta/90 via-brand-purple/85 to-brand-purple p-7 text-white shadow-2xl">
          <div className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full bg-white/15 blur-3xl" aria-hidden />
          <h3 className="relative font-display text-2xl font-700">Automated Journaling</h3>
          <p className="relative mt-3 text-sm leading-relaxed text-white/85">
            Broker sync, file upload, or manual trade adds — everything flows into one journal
            automatically.
          </p>

          <div className="relative mt-7 grid grid-cols-3 gap-2.5">
            {BROKER_BADGES.map((b) => (
              <div
                key={b}
                className="flex aspect-square items-center justify-center rounded-2xl border border-white/15 bg-white/10 text-sm font-700 backdrop-blur"
              >
                {b}
              </div>
            ))}
          </div>

          <div className="relative mt-5 grid grid-cols-2 gap-2.5">
            <button className="inline-flex items-center justify-center gap-2 rounded-xl bg-white/95 px-4 py-3 text-sm font-semibold text-brand-purple">
              <Link2 className="h-4 w-4" /> Connect
            </button>
            <button className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/25 bg-white/10 px-4 py-3 text-sm font-semibold backdrop-blur">
              <Upload className="h-4 w-4" /> Upload
            </button>
          </div>
        </article>

        {/* Card 2 — Unlimited Accounts */}
        <article className="group relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-brand-blue/95 via-brand-blue to-brand-purple/80 p-7 text-white shadow-2xl">
          <div className="pointer-events-none absolute -left-16 -top-16 h-48 w-48 rounded-full bg-white/15 blur-3xl" aria-hidden />
          <h3 className="relative font-display text-2xl font-700">Unlimited Accounts</h3>
          <p className="relative mt-3 text-sm leading-relaxed text-white/85">
            Link every account you trade. Compare performance across files, brokers, and playbooks
            in one hub.
          </p>

          <div className="relative mt-7 rounded-2xl border border-white/15 bg-white/10 p-3 backdrop-blur">
            <div className="space-y-1.5">
              {ACCOUNTS.map((a) => (
                <div
                  key={a.name}
                  className={
                    "flex items-center justify-between rounded-lg px-3 py-2 text-sm " +
                    (a.primary ? "bg-white/15 font-semibold" : "")
                  }
                >
                  <span className="flex items-center gap-2">
                    <span
                      className={
                        "flex h-4 w-4 items-center justify-center rounded border " +
                        (a.primary
                          ? "border-white bg-white text-brand-blue"
                          : "border-white/40 bg-white/90 text-brand-blue")
                      }
                    >
                      <Check className="h-3 w-3" strokeWidth={3} />
                    </span>
                    {a.name}
                  </span>
                  {a.live && (
                    <span className="rounded-full bg-emerald-400/20 px-2 py-0.5 text-[10px] font-semibold text-emerald-200">
                      live
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="relative mt-4 rounded-2xl border border-white/15 bg-white/10 p-3 backdrop-blur">
            <div className="text-[11px] text-white/70">Combined equity</div>
            <MiniEquityCurve className="mt-2 h-16 w-full" />
          </div>
        </article>

        {/* Card 3 — Automated Statistics */}
        <article className="group relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-brand-purple/90 via-brand-purple to-brand-magenta/70 p-7 text-white shadow-2xl">
          <div className="pointer-events-none absolute -right-16 -bottom-16 h-48 w-48 rounded-full bg-white/15 blur-3xl" aria-hidden />
          <h3 className="relative font-display text-2xl font-700">Automated Statistics</h3>
          <p className="relative mt-3 text-sm leading-relaxed text-white/85">
            Win rate, profit factor, GS Score, streaks, and deep reports — calculated for you on
            every import.
          </p>

          <div className="relative mt-7 grid grid-cols-2 gap-2.5">
            {STAT_TILES.slice(0, 4).map((t) => (
              <div key={t.label} className="rounded-xl bg-white/95 p-3 text-brand-purple">
                <div className="text-[10px] font-medium text-brand-purple/60">{t.label}</div>
                <div className="mt-0.5 text-base font-700">{t.value}</div>
              </div>
            ))}
          </div>

          <div className="relative mt-2.5 flex items-center gap-3 rounded-xl bg-white/10 p-3 backdrop-blur">
            <GsScoreRadar className="h-20 w-20 shrink-0 text-white" />
            <div>
              <div className="text-[11px] text-white/70">GS Score</div>
              <div className="text-2xl font-700">82</div>
              <div className="text-[11px] text-white/70">Edge radar · Pro</div>
            </div>
          </div>
        </article>
      </div>
    </section>
  )
}

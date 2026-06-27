import { ArrowRight, Plus, Sparkles } from "lucide-react"
import { Wordmark } from "@/components/wordmark"

const HERO_BARS = [28, 34, 30, 41, 38, 47, 45, 52, 49, 58, 55, 64, 60, 71]

const STAT_TILES = [
  { label: "Net P&L", value: "+$4.2k", tone: "pos" as const },
  { label: "Win %", value: "58.2%", tone: "neutral" as const },
  { label: "Profit factor", value: "1.86", tone: "neutral" as const },
  { label: "GS Score", value: "82", tone: "brand" as const },
]

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden pt-28 pb-16 lg:pt-36 lg:pb-24">
      {/* ambient glow */}
      <div className="pointer-events-none absolute inset-0 grid-faint opacity-40" aria-hidden />
      <div
        className="pointer-events-none absolute -top-40 left-1/2 h-[36rem] w-[36rem] -translate-x-1/2 rounded-full bg-brand-purple/20 blur-[120px]"
        aria-hidden
      />
      <div
        className="pointer-events-none absolute -right-32 top-40 h-96 w-96 rounded-full bg-brand-blue/15 blur-[120px]"
        aria-hidden
      />

      <div className="relative mx-auto grid max-w-7xl items-center gap-12 px-5 lg:grid-cols-2 lg:gap-10 lg:px-8">
        <div>
          <span className="inline-flex items-center gap-2 rounded-full border border-border bg-muted/40 px-3.5 py-1.5 text-xs font-medium text-muted-foreground backdrop-blur">
            <Sparkles className="h-3.5 w-3.5 text-brand-cyan" />
            All-in-one trading intelligence
          </span>

          <h1 className="mt-6 font-display text-5xl font-700 leading-[1.05] tracking-tight text-balance sm:text-6xl lg:text-[4.25rem]">
            Trade with clarity.
            <br />
            <span className="text-gradient-brand">Grow with confidence.</span>
          </h1>

          <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted-foreground text-pretty">
            Journal every trade, score playbooks against linked accounts, run prop-firm
            simulations, and track how far you&apos;ve come since day one. Layer GS Score with lunar
            cycles and numerology — one hub for serious traders.
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <a
              href="#start"
              className="bg-gradient-brand glow-brand inline-flex items-center gap-2 rounded-xl px-6 py-3.5 text-base font-semibold text-white transition-transform hover:scale-[1.02]"
            >
              Start free
              <ArrowRight className="h-4 w-4" />
            </a>
            <a
              href="#features"
              className="inline-flex items-center gap-2 rounded-xl border border-border bg-muted/30 px-6 py-3.5 text-base font-semibold text-foreground transition-colors hover:bg-muted"
            >
              See features
            </a>
          </div>

          <p className="mt-7 text-sm text-muted-foreground">
            50+ reports · Unlimited playbooks · CSV &amp; XLSX imports
          </p>
        </div>

        {/* Dashboard mockup */}
        <div className="relative">
          <div
            className="pointer-events-none absolute -inset-4 rounded-[2rem] bg-gradient-to-tr from-brand-cyan/10 via-brand-blue/10 to-brand-purple/20 blur-2xl"
            aria-hidden
          />
          <div className="relative rounded-2xl border border-border bg-card/80 p-5 shadow-2xl backdrop-blur-xl">
            <div className="flex items-center justify-between">
              <Wordmark className="text-base text-foreground" />
              <span className="bg-gradient-brand inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold text-white">
                <Plus className="h-3.5 w-3.5" />
                Add Trade
              </span>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {STAT_TILES.map((t) => (
                <div key={t.label} className="rounded-xl border border-border bg-background/50 p-3">
                  <div className="text-[11px] text-muted-foreground">{t.label}</div>
                  <div
                    className={
                      "mt-1 text-lg font-700 " +
                      (t.tone === "pos"
                        ? "text-pos"
                        : t.tone === "brand"
                          ? "text-brand-purple"
                          : "text-foreground")
                    }
                  >
                    {t.value}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 rounded-xl border border-border bg-background/50 p-4">
              <div className="text-xs text-muted-foreground">Daily net cumulative P&amp;L</div>
              <div className="mt-4 flex h-40 items-end gap-1.5" role="img" aria-label="Rising daily cumulative P&L bar chart">
                {HERO_BARS.map((h, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-t-[3px] bg-gradient-to-t from-brand-blue/60 to-brand-purple"
                    style={{ height: `${h}%` }}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

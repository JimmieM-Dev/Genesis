const STATS = [
  { value: "50+", label: "Advanced reports" },
  { value: "Unlimited", label: "Playbooks & accounts" },
  { value: "4", label: "Themes built in" },
  { value: "GS", label: "Score & edge radar" },
]

const BROKERS = [
  "MT4",
  "MT5",
  "Interactive Brokers",
  "HFM",
  "Exness",
  "Pepperstone",
  "IC Markets",
  "XM",
  "JustMarkets",
  "cTrader",
  "NinjaTrader",
  "TradeLocker",
]

export function StatsBrokers() {
  return (
    <section className="relative mx-auto max-w-7xl px-5 py-10 lg:px-8">
      <div className="grid grid-cols-2 divide-border overflow-hidden rounded-2xl border border-border bg-card/50 md:grid-cols-4 md:divide-x">
        {STATS.map((s) => (
          <div key={s.label} className="px-6 py-8 text-center">
            <div className="font-display text-3xl font-700 sm:text-4xl">{s.value}</div>
            <div className="mt-1 text-sm text-muted-foreground">{s.label}</div>
          </div>
        ))}
      </div>

      <p className="mt-12 text-center text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
        More interactive broker exports
      </p>

      <div className="group relative mt-6 overflow-hidden [mask-image:linear-gradient(to_right,transparent,black_12%,black_88%,transparent)]">
        <div className="flex w-max animate-[marquee_32s_linear_infinite] gap-3">
          {[...BROKERS, ...BROKERS].map((b, i) => (
            <span
              key={`${b}-${i}`}
              className="whitespace-nowrap rounded-xl border border-border bg-muted/40 px-5 py-3 text-sm font-medium text-foreground/90"
            >
              {b}
            </span>
          ))}
        </div>
      </div>

      <p className="mt-5 text-center text-sm text-muted-foreground">
        MT4 · MT5 · Interactive Brokers · HFM · Exness · Pepperstone · IC Markets · XM · JustMarkets
      </p>

      <style>{`
        @keyframes marquee {
          from { transform: translateX(0); }
          to { transform: translateX(-50%); }
        }
      `}</style>
    </section>
  )
}

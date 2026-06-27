import { GsMark, Wordmark } from "@/components/wordmark"

const COLUMNS = [
  {
    heading: "Product",
    links: ["Features", "GS School", "Analytics", "Community", "Supported Brokers", "Pricing"],
  },
  {
    heading: "Solutions",
    links: ["Journaling & reports", "Playbooks", "Prop firm simulator", "Numerology & astrology"],
  },
  {
    heading: "Legal",
    links: ["Privacy Policy", "Terms & Conditions", "Contact Us"],
  },
  {
    heading: "Account",
    links: ["Log In", "Get Started Free"],
  },
]

export function SiteFooter() {
  return (
    <footer className="border-t border-border bg-background">
      <div className="mx-auto max-w-7xl px-5 py-16 lg:px-8">
        <div className="grid gap-12 lg:grid-cols-[1.4fr_repeat(4,1fr)]">
          <div>
            <div className="flex items-center gap-2">
              <GsMark className="text-2xl" />
              <Wordmark className="text-lg" />
            </div>
            <p className="mt-5 max-w-sm text-sm leading-relaxed text-muted-foreground">
              Trading foreign exchange, CFDs, and other leveraged products carries a high level of
              risk and may not be suitable for all investors. Past performance is not indicative of
              future results. <Wordmark className="text-foreground/80" /> is an analytics and
              journaling tool — not financial advice.
            </p>
          </div>

          {COLUMNS.map((col) => (
            <div key={col.heading}>
              <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                {col.heading}
              </h3>
              <ul className="mt-4 space-y-3">
                {col.links.map((link) => (
                  <li key={link}>
                    <a
                      href="#"
                      className="text-sm text-foreground/80 transition-colors hover:text-foreground"
                    >
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-12 flex flex-col items-start justify-between gap-3 border-t border-border pt-6 text-sm text-muted-foreground sm:flex-row sm:items-center">
          <p className="flex items-center gap-1.5">
            © 2026 <Wordmark className="text-foreground/70" /> . All rights reserved.
          </p>
          <p>Journal. Analyze. Evolve.</p>
        </div>
      </div>
    </footer>
  )
}

import {
  LineChart,
  BookOpen,
  Calculator,
  BarChart2,
  Sparkles,
  Users,
} from "lucide-react"

const HUB = [
  {
    title: "Automated journaling",
    desc: "CSV/XLSX from any broker, unlimited accounts, equity curve, and session breakdowns.",
    icon: LineChart,
  },
  {
    title: "Playbooks",
    desc: "Link accounts, define rules, and score adherence on every trade.",
    icon: BookOpen,
  },
  {
    title: "Prop firm simulator",
    desc: "Replay trades against daily DD, max DD, and profit targets.",
    icon: Calculator,
  },
  {
    title: "Deep reports",
    desc: "Risk, lunar performance, calendar views, and filterable deep stats.",
    icon: BarChart2,
  },
  {
    title: "Numerology & astrology",
    desc: "Life path, zodiac, compatibility, and lunar forecast tied to your profile.",
    icon: Sparkles,
  },
  {
    title: "Notebook & recaps",
    desc: "Resolutions, 3D cards, weekly recaps, and shareable screenshots.",
    icon: Users,
  },
]

export function FeatureHub() {
  return (
    <section className="relative mx-auto max-w-7xl px-5 py-20 lg:px-8 lg:py-28">
      <h2 className="text-center font-display text-3xl font-700 tracking-tight text-balance sm:text-4xl lg:text-5xl">
        Everything in one hub
      </h2>

      <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {HUB.map((f) => (
          <article
            key={f.title}
            className="group relative overflow-hidden rounded-2xl border border-border bg-card/50 p-7 transition-colors hover:border-brand-purple/40 hover:bg-card"
          >
            <div className="pointer-events-none absolute -right-10 -top-10 h-28 w-28 rounded-full bg-brand-purple/10 opacity-0 blur-2xl transition-opacity group-hover:opacity-100" aria-hidden />
            <div className="inline-flex h-11 w-11 items-center justify-center rounded-xl border border-border bg-muted/50">
              <f.icon className="h-5 w-5 text-brand-cyan" />
            </div>
            <h3 className="mt-5 font-display text-xl font-600">{f.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{f.desc}</p>
          </article>
        ))}
      </div>
    </section>
  )
}

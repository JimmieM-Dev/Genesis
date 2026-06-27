import { GraduationCap, Users } from "lucide-react"
import { Wordmark } from "@/components/wordmark"

const STEPS = [
  {
    n: 1,
    title: "Import your history",
    desc: "MT4, MT5, CSV, or XLSX. Tag accounts and let your journal build automatically.",
  },
  {
    n: 2,
    title: "Define your edge",
    desc: "Playbooks, setups, and optional birth profile for numerology layers.",
  },
  {
    n: 3,
    title: "Review with clarity",
    desc: "Dashboard, reports, streaks, and GS Score show what is working.",
  },
  {
    n: 4,
    title: "Scale with confidence",
    desc: "Prop sim, recaps, and GS School (soon) keep you accountable.",
  },
]

export function HowItWorks() {
  return (
    <section id="learn" className="relative mx-auto max-w-7xl px-5 py-20 lg:px-8 lg:py-28">
      {/* Learn and level up */}
      <div className="mx-auto max-w-2xl text-center">
        <h2 className="font-display text-3xl font-700 tracking-tight text-balance sm:text-4xl lg:text-5xl">
          Learn and level up
        </h2>
        <p className="mt-4 text-lg text-muted-foreground text-pretty">
          Education and feedback built into the platform.
        </p>
      </div>

      <div className="mt-12 grid gap-6 lg:grid-cols-2">
        <article className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-brand-purple/85 to-brand-magenta/70 p-8 text-white shadow-2xl">
          <div className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-white/15">
            <GraduationCap className="h-6 w-6" />
          </div>
          <h3 className="mt-5 font-display text-2xl font-700">GS School</h3>
          <p className="mt-3 max-w-md text-sm leading-relaxed text-white/85">
            Structured lessons on risk, journaling, playbooks, and psychology. Master the habits
            behind funded-account passes, not just entries and exits.
          </p>
          <span className="mt-5 inline-block rounded-full bg-white/15 px-4 py-1.5 text-sm font-semibold">
            Coming in Wave 6
          </span>
        </article>

        <article className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-brand-blue/90 to-brand-purple/75 p-8 text-white shadow-2xl">
          <div className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-white/15">
            <Users className="h-6 w-6" />
          </div>
          <h3 className="mt-5 font-display text-2xl font-700">Mentor mode</h3>
          <p className="mt-3 max-w-md text-sm leading-relaxed text-white/85">
            Share trades with a mentor for feedback, compare execution to your playbooks, and close
            the loop between review and the next session.
          </p>
          <span className="mt-5 inline-block rounded-full bg-white/15 px-4 py-1.5 text-sm font-semibold">
            Coming in Wave 6
          </span>
        </article>
      </div>

      {/* How Genesis works timeline */}
      <h2 className="mt-24 text-center font-display text-3xl font-700 tracking-tight text-balance sm:text-4xl lg:text-5xl">
        How <Wordmark className="text-gradient-brand" /> works
      </h2>

      <ol className="mx-auto mt-12 max-w-2xl space-y-8">
        {STEPS.map((s, i) => (
          <li key={s.n} className="relative flex gap-5 pl-2">
            <div className="flex flex-col items-center">
              <span className="bg-gradient-brand flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-sm font-700 text-white">
                {s.n}
              </span>
              {i < STEPS.length - 1 && (
                <span className="mt-1 w-px flex-1 bg-gradient-to-b from-brand-purple/60 to-brand-cyan/30" />
              )}
            </div>
            <div className="pb-2">
              <h3 className="font-display text-xl font-600">{s.title}</h3>
              <p className="mt-1.5 text-muted-foreground">{s.desc}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}

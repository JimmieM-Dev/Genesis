import { ArrowRight } from "lucide-react"
import { Wordmark } from "@/components/wordmark"

export function FinalCta() {
  return (
    <section id="start" className="relative overflow-hidden py-24 lg:py-32">
      <div
        className="pointer-events-none absolute left-1/2 top-1/2 h-80 w-[44rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand-purple/20 blur-[120px]"
        aria-hidden
      />
      <div className="relative mx-auto max-w-3xl px-5 text-center lg:px-8">
        <h2 className="font-display text-4xl font-700 tracking-tight text-balance sm:text-5xl lg:text-6xl">
          Your edge is already there.
        </h2>
        <p className="mt-4 text-xl text-muted-foreground">
          <Wordmark className="text-gradient-brand" /> helps you see it.
        </p>
        <p className="mt-2 text-lg text-muted-foreground">Journal. Analyze. Evolve.</p>

        <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
          <a
            href="#start"
            className="bg-gradient-brand glow-brand inline-flex items-center gap-2 rounded-xl px-7 py-3.5 text-base font-semibold text-white transition-transform hover:scale-[1.02]"
          >
            Start free
            <ArrowRight className="h-4 w-4" />
          </a>
          <a
            href="#login"
            className="inline-flex items-center gap-2 rounded-xl border border-border bg-muted/30 px-7 py-3.5 text-base font-semibold text-foreground transition-colors hover:bg-muted"
          >
            Log in
          </a>
        </div>
      </div>
    </section>
  )
}

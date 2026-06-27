import { MiniDonut } from "@/components/mini-visuals"

const PALETTE = [
  "#a78bfa",
  "#60a5fa",
  "#f59e0b",
  "#ec4899",
  "#22d3ee",
  "#f59e0b",
  "#34d399",
  "#818cf8",
  "#fb7185",
]

const DECKS = [
  {
    title: "Western zodiac",
    items: [
      ["Cancer", 12],
      ["Leo", 10],
      ["Pisces", 8],
      ["Virgo", 7],
      ["Aries", 7],
      ["Gemini", 7],
    ],
  },
  {
    title: "Chinese zodiac",
    items: [
      ["Rooster", 27],
      ["Dog", 14],
      ["Monkey", 9],
      ["Goat", 6],
      ["Horse", 4],
      ["Dragon", 3],
    ],
  },
  {
    title: "Life Path",
    items: [
      ["Path 7", 13],
      ["Path 9", 10],
      ["Path 8", 9],
      ["Path 6", 6],
      ["Path 1", 8],
      ["Path 11", 4],
    ],
  },
]

export function Numerology() {
  return (
    <section id="community" className="relative overflow-hidden py-20 lg:py-24">
      <div
        className="pointer-events-none absolute right-0 top-1/3 h-72 w-72 rounded-full bg-brand-magenta/10 blur-[120px]"
        aria-hidden
      />
      <div className="relative mx-auto max-w-7xl px-5 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-brand-cyan">
            Numerology &amp; astrology
          </p>
          <h2 className="mt-3 font-display text-3xl font-700 tracking-tight text-balance sm:text-4xl lg:text-5xl">
            A layer no other journal has
          </h2>
          <p className="mt-4 text-lg text-muted-foreground text-pretty">
            Life path, zodiac, compatibility, and lunar forecast — tied to your profile and your
            real trade history.
          </p>
        </div>

        <div className="mt-12 grid gap-6 lg:grid-cols-3">
          {DECKS.map((deck) => (
            <div key={deck.title} className="rounded-2xl border border-border bg-card/50 p-6">
              <h3 className="font-600">{deck.title} distribution</h3>
              <div className="mt-5 flex items-center gap-5">
                <MiniDonut
                  className="h-28 w-28 shrink-0"
                  segments={deck.items.map((it, i) => ({
                    value: it[1] as number,
                    color: PALETTE[i % PALETTE.length],
                  }))}
                />
                <ul className="grid flex-1 grid-cols-1 gap-1.5 text-sm">
                  {deck.items.map((it, i) => (
                    <li key={it[0]} className="flex items-center justify-between gap-2">
                      <span className="flex items-center gap-2 text-muted-foreground">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ background: PALETTE[i % PALETTE.length] }}
                        />
                        {it[0]}
                      </span>
                      <span className="font-600 text-foreground">{it[1]}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

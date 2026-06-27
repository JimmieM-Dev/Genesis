import { cn } from "@/lib/utils"

/**
 * GENESIS wordmark with the reversed "E" letters (GƎNƎSIS),
 * matching the in-app brand mark.
 */
export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cn("wordmark font-display font-700 tracking-tight", className)}>
      G<span className="flip">E</span>N<span className="flip">E</span>SIS
    </span>
  )
}

export function GsMark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "font-display font-700 text-gradient-brand leading-none tracking-tighter",
        className,
      )}
    >
      GS
    </span>
  )
}

export function BrandLogo({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <GsMark className="text-2xl" />
      <Wordmark className="text-lg text-foreground" />
    </div>
  )
}

import type { Metadata } from "next"
import { Inter, Space_Grotesk } from "next/font/google"
import type { ReactNode } from "react"
import "./globals.css"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
})

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  display: "swap",
  weight: ["500", "600", "700"],
})

export const metadata: Metadata = {
  title: "Genesis — Trade with clarity. Grow with confidence.",
  description:
    "Genesis is the all-in-one trading intelligence hub. Journal every trade, score playbooks against linked accounts, run prop-firm simulations, and layer GS Score with lunar cycles and numerology.",
  keywords: [
    "trading journal",
    "trade analytics",
    "prop firm simulator",
    "playbooks",
    "GS Score",
    "MT4 MT5 import",
  ],
  openGraph: {
    title: "Genesis — Trade with clarity. Grow with confidence.",
    description:
      "Journal, analyze, and evolve. One hub for serious traders with automated stats, prop-firm sims, and numerology layers.",
    type: "website",
  },
}

export const viewport = {
  themeColor: "#0a0a0f",
  width: "device-width",
  initialScale: 1,
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${spaceGrotesk.variable} bg-background`}>
      <body className="antialiased">{children}</body>
    </html>
  )
}

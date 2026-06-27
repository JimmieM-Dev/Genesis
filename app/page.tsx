import { SiteHeader } from "@/components/site-header"
import { Hero } from "@/components/hero"
import { StatsBrokers } from "@/components/stats-brokers"
import { FeatureShowcase } from "@/components/feature-showcase"
import { WhyTraders } from "@/components/why-traders"
import { FeatureHub } from "@/components/feature-hub"
import { InsightSections } from "@/components/insight-sections"
import { Numerology } from "@/components/numerology"
import { Network } from "@/components/network"
import { HowItWorks } from "@/components/how-it-works"
import { FinalCta } from "@/components/final-cta"
import { SiteFooter } from "@/components/site-footer"

export default function Page() {
  return (
    <>
      <SiteHeader />
      <main>
        <Hero />
        <StatsBrokers />
        <FeatureShowcase />
        <WhyTraders />
        <FeatureHub />
        <InsightSections />
        <Numerology />
        <Network />
        <HowItWorks />
        <FinalCta />
      </main>
      <SiteFooter />
    </>
  )
}

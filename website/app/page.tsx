"use client";

import Hero from "@/components/Hero";
import TrustedBy from "@/components/TrustedBy";
// import ReleaseTimeline from "@/components/ReleaseTimeline";
import Features from "@/components/Features";
import HowItWorks from "@/components/HowItWorks";
import ToolShowcase from "@/components/ToolShowcase";
import PresetShowcase from "@/components/PresetShowcase";
import QuickStart from "@/components/QuickStart";
import Examples from "@/components/Examples";
import Community from "@/components/Community";
import CTA from "@/components/CTA";
import Footer from "@/components/Footer";
import Navbar from "@/components/Navbar";

export default function Home() {
  return (
    <main className="min-h-screen bg-white dark:bg-[#020c08] transition-colors duration-300">
      <Navbar />
      <Hero />
      <TrustedBy />
      {/* <ReleaseTimeline /> */}
      <Features />
      <HowItWorks />
      <ToolShowcase />
      <PresetShowcase />
      <QuickStart />
      <Examples />
      <Community />
      <CTA />
      <Footer />
    </main>
  );
}

'use client'

import { useEffect, useState } from 'react'
import { SplineScene } from "@/components/ui/spline"
import { ShimmerButton } from "@/components/ui/shimmer-button"
import { Spotlight } from "@/components/ui/spotlight"
import { AuthOverlay } from "@/components/auth/AuthOverlay"
import { BackgroundBeams } from '@/components/ui/background-beams'
import { TrendingCarousel } from '@/components/ui/trending-carousel'

export default function Home() {
  const [showButton, setShowButton] = useState(false)
  const [pageLoaded, setPageLoaded] = useState(false)
  const [showAuthModal, setShowAuthModal] = useState(false)

  useEffect(() => {
    // Show initial content
    setPageLoaded(true)
    
    // Show button after a shorter delay (5 seconds instead of 10)
    const timer = setTimeout(() => {
      setShowButton(true)
    }, 5000)

    return () => clearTimeout(timer)
  }, [])

  const handleGetStarted = () => {
    setShowAuthModal(true)
  }

  return (
    <>
      <main className={`min-h-screen w-full bg-black relative flex flex-col transition-opacity duration-700 ${pageLoaded ? 'opacity-100' : 'opacity-0'}`}>
        <BackgroundBeams className="opacity-20" />
        {/* Hero Section with Scene */}
        <div className="h-screen relative flex flex-col items-center justify-center overflow-hidden">
          <Spotlight
            className="-top-40 left-0 md:left-60 md:-top-20"
            fill="white"
          />
          
          {/* Title & Button Container */}
          <div className="absolute z-10 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center flex flex-col items-center space-y-12">
            <div className="space-y-6">
              <h1 className="text-7xl md:text-9xl font-bold">
                <span className="bg-gradient-to-r from-slate-300 via-slate-100 to-slate-300 bg-clip-text text-transparent animate-gradient">
                  NexBo
                </span>
              </h1>
              <p className="text-xl md:text-2xl text-slate-400 tracking-widest font-light">
                THE GEN-AI FINANCE ASSISTANCE
              </p>
            </div>
            
            {/* Animated Button */}
            <div className={`transition-opacity duration-1000 ${showButton ? 'opacity-100' : 'opacity-0'} transform scale-110`}>
              <ShimmerButton
                onClick={handleGetStarted}
                className="shadow-2xl px-12 py-4"
              >
                <span className="text-xl md:text-2xl font-medium text-[#F0F0F0]">
                  Get Started
                </span>
              </ShimmerButton>
            </div>
          </div>

          {/* 3D Scene Background */}
          <div className="absolute inset-0">
            <SplineScene 
              scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
              className="opacity-80"
            />
          </div>
        </div>

        {/* Trending Updates */}
        <div className="w-full max-w-6xl mx-auto px-4 mb-12">
          <TrendingCarousel />
        </div>

        {/* Features Section */}
        <div className="px-4 py-20 bg-black/90">
          {/* Section Title */}
          <div className="text-center mb-16">
            <h2 className="text-5xl md:text-6xl font-bold bg-[linear-gradient(45deg,#c0c0c0,#ffffff,#8f8f8f,#ffffff)] bg-clip-text text-transparent bg-[length:200%] animate-gradient">
              Why NexBo?
            </h2>
            <div className="w-24 h-1 bg-gradient-to-r from-slate-500 to-slate-300 mx-auto mt-4 rounded-full" />
          </div>

          <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
            <FeatureCard 
              title="AI-Powered Analysis"
              description="Get real-time financial insights powered by advanced AI algorithms."
            />
            <FeatureCard 
              title="Smart Portfolio"
              description="Manage and track your investments with intelligent portfolio recommendations."
            />
            <FeatureCard 
              title="Financial Learning"
              description="Access personalized financial education and market insights." 
            />
          </div>
        </div>
      </main>

      <AuthOverlay 
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        initialView="signup"
      />
    </>
  )
}

function FeatureCard({ title, description }: { title: string, description: string }) {
  return (
    <div className="relative group">
      <div className="absolute inset-0 bg-gradient-to-r from-pink-500 to-violet-500 rounded-lg blur opacity-25 group-hover:opacity-75 transition duration-200" />
      <div className="relative p-6 bg-black rounded-lg border border-slate-800">
        <h3 className="text-xl font-semibold text-white mb-2">{title}</h3>
        <p className="text-slate-400">{description}</p>
      </div>
    </div>
  )
}
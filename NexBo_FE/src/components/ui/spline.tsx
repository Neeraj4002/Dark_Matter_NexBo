'use client'

import { Suspense, useEffect, useState } from 'react'

// We'll conditionally import Spline only on the client side
let SplineComponent: React.ComponentType<{ scene: string }> | null = null;// Create a visual fallback for the 3D component
function SplineFallback() {
  return (
    <div className="w-full h-full flex items-center justify-center">
      <div className="relative w-full h-full overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 via-purple-500/20 to-pink-500/20" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-64 w-64 rounded-full bg-gradient-to-br from-blue-500/30 to-purple-600/30 blur-xl animate-pulse" />
        <div className="absolute top-1/4 left-1/3 h-24 w-24 rounded-full bg-blue-400/20 blur-lg animate-pulse delay-300" />
        <div className="absolute bottom-1/4 right-1/3 h-32 w-32 rounded-full bg-purple-400/20 blur-lg animate-pulse delay-700" />
      </div>
    </div>
  )
}
interface SplineProps {
  scene: string
  className?: string
}

export function SplineScene({ scene, className }: SplineProps) {
  const [isClient, setIsClient] = useState(false)
  const [SplineLoaded, setSplineLoaded] = useState(false)
  
  useEffect(() => {
    setIsClient(true)
    
    // Only load Spline on client side
    const loadSpline = async () => {
      try {
        // Dynamic import with error handling
        const splineModule = await import('@splinetool/react-spline').catch(() => null)
        if (splineModule) {
          SplineComponent = splineModule.default
          setSplineLoaded(true)
        }
      } catch (error) {
        console.error("Could not load Spline component:", error)
      }
    }    
    loadSpline()
  }, [])

  // Always render the fallback on server
  if (!isClient) {
    return <SplineFallback />
  }
  
  // Render the fallback if Spline failed to load
  if (!SplineLoaded || !SplineComponent) {
    return <SplineFallback />
  }

  return (
    <Suspense fallback={<SplineFallback />}>
      <div className={className}>
        <SplineComponent scene={scene} />
      </div>
    </Suspense>
  )
}
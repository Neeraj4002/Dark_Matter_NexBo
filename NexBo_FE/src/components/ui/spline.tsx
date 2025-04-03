'use client'

import { Suspense } from 'react'
import dynamic from 'next/dynamic'

const Spline = dynamic(() => import('@splinetool/react-spline'), {
  ssr: false,
})

interface SplineProps {
  scene: string
}

export function SplineScene({ scene }: SplineProps) {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Spline scene={scene} />
    </Suspense>
  )
}
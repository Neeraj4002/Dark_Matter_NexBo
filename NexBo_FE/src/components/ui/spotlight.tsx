'use client'

import { cn } from "@/lib/utils"

export function Spotlight({
  className,
  fill = 'white',
}: {
  className?: string
  fill?: string
}) {
  return (
    <svg
      className={cn(
        'animate-spotlight pointer-events-none absolute z-[1] h-[169%] w-[138%] opacity-0',
        className
      )}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 3787 2842"
      fill="none"
    >
      <g filter="url(#filter)">
        <ellipse
          cx="1924"
          cy="273"
          rx="1924"
          ry="273"
          transform="matrix(-0.822377 -0.568943 -0.568943 0.822377 3787 2210)"
          fill={fill}
          fillOpacity="0.21"
        ></ellipse>
      </g>
      <defs>
        <filter
          id="filter"
          x="0"
          y="0"
          width="3787"
          height="2842"
          filterUnits="userSpaceOnUse"
          colorInterpolationFilters="sRGB"
        >
          <feFlood floodOpacity="0" result="BackgroundImageFix" />
          <feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape" />
          <feGaussianBlur stdDeviation="346" result="effect1_foregroundBlur_1065_8" />
        </filter>
      </defs>
    </svg>
  )
}
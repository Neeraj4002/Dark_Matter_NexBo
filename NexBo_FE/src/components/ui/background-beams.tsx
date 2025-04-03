"use client"
import React from "react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

export const BackgroundBeams = React.memo(
  ({ className }: { className?: string }) => {
    const paths = Array.from({ length: 40 }, (_, i) => {
      const offset = i * 10
      return `M${-380 + offset} ${-189 - offset}C${-380 + offset} ${-189 - offset} ${
        -312 + offset
      } ${216 - offset} ${152 + offset} ${343 - offset}C${616 + offset} ${
        470 - offset
      } ${684 + offset} ${875 - offset} ${684 + offset} ${875 - offset}`
    })

    return (
      <div
        className={cn(
          "absolute h-full w-full inset-0 [mask-size:40px] [mask-repeat:no-repeat] flex items-center justify-center",
          className,
        )}
      >
        <svg
          className="z-0 h-full w-full pointer-events-none absolute"
          width="100%"
          height="100%"
          viewBox="0 0 696 316"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M-380 -189C-380 -189 -312 216 152 343C616 470 684 875 684 875"
            stroke="url(#paint0_radial_242_278)"
            strokeOpacity="0.1"
            strokeWidth="1"
          />

          {paths.map((path, index) => (
            <motion.path
              key={`path-${index}`}
              d={path}
              stroke={`url(#linearGradient-${index})`}
              strokeOpacity="0.6"
              strokeWidth="0.8"
            />
          ))}

          <defs>
            {paths.map((_, index) => (
              <motion.linearGradient
                id={`linearGradient-${index}`}
                key={`gradient-${index}`}
                initial={{
                  x1: "0%",
                  x2: "0%",
                  y1: "0%",
                  y2: "0%",
                }}
                animate={{
                  x1: ["0%", "100%"],
                  x2: ["0%", "95%"],
                  y1: ["0%", "100%"],
                  y2: ["0%", `${93 + Math.random() * 8}%`],
                }}
                transition={{
                  duration: Math.random() * 10 + 10,
                  ease: "easeInOut",
                  repeat: Infinity,
                  delay: Math.random() * 10,
                }}
              >
                <stop stopColor="#FFFFFF" stopOpacity="0" />
                <stop stopColor="#FFFFFF" stopOpacity="0.8" />
                <stop offset="32.5%" stopColor="#FFFFFF" stopOpacity="0.6" />
                <stop offset="100%" stopColor="#FFFFFF" stopOpacity="0" />
              </motion.linearGradient>
            ))}

            <radialGradient
              id="paint0_radial_242_278"
              cx="0"
              cy="0"
              r="1"
              gradientUnits="userSpaceOnUse"
              gradientTransform="translate(352 34) rotate(90) scale(555 1560.62)"
            >
              <stop offset="0.0666667" stopColor="#FFFFFF" stopOpacity="0.8" />
              <stop offset="0.243243" stopColor="#FFFFFF" stopOpacity="0.4" />
              <stop offset="0.43594" stopColor="#FFFFFF" stopOpacity="0" />
            </radialGradient>
          </defs>
        </svg>
      </div>
    )
  },
)

BackgroundBeams.displayName = "BackgroundBeams"

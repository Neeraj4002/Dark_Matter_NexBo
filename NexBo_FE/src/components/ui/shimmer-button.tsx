import React from "react";
import { cn } from "@/lib/utils";

export interface ShimmerButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  shimmerColor?: string;
  shimmerSize?: string;
  borderRadius?: string;
  shimmerDuration?: string;
  background?: string;
  className?: string;
  children?: React.ReactNode;
}

const ShimmerButton = React.forwardRef<HTMLButtonElement, ShimmerButtonProps>(
  (
    {
      shimmerColor = "rgba(255, 255, 255, 0.2)",
      shimmerSize = "0.1em",
      shimmerDuration = "2s",
      borderRadius = "100px",
      background = "linear-gradient(to right, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05))",
      className,
      children,
      ...props
    },
    ref,
  ) => {
    return (
      <button
        data-spread="90deg"
        data-shimmer-color={shimmerColor}
        data-radius={borderRadius}
        data-speed={shimmerDuration}
        data-cut={shimmerSize}
        data-bg={background}
        className={cn(
          "group relative z-0 flex cursor-pointer items-center justify-center overflow-hidden whitespace-nowrap border border-white/20 px-6 py-3",
          "text-base font-semibold tracking-wide text-white/90",
          "bg-gradient-to-r from-white/10 to-white/5 backdrop-blur-sm",
          "hover:border-white/30 hover:from-white/20 hover:to-white/10 hover:text-white",
          "transform-gpu transition-all duration-300 ease-in-out active:translate-y-px",
          "[border-radius:var(--radius)]",
          className,
        )}
        ref={ref}
        {...props}
      >
        {/* spark container */}
        <div
          className="absolute inset-0 overflow-visible [container-type:size]"
          aria-hidden="true"
        >
          {/* spark */}
          <div className="absolute inset-0 h-[100cqh] animate-shimmer [background:linear-gradient(var(--spread),transparent_calc(0%-var(--cut)),var(--shimmer-color)_calc(0%+var(--cut))_calc(100%-var(--cut)),transparent_calc(100%+var(--cut)))] [translate:0_-100cqh]" />
        </div>

        {/* content */}
        <div className="relative z-10 flex items-center justify-center gap-2">
          {children}
        </div>

        {/* gradient overlay */}
        <div className="absolute inset-0 rounded-[inherit] bg-gradient-to-r from-white/[0.08] to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
      </button>
    );
  },);

ShimmerButton.displayName = "ShimmerButton";

export { ShimmerButton };

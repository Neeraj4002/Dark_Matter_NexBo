"use client";

import { cn } from "@/lib/utils";

interface NexBoLogoProps {
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function NexBoLogo({ size = "md", className }: NexBoLogoProps) {
  const sizeClasses = {
    sm: "w-8 h-8 text-sm",
    md: "w-10 h-10 text-base",
    lg: "w-12 h-12 text-lg",
  };

  return (
    <div
      className={cn(
        "rounded-lg flex items-center justify-center font-bold",
        "bg-gradient-to-br from-zinc-200 to-zinc-400",
        sizeClasses[size],
        className
      )}
    >
      <span className="bg-clip-text text-transparent bg-gradient-to-br from-zinc-800 to-zinc-600">
        NB
      </span>
    </div>
  );
}

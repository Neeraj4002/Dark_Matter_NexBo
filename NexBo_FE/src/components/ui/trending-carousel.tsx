"use client";

import { useState, useEffect } from "react";
import { Sparkles } from "lucide-react";

const trendingUpdates = [
  {
    title: "Stock Market Updates",
    content: "S&P 500 hits new all-time high as tech stocks surge",
    url: "https://www.marketwatch.com/"
  },
  {
    title: "Crypto Insights",
    content: "Bitcoin reaches $75,000 milestone amid growing institutional adoption",
    url: "https://www.coindesk.com/"
  },
  {
    title: "Financial News",
    content: "Federal Reserve announces key policy changes for 2025",
    url: "https://www.bloomberg.com/"
  }
];

export const TrendingCarousel = () => {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex((current) => (current + 1) % trendingUpdates.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full rounded-2xl overflow-hidden bg-black/40 backdrop-blur-xl border border-white/20">
      <div className="p-4 border-b border-white/10 flex items-center gap-2">
        <Sparkles className="w-5 h-5 text-white" />
        <h3 className="font-bold text-lg animate-blink text-white drop-shadow-[0_0_10px_rgba(255,255,255,0.5)]">
          Trending Updates
        </h3>
      </div>
      <div className="p-6 overflow-hidden">
        <div className="flex flex-col gap-4">
          {trendingUpdates.map((update, index) => (
            <a
              href={update.url}
              target="_blank"
              rel="noopener noreferrer"
              key={index}
              className={index === activeIndex ? "block" : "hidden"}
            >
              <div className="p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors cursor-pointer group">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-white font-bold text-lg drop-shadow-[0_0_8px_rgba(255,255,255,0.3)]">
                    {update.title}
                  </h4>
                  <div className="bg-white/10 rounded-full p-1.5 group-hover:bg-white/20 transition-colors">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                </div>
                <p className="text-white/90 text-base font-medium">
                  {update.content}
                </p>
              </div>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
};
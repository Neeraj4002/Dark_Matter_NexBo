"use client";

import { BackgroundBeams } from '@/components/ui/background-beams';
import { NexBoLogo } from '@/components/ui/nexbo-logo';
import { Button } from "@/components/ui/button";
import { ArrowLeft, BookOpen, TrendingUp, LineChart } from "lucide-react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { TrendingCarousel } from "@/components/ui/trending-carousel";
import { cn } from "@/lib/utils";

interface LevelCardProps {
  level: string;
  books: string[];
  isComingSoon?: boolean;
}

const LevelCard = ({ level, books, isComingSoon = false }: LevelCardProps) => {
  const router = useRouter();

  // Define icon colors and components based on level
  const levelConfig = {
    Beginner: {
      icon: BookOpen,
      color: "emerald",
    },
    Intermediate: {
      icon: TrendingUp,
      color: "amber",
    },
    Advanced: {
      icon: LineChart,
      color: "rose",
    },
  };

  const { icon: Icon, color } = levelConfig[level as keyof typeof levelConfig];

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      onClick={() => !isComingSoon && router.push(`/learn/beginner`)}
      className={cn(
        "relative p-8 rounded-2xl overflow-hidden",
        "bg-black/40 backdrop-blur-xl border border-white/20",
        "transition-all duration-300 shadow-xl",
        `hover:border-${color}-500/50 hover:shadow-${color}-500/20`
      )}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-pearl-50/10 via-steel-100/10 to-ivory-50/10 opacity-30" />
      
      <div className="relative">
        <div className="flex items-center gap-4 mb-6">
          <div className={cn(
            "w-12 h-12 rounded-xl flex items-center justify-center",
            `bg-${color}-500/20 border border-${color}-500/30`,
            "transition-colors duration-300"
          )}>
            <Icon className={`w-7 h-7 text-${color}-400`} />
          </div>
          <h3 className="text-2xl font-bold text-white">{level}</h3>
        </div>

        <ul className="space-y-4 mb-8">
          {books.map((book, index) => (
            <li key={index} 
                className="text-white/80 hover:text-white 
                          transition-colors duration-200 pl-4 border-l-2 
                          border-white/20 hover:border-white/40">
              {book}
            </li>
          ))}
        </ul>

        <Button
          onClick={() => !isComingSoon && router.push(`/home/learn/${level.toLowerCase()}`)}
          disabled={isComingSoon}
          className={cn(
            "w-full py-6 text-lg font-medium transition-all duration-300 rounded-xl",
            isComingSoon 
              ? "bg-white/10 text-white/50 cursor-not-allowed"
              : "bg-white/10 hover:bg-white/20 text-white shadow-lg"
          )}
        >
          {isComingSoon ? "Coming Soon" : "Start Learning"}
        </Button>
      </div>
    </motion.div>
  );
};

export default function LearnPage() {
  const router = useRouter();

  const levels = [
    {
      level: "Beginner",
      books: [
        "Rich Dad Poor Dad - Robert Kiyosaki",
        "The Psychology of Money - Morgan Housel",
        "The Total Money Makeover - Dave Ramsey",
      ],
    },
    {
      level: "Intermediate",
      books: [
        "The Intelligent Investor - Benjamin Graham",
        "Common Stocks and Uncommon Profits - Philip Fisher",
        "One Up On Wall Street - Peter Lynch",
      ],
      isComingSoon: true,
    },
    {
      level: "Advanced",
      books: [
        "Security Analysis - Benjamin Graham",
        "Options, Futures, and Other Derivatives - John Hull",
        "Technical Analysis of the Financial Markets - John Murphy",
      ],
      isComingSoon: true,
    },
  ];

  return (
    <div className="min-h-screen bg-black relative overflow-hidden">
      <BackgroundBeams className="opacity-30" />
      
      <nav className="fixed top-0 w-full bg-black/80 backdrop-blur-xl border-b border-white/20 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              onClick={() => router.push("/home")}
              className="group p-3 rounded-xl bg-white/10 border border-white/20 
                       hover:bg-white/20 hover:border-white/30 transition-all duration-300"
            >
              <ArrowLeft className="w-5 h-5 text-white group-hover:text-white" />
            </Button>
            <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-4">
              <NexBoLogo className="w-8 h-8" />
              <div className="hidden sm:block h-6 w-px bg-white/20" />
              <div>
                <h1 className="text-xl font-bold text-white">
                  Learning Hub
                </h1>
                <p className="text-sm text-white/80">
                  Financial Education Platform
                </p>
              </div>
            </div>
            <div className="w-14" /> {/* Spacer */}
          </div>
        </div>
      </nav>

      <main className="pt-32 pb-16 px-6 max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-6xl font-bold tracking-tight mb-4
                       bg-clip-text text-transparent 
                       bg-gradient-to-r from-white via-pearl-50 to-steel-200
                       drop-shadow-lg">
            Learn to Grow Faster
          </h1>
          <p className="text-lg text-white/90 font-medium">
            Master financial literacy through our curated collection of resources
          </p>
        </motion.div>

        <TrendingCarousel />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12">
          {levels.map((levelData, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2 }}
            >
              <LevelCard {...levelData} />
            </motion.div>
          ))}
        </div>
      </main>
    </div>
  );
}
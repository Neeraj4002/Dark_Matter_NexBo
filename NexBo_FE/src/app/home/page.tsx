"use client";

import { BackgroundBeams } from "@/components/ui/background-beams";
import { GridPatternCard, GridPatternCardBody } from "@/components/ui/card-with-grid-pattern";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { Home, MessageSquare, GraduationCap, LogOut, CircleUserRound } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/context/auth-context";
import { Button } from "@/components/ui/button";
import { NavBar } from "@/components/ui/tubelight-navbar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { auth } from '@/lib/firebase';
import { WordPullUp } from "@/components/ui/word-pull-up";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export default function HomePage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  if (loading) {
    return (
      <div className="min-h-screen w-full bg-black flex items-center justify-center">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-white border-t-transparent" />
      </div>
    );
  }

  if (!user) {
    router.replace('/');
    return null;
  }

  const navItems = [
    { name: 'Home', url: '/home', icon: Home },
    { name: 'NexBo Chat', url: '/chat', icon: MessageSquare },
    { name: 'Learning Hub', url: '/learn', icon: GraduationCap }
  ];

  const handleSignOut = async () => {
    try {
      await auth.signOut();
      router.push('/');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  return (
    <div className="min-h-screen bg-black">
      <BackgroundBeams className="opacity-20" />
      
      {/* Profile Dropdown */}
      <div className="fixed top-0 right-0 z-50 p-4">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size="icon" variant="outline" className="border-white/10 bg-white/5">
              <CircleUserRound className="h-5 w-5 text-white" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56 bg-black/90 backdrop-blur-sm border-white/10">
            <DropdownMenuLabel className="flex flex-col">
              <span className="text-white/80">Signed in as</span>
              <span className="text-xs font-normal text-white/60">{user?.email}</span>
            </DropdownMenuLabel>
            <DropdownMenuSeparator className="bg-white/10" />
            <DropdownMenuItem 
              className="text-white/70 hover:text-white focus:text-white cursor-pointer"
              onClick={handleSignOut}
            >
              <LogOut className="mr-2 h-4 w-4" />
              <span>Sign Out</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Hero Section */}
      <div className="flex flex-col items-center justify-center py-32 space-y-8">
        <div className="relative">
          {/* Text with steel gradient */}
          <WordPullUp
            words="Welcome to NexBo"
            className={cn(
              "text-6xl md:text-8xl font-bold",
              "bg-clip-text text-transparent",
              "bg-[linear-gradient(45deg,_#c9c9c9_0%,_#ffffff_25%,_#b8b8b8_50%,_#ffffff_75%,_#c9c9c9_100%)]",
              "animate-shimmer bg-[length:200%_100%]",
              "drop-shadow-[0_0_30px_rgba(255,255,255,0.15)]"
            )}
            wrapperFramerProps={{
              hidden: { opacity: 0 },
              show: {
                opacity: 1,
                transition: {
                  staggerChildren: 0.15,
                  delayChildren: 0.3,
                },
              },
            }}
            framerProps={{
              hidden: { y: 40, opacity: 0 },
              show: { 
                y: 0, 
                opacity: 1,
                transition: {
                  type: "spring",
                  damping: 12,
                  stiffness: 100,
                },
              },
            }}
          />

          {/* Subtle glow effect */}
          <div className="absolute inset-0 blur-3xl opacity-30 bg-gradient-to-r from-blue-500 via-transparent to-purple-500" />
        </div>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ 
            delay: 1.2,
            duration: 0.5,
          }}
          className="mt-6 max-w-2xl text-center text-lg md:text-xl text-white/60"
        >
          Your AI-powered financial companion for intelligent investment decisions 
          and structured learning in finance.
        </motion.p>
      </div>

      {/* Features Cards */}
      <div className="max-w-6xl mx-auto px-4 py-10">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* NexBo Chat Card */}
          <GridPatternCard 
            className="group hover:shadow-2xl hover:shadow-blue-500/20 transition-all duration-500"
            gradientClassName="from-white/5 via-white/5 to-white/10"
          >
            <GridPatternCardBody className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-400 to-blue-600 flex items-center justify-center">
                  <MessageSquare className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white">NexBo Chat</h3>
              </div>
              <p className="text-white/70 leading-relaxed">
                Experience intelligent financial conversations with our AI assistant. Get personalized stock recommendations, market insights, and expert financial guidance tailored to your needs.
              </p>
              <Link href="/chat" className="block w-full">
                <ShimmerButton 
                  className="w-full tracking-wide"
                  shimmerColor="rgba(255, 255, 255, 0.3)"
                  background="linear-gradient(to right, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05))"
                >
                  Let&apos;s Get Started →
                </ShimmerButton>
              </Link>
            </GridPatternCardBody>
          </GridPatternCard>

          {/* Learning Hub Card */}
          <GridPatternCard 
            className="group hover:shadow-2xl hover:shadow-emerald-500/20 transition-all duration-500"
            gradientClassName="from-white/5 via-white/5 to-white/10"
          >
            <GridPatternCardBody className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-emerald-400 to-emerald-600 flex items-center justify-center">
                  <GraduationCap className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white">Learning Hub</h3>
              </div>
              <p className="text-white/70 leading-relaxed">
                Embark on a structured learning journey from basics to expert level. Access curated finance lessons, interactive summaries, and get real-time assistance at every step of your learning path.
              </p>
              <Link href="/learn" className="block w-full">
                <ShimmerButton 
                  className="w-full tracking-wide"
                  shimmerColor="rgba(255, 255, 255, 0.3)"
                  background="linear-gradient(to right, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05))"
                >
                  Explore Learning Path →
                </ShimmerButton>
              </Link>
            </GridPatternCardBody>
          </GridPatternCard>
        </div>
      </div>

      {/* Navigation Bar */}
      <NavBar items={navItems} />
    </div>
  );
}

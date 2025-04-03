"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { BackgroundBeams } from "@/components/ui/background-beams"
import { NexBoLogo } from "@/components/ui/nexbo-logo"
import { Button } from "@/components/ui/button"
import { ArrowLeft, BookOpen } from "lucide-react"
import { motion } from "framer-motion"
import Image from "next/image"
import { bookData } from "@/data/books"

// Extract book data from the same source as the book pages
const books = Object.values(bookData).map(book => ({
  id: book.id,
  title: book.title,
  author: book.author,
  cover: book.cover,
  description: book.description,
  lessonCount: book.lessons.length,
  progress: 0,
}))

export default function BeginnerPage() {
  const router = useRouter()
  const [userProgress, setUserProgress] = useState<Record<string, number>>({})

  // Load progress from localStorage
  useEffect(() => {
    const savedProgress = localStorage.getItem("bookProgress")
    if (savedProgress) {
      setUserProgress(JSON.parse(savedProgress))
    }
  }, [])

  return (
    <div className="min-h-screen bg-black relative overflow-hidden">
      <BackgroundBeams className="opacity-20" />
      
      {/* Navigation Bar */}
      <nav className="fixed top-0 w-full bg-black/80 backdrop-blur-xl border-b border-white/20 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              onClick={() => router.push("/learn")}
              className="group p-3 rounded-xl bg-white/10 border border-white/20 
                       hover:bg-white/20 hover:border-white/30 transition-all duration-300"
            >
              <ArrowLeft className="w-5 h-5 text-white" />
            </Button>
            <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-4">
              <NexBoLogo className="w-8 h-8" />
              <div className="hidden sm:block h-6 w-px bg-white/20" />
              <div>
                <h1 className="text-xl font-bold text-white">Beginner Resources</h1>
                <p className="text-sm text-white/80">Financial Education Platform</p>
              </div>
            </div>
            <div className="w-14" /> {/* Spacer for balance */}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-32 pb-16 px-6 max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-5xl font-bold tracking-tight mb-4 text-white
                       drop-shadow-[0_0_10px_rgba(255,255,255,0.2)]">
            Essential Reading List
          </h1>
          <p className="text-lg text-white/90 font-medium">
            Curated collection of books to build your financial foundation
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {books.map((book, index) => (
            <motion.div
              key={book.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2 }}
              whileHover={{ scale: 1.02 }}
              className="relative p-6 rounded-2xl overflow-hidden
                        bg-black/60 backdrop-blur-xl border border-white/20
                        hover:border-white/40 transition-all duration-300
                        shadow-lg group cursor-pointer"
              onClick={() => router.push(`/learn/beginner/${book.id}`)}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-white/5 to-transparent" />
              
              <div className="relative">
                {/* Book Cover */}
                <div className="relative aspect-[3/4] w-full mb-6 rounded-xl overflow-hidden
                              border border-white/20 group-hover:border-white/40
                              transition-all duration-300 shadow-lg bg-black/40">
                  {/* Fallback/Loading state */}
                  <div className="absolute inset-0 flex items-center justify-center bg-white/5">
                    <BookOpen className="w-12 h-12 text-white/20" />
                  </div>
                  
                  <Image
                    src={book.cover}
                    alt={book.title}
                    fill
                    className="relative z-10 object-contain p-2"
                    priority={index < 3}
                  />
                </div>

                {/* Book Details */}
                <h3 className="text-xl font-bold text-white mb-2 
                           drop-shadow-[0_0_8px_rgba(255,255,255,0.2)]">
                  {book.title}
                </h3>
                <p className="text-white/90 mb-3 text-sm font-medium">{book.author}</p>
                <p className="text-white/80 mb-4 text-sm line-clamp-2">{book.description}</p>

                {/* Progress info */}
                <div className="flex items-center justify-between mb-2 text-xs text-white/60">
                  <span>{book.lessonCount} lessons</span>
                  <span>{userProgress[book.id] || 0}% completed</span>
                </div>

                {/* Progress Bar */}
                <div className="w-full h-2 bg-white/10 rounded-full mb-4">
                  <div 
                    className={`h-full rounded-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-300 w-[${userProgress[book.id] || 0}%]`}
                  />
                </div>

                {/* Start Reading Button */}
                <Button
                  className="w-full py-4 text-base font-medium rounded-xl
                           bg-white/10 hover:bg-white/20 text-white
                           border border-white/20 hover:border-white/40
                           transition-all duration-300 shadow-lg
                           flex items-center justify-center gap-2
                           group-hover:bg-white/20"
                >
                  <BookOpen className="w-4 h-4" />
                  {userProgress[book.id] ? 'Continue Reading' : 'Start Reading'}
                </Button>
              </div>
            </motion.div>


          ))}        </div>      </main>
    </div>
  )
}
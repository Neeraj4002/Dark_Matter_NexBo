"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import Image from "next/image"
import { motion } from "framer-motion"
import { bookData } from "@/data/books"
import { BackgroundBeams } from "@/components/ui/background-beams"
import { NexBoLogo } from "@/components/ui/nexbo-logo"
import { Button } from "@/components/ui/button"
import { ArrowLeft, ArrowRight, BookOpen, Check, CheckCircle2 } from "lucide-react"
import { cn } from "@/lib/utils"

export default function RichDadPoorDadPage() {
  const router = useRouter()
  const [activeLesson, setActiveLesson] = useState<string | null>(null)
  const [completedLessons, setCompletedLessons] = useState<string[]>([])
  
  // Get book data
  const book = bookData["rich-dad-poor-dad"]
  const lessons = book.lessons

  // Load completed lessons from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("richDadPoorDadProgress")
    if (saved) {
      setCompletedLessons(JSON.parse(saved))
    }
  }, [])

  // Save completed lessons to localStorage when changed
  useEffect(() => {
    if (completedLessons.length > 0) {
      localStorage.setItem("richDadPoorDadProgress", JSON.stringify(completedLessons))
      
      // Update overall book progress
      const progress = Math.round((completedLessons.length / lessons.length) * 100)
      const bookProgress = JSON.parse(localStorage.getItem("bookProgress") || "{}")
      bookProgress[book.id] = progress
      localStorage.setItem("bookProgress", JSON.stringify(bookProgress))
    }
  }, [completedLessons, lessons.length, book.id])

  // Mark lesson as completed
  const markAsCompleted = (lessonId: string) => {
    if (!completedLessons.includes(lessonId)) {
      setCompletedLessons([...completedLessons, lessonId])
    }
  }

  // Navigate to next or previous lesson
  const navigateLesson = (direction: 'next' | 'prev') => {
    if (!activeLesson) {
      setActiveLesson(lessons[0].id)
      return
    }

    const currentIndex = lessons.findIndex(lesson => lesson.id === activeLesson)
    if (direction === 'next' && currentIndex < lessons.length - 1) {
      setActiveLesson(lessons[currentIndex + 1].id)
    } else if (direction === 'prev' && currentIndex > 0) {
      setActiveLesson(lessons[currentIndex - 1].id)
    }
  }

  return (
    <div className="min-h-screen bg-black relative">
      <BackgroundBeams className="opacity-20" />
      
      {/* Navigation Bar */}
      <nav className="fixed top-0 w-full bg-black/80 backdrop-blur-xl border-b border-white/20 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              onClick={() => router.push("/learn/beginner")}
              className="group p-3 rounded-xl bg-white/10 border border-white/20 
                       hover:bg-white/20 hover:border-white/30 transition-all duration-300"
            >
              <ArrowLeft className="w-5 h-5 text-white" />
            </Button>
            <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-4">
              <NexBoLogo className="w-8 h-8" />
              <div className="hidden sm:block h-6 w-px bg-white/20" />
              <div>
                <h1 className="text-xl font-bold text-white">{book.title}</h1>
                <p className="text-sm text-white/80">by {book.author}</p>
              </div>
            </div>
            
            {activeLesson && (
              <Button
                onClick={() => markAsCompleted(activeLesson)}
                className={cn(
                  "transition-all duration-300",
                  completedLessons.includes(activeLesson)
                    ? "bg-emerald-600/30 hover:bg-emerald-600/40 text-emerald-300 border border-emerald-500/30"
                    : "bg-white/10 hover:bg-white/20 text-white/90 border border-white/20"
                )}
              >
                {completedLessons.includes(activeLesson) ? (
                  <><CheckCircle2 className="w-4 h-4 mr-2" /> Completed</>
                ) : (
                  <><Check className="w-4 h-4 mr-2" /> Mark as Complete</>
                )}
              </Button>
            )}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="pt-24 pb-16 px-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row gap-8">
          {/* Sidebar */}
          <div className="w-full md:w-80 lg:w-96 flex-shrink-0">
            <div className="bg-black/60 backdrop-blur-md border border-white/20 rounded-2xl overflow-hidden sticky top-24">
              {/* Book Cover */}
              <div className="relative aspect-[3/4] w-full border-b border-white/10">
                <Image
                  src={book.cover}
                  alt={book.title}
                  fill
                  className="object-cover"
                  priority
                />
              </div>
              
              {/* Lessons List */}
              <div className="p-4">
                <h3 className="text-lg font-semibold text-white mb-4">Lessons</h3>
                <ul className="space-y-3">
                  {/* Introduction item */}
                  <li>
                    <button
                      onClick={() => setActiveLesson(null)}
                      className={cn(
                        "w-full text-left px-4 py-3 rounded-lg transition-all duration-300",
                        "flex items-center gap-3",
                        activeLesson === null
                          ? "bg-white/20 text-white"
                          : "bg-white/5 hover:bg-white/10 text-white/70 hover:text-white/90"
                      )}
                    >
                      <BookOpen className="w-5 h-5 flex-shrink-0" />
                      <span>Introduction</span>
                    </button>
                  </li>
                  
                  {/* Lesson items */}
                  {lessons.map((lesson, index) => (
                    <li key={lesson.id}>
                      <button
                        onClick={() => setActiveLesson(lesson.id)}
                        className={cn(
                          "w-full text-left px-4 py-3 rounded-lg transition-all duration-300",
                          "flex items-center gap-3",
                          activeLesson === lesson.id
                            ? "bg-white/20 text-white"
                            : "bg-white/5 hover:bg-white/10 text-white/70 hover:text-white/90"
                        )}
                      >
                        <div className="relative">
                          <span className="flex items-center justify-center w-6 h-6 rounded-full bg-white/10 text-white/80 text-xs">
                            {completedLessons.includes(lesson.id) ? (
                              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                            ) : (
                              index + 1
                            )}
                          </span>
                        </div>
                        <span>{lesson.title}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
          
          {/* Main Content Area */}
          <div className="flex-1">
            <div className="bg-black/60 backdrop-blur-md border border-white/20 rounded-2xl p-8">
              {activeLesson ? (
                // Lesson content
                <div>
                  {lessons.filter(lesson => lesson.id === activeLesson).map(lesson => (
                    <motion.div
                      key={lesson.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <h2 className="text-3xl font-bold text-white mb-6">
                        {lesson.title}
                      </h2>
                      <div className="prose prose-invert max-w-none">
                        <div className="text-white/90 text-lg leading-relaxed whitespace-pre-line">
                          {lesson.content}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                  
                  {/* Navigation buttons */}
                  <div className="mt-12 flex items-center justify-between">
                    <Button
                      onClick={() => navigateLesson('prev')}
                      className="bg-white/10 hover:bg-white/20 text-white/90 border border-white/20"
                      disabled={lessons.findIndex(l => l.id === activeLesson) === 0}
                    >
                      <ArrowLeft className="w-4 h-4 mr-2" /> Previous Lesson
                    </Button>
                    
                    <Button
                      onClick={() => navigateLesson('next')}
                      className="bg-white/10 hover:bg-white/20 text-white/90 border border-white/20"
                      disabled={lessons.findIndex(l => l.id === activeLesson) === lessons.length - 1}
                    >
                      Next Lesson <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                </div>
              ) : (
                // Introduction/Summary content
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <h2 className="text-3xl font-bold text-white mb-6">
                    Introduction to &quot;{book.title}&quot;
                  </h2>
                  
                  <div className="prose prose-invert max-w-none">
                    <p className="text-white/90 text-lg leading-relaxed">
                    </p>z
                    
                    <p className="text-white/90 text-lg leading-relaxed mt-4">
                      The book explodes the myth that you need to earn a high income to be rich and explains the difference between working for money and having your money work for you.
                    </p>
                    
                    <h3 className="text-2xl font-semibold text-white mt-8 mb-4">
                      Key Takeaways:
                    </h3>
                    
                    <ul className="space-y-3 list-disc pl-5 text-white/90">
                      <li>The rich don&apos;t work for money — they make their money work for them</li>
                      <li>Financial education is more important than academic education for financial success</li>
                      <li>Understanding the difference between assets and liabilities</li>
                      <li>How to focus on building and buying assets</li>
                      <li>The importance of entrepreneurship and investment skills</li>
                    </ul>
                    
                    <div className="mt-8 p-6 bg-white/5 border border-white/10 rounded-xl">
                      <h4 className="text-xl font-semibold text-white mb-2">Begin Your Learning Journey</h4>
                      <p className="text-white/80 mb-4">
                        Explore the valuable lessons from this book by navigating through the lessons in the sidebar.
                      </p>
                      <Button
                        onClick={() => navigateLesson('next')}
                        className="bg-gradient-to-r from-blue-600/80 to-purple-600/80 hover:from-blue-600 hover:to-purple-600 text-white border-none"
                      >
                        Start First Lesson <ArrowRight className="w-4 h-4 ml-2" />
                      </Button>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
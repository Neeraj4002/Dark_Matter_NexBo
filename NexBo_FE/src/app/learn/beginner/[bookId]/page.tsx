"use client"

import React from "react"
import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { motion } from "framer-motion"
import { bookData } from "@/data/books"
import { BackgroundBeams } from "@/components/ui/background-beams"
import { NexBoLogo } from "@/components/ui/nexbo-logo"
import { Button } from "@/components/ui/button"
import { ArrowLeft, ArrowRight, Check, CheckCircle2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { BookSidebar } from "@/components/ui/book-sidebar"
import { BookChat } from "@/components/ui/book-chat"

interface BookPageProps {
  params: {
    bookId: string;
  };
}

export default function BookPage(props: BookPageProps) {
  const router = useRouter()
  const [activeLesson, setActiveLesson] = useState<string | null>(null)
  const [completedLessons, setCompletedLessons] = useState<string[]>([])

  // Get book data
  const bookId = props.params?.bookId
  const book = bookData[bookId as keyof typeof bookData]

  const lessons = book?.lessons

  // Load completed lessons from localStorage
  useEffect(() => {
    if (!book) {
      router.push("/learn/beginner")
      return
    }

    const saved = localStorage.getItem(`${bookId}Progress`)
    if (saved) {
      try {
        setCompletedLessons(JSON.parse(saved))
      } catch (error) {
        console.error("Failed to parse saved progress:", error)
      }
    }
  }, [bookId, book, router])

  // Save completed lessons to localStorage when changed
  useEffect(() => {
    if (!book || !lessons) return

    if (completedLessons.length > 0) {
      localStorage.setItem(`${bookId}Progress`, JSON.stringify(completedLessons))

      // Update overall book progress
      const progress = Math.round((completedLessons.length / lessons.length) * 100)
      const bookProgress = JSON.parse(localStorage.getItem("bookProgress") || "{}")
      bookProgress[book.id] = progress
      localStorage.setItem("bookProgress", JSON.stringify(bookProgress))
    }
  }, [completedLessons, lessons, book, bookId])

  if (!book || !lessons) {
    return null
  }

  // Mark lesson as completed
  const markAsCompleted = (lessonId: string) => {
    if (!completedLessons.includes(lessonId)) {
      setCompletedLessons([...completedLessons, lessonId])
    }
  }

  // Navigate to next or previous lesson
  const navigateLesson = (direction: "next" | "prev") => {
    if (!activeLesson) {
      setActiveLesson(lessons[0].id)
      return
    }

    const currentIndex = lessons.findIndex((lesson) => lesson.id === activeLesson)
    if (direction === "next" && currentIndex < lessons.length - 1) {
      setActiveLesson(lessons[currentIndex + 1].id)
    } else if (direction === "prev" && currentIndex > 0) {
      setActiveLesson(lessons[currentIndex - 1].id)
    }
  }

  // Book introductions mapping
  const bookIntroductions: Record<string, string> = {
    "rich-dad-poor-dad": `"Rich Dad Poor Dad" is Robert Kiyosaki's story of growing up with two dads — his real father and the father of his best friend, his rich dad — and the ways in which both men shaped his thoughts about money and investing.

The book explodes the myth that you need to earn a high income to be rich and explains the difference between working for money and having your money work for you.`,

    "psychology-of-money": `"The Psychology of Money" by Morgan Housel explores the complex relationship humans have with money and investing. Through a collection of short stories, Housel illustrates how our personal experiences, biases, and emotions dramatically influence our financial decisions.

The book emphasizes that good financial decisions aren't necessarily about what you know, but how you behave. It demonstrates that doing well with money has little to do with how smart you are and everything to do with how you behave.`,

    "total-money-makeover": `"The Total Money Makeover" by Dave Ramsey provides a straightforward, step-by-step plan to help you get out of debt, build wealth, and achieve financial freedom. The book challenges many of the common myths about money and presents a clear path to financial fitness.

Ramsey's approach is based on personal responsibility, discipline, and changing your behaviors with money. His famous "Baby Steps" provide a roadmap that anyone can follow to transform their financial situation.`
  }

  // Book key takeaways mapping
  const bookTakeaways: Record<string, string[]> = {
    "rich-dad-poor-dad": [
      "The rich don't work for money — they make their money work for them",
      "Financial education is more important than academic education for financial success",
      "Understanding the difference between assets and liabilities",
      "How to focus on building and buying assets",
      "The importance of entrepreneurship and investment skills"
    ],

    "psychology-of-money": [
      "Your personal history with money heavily influences your decisions",
      "Luck and risk play a huge role in success and failure",
      "Getting wealthy and staying wealthy are different skills",
      "Reasonable > Rational when making financial decisions",
      "Compounding works in finance and in other areas of life"
    ],

    "total-money-makeover": [
      "Debt is not a tool for building wealth",
      "Building an emergency fund is critical before investing",
      "The debt snowball method helps create momentum for debt payoff",
      "Financial peace comes from having a plan and sticking to it",
      "Wealth building is a marathon, not a sprint"
    ]
  }

  const bookName = book?.title || "Introduction to Programming"

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
                  <>
                    <CheckCircle2 className="w-4 h-4 mr-2" /> Completed
                  </>
                ) : (
                  <>
                    <Check className="w-4 h-4 mr-2" /> Mark as Complete
                  </>
                )}
              </Button>
            )}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="pt-24 pb-16 px-4">
        <div className="max-w-7xl mx-auto flex">
          {book && lessons && (
            <BookSidebar
              book={book}
              activeLesson={activeLesson}
              completedLessons={completedLessons}
              onLessonSelect={setActiveLesson}
            />
          )}

          {/* Main Content Area */}
          <div className="flex-1 md:ml-8">
            <div className="bg-black/60 backdrop-blur-md border border-white/20 rounded-2xl p-8">
              {activeLesson ? (
                // Lesson content
                <div>
                  {lessons
                    .filter((lesson) => lesson.id === activeLesson)
                    .map((lesson) => (
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
                      onClick={() => navigateLesson("prev")}
                      className="bg-white/10 hover:bg-white/20 text-white/90 border border-white/20"
                      disabled={
                        lessons.findIndex((l) => l.id === activeLesson) === 0
                      }
                    >
                      <ArrowLeft className="w-4 h-4 mr-2" /> Previous Lesson
                    </Button>

                    <Button
                      onClick={() => navigateLesson("next")}
                      className="bg-white/10 hover:bg-white/20 text-white/90 border border-white/20"
                      disabled={
                        lessons.findIndex((l) => l.id === activeLesson) ===
                        lessons.length - 1
                      }
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
                      {(bookId && bookIntroductions[bookId]) || book.description}
                    </p>

                    <h3 className="text-2xl font-semibold text-white mt-8 mb-4">
                      Key Takeaways:
                    </h3>

                    <ul className="space-y-3 list-disc pl-5 text-white/90">
                      {(bookId && bookTakeaways[bookId] || []).map((takeaway: string, index: number) => (
                        <li key={index}>{takeaway}</li>
                      ))}
                    </ul>

                    <div className="mt-8 p-6 bg-white/5 border border-white/10 rounded-xl">
                      <h4 className="text-xl font-semibold text-white mb-2">
                        Begin Your Learning Journey
                      </h4>
                      <p className="text-white/80 mb-4">
                        Explore the valuable lessons from this book by
                        navigating through the lessons in the sidebar.
                      </p>
                      <Button
                        onClick={() => navigateLesson("next")}
                        className="bg-gradient-to-r from-blue-600/80 to-purple-600/80 hover:from-blue-600 hover:to-purple-600 text-white border-none"
                      >
                        Start First Lesson{" "}
                        <ArrowRight className="w-4 h-4 ml-2" />
                      </Button>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Add Chat Component */}
      <BookChat bookName={bookName} />
    </div>
  )
}
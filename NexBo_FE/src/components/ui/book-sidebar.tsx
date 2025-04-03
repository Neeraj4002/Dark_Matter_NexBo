"use client"

import { useState } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { ChevronLeft, Menu, BookOpen, CheckCircle2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { motion } from "framer-motion"

type BookSidebarProps = {
  book: {
    id: string
    title: string
    author: string
    cover: string
    lessons: Array<{
      id: string
      title: string
      content: string
    }>
  }
  activeLesson: string | null
  completedLessons: string[]
  onLessonSelect: (lessonId: string | null) => void
}

export function BookSidebar({ 
  book, 
  activeLesson, 
  completedLessons, 
  onLessonSelect 
}: BookSidebarProps) {
  const [isOpen, setIsOpen] = useState(true)

  return (
    <>
      {/* Mobile sidebar toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-24 left-4 z-40 md:hidden"
        onClick={() => setIsOpen(!isOpen)}
      >
        <Menu className="h-4 w-4 text-white" />
      </Button>

      <aside
        className={cn(
          "bg-black/60 backdrop-blur-xl border-r border-white/10",
          "w-[300px] flex-shrink-0 flex flex-col",
          "fixed inset-y-24 left-0 z-30 md:relative md:inset-y-0",
          "transition-all duration-300 ease-in-out",
          isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        )}
      >
        {/* Book cover and title */}
        <div className="p-6 border-b border-white/10">
          <div className="relative aspect-[3/4] w-full rounded-xl overflow-hidden mb-4">
            <Image
              src={book.cover}
              alt={book.title}
              fill
              className="object-contain"
              priority
            />
          </div>
          <h2 className="text-lg font-bold text-center text-white mb-1">
            {book.title}
          </h2>
          <p className="text-sm text-white/70 text-center">
            {book.author}
          </p>
        </div>

        {/* Lessons list */}
        <div className="flex-1 overflow-auto p-4">
          <h3 className="text-sm font-medium text-white/50 mb-3 uppercase tracking-wider px-2">
            Lessons
          </h3>
          <ul className="space-y-1">
            {/* Introduction */}
            <motion.li whileHover={{ x: 4 }} transition={{ duration: 0.2 }}>
              <li>
                <button
                  onClick={() => onLessonSelect(null)}
                  className={cn(
                    "w-full text-left px-3 py-2 rounded-lg text-sm transition-colors",
                    "flex items-center gap-3",
                    activeLesson === null
                      ? "bg-white/10 text-white"
                      : "text-white/70 hover:bg-white/5 hover:text-white"
                  )}
                >
                  <BookOpen className="w-4 h-4 flex-shrink-0" />
                  <span>Introduction</span>
                </button>
              </li>
            </motion.li>

            {/* Lessons */}
            {book.lessons.map((lesson, index) => (
              <motion.li 
                key={lesson.id}
                whileHover={{ x: 4 }}
                transition={{ duration: 0.2 }}
              >
                <li>
                  <button
                    type="button"
                    onClick={() => onLessonSelect(lesson.id)}
                    className={cn(
                      "w-full text-left px-3 py-2 rounded-lg text-sm transition-colors",
                      "flex items-center gap-3",
                      activeLesson === lesson.id
                        ? "bg-white/10 text-white"
                        : completedLessons.includes(lesson.id)
                          ? "text-emerald-400 hover:text-white"
                          : "text-white/70 hover:text-white"
                    )}
                  >
                    <span className="w-6 flex-shrink-0 flex items-center justify-center">
                      {completedLessons.includes(lesson.id) ? (
                        <CheckCircle2 className="w-4 h-4" />
                      ) : (
                        <span className="text-white/50">{index + 1}.</span>
                      )}
                    </span>
                    <span className="flex-1">{lesson.title}</span>
                  </button>
                </li>
              </motion.li>
            ))}
          </ul>
        </div>

        {/* Mobile close button */}
        <div className="p-4 border-t border-white/10 md:hidden">
          <Button
            variant="ghost"
            className="w-full justify-start text-white/70 hover:text-white"
            onClick={() => setIsOpen(false)}
          >
            <ChevronLeft className="mr-2 h-4 w-4" />
            Close Sidebar
          </Button>
        </div>
      </aside>
    </>
  )}
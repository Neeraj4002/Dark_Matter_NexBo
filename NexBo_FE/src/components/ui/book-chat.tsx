"use client"

import React, { useState, useEffect, JSX } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { MessageCircle, X, Send } from "lucide-react"
import { cn } from "@/lib/utils"

type Message = {
  id: string
  text: string
  isUser: boolean
  timestamp: Date
}

type BookChatProps = {
  bookName: string
}

// Formatter: If the AI response starts with a title marker, extract it and display as a heading.
function formatAIResponse(response: string): JSX.Element {
  // Look for a pattern starting with **title**: and capture the title text.
  const titleRegex = /^\*\*title\*\*\s*:\s*(.*?)(\n|$)/i
  const match = response.match(titleRegex)
  if (match) {
    const title = match[1].trim()
    // Remove the title part from the response
    const remaining = response.replace(titleRegex, '').trim()
    return (
      <div>
        <h2 className="text-xl font-bold text-white mb-2">{title}</h2>
        {remaining && <p className="text-sm text-white/90">{remaining}</p>}
      </div>
    )
  }
  return <p className="text-sm text-white/90">{response}</p>
}

export function BookChat({ bookName }: BookChatProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")

  // Set a welcome message that includes the book context when the bookName changes.
  useEffect(() => {
    const welcomeMessage: Message = {
      id: "welcome",
      text: `👋 Hi there! I'm your learning assistant for "${bookName}". Feel free to:\n\n` +
            "• Ask questions about this book\n" +
            "• Get clarification on concepts\n" +
            "• Request examples or explanations\n" +
            "• Get help with exercises\n\n" +
            "How can I help you today?",
      isUser: false,
      timestamp: new Date()
    }
    setMessages([welcomeMessage])
  }, [bookName])

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    console.log("📚 Console Book: ", bookName)
    console.log("💬 Console Message: ", input)

    // Add the user's message to the chat
    const userMessage: Message = {
      id: Date.now().toString(),
      text: input,
      isUser: true,
      timestamp: new Date()
    }
    setMessages(prev => [...prev, userMessage])
    setInput("")

    try {
      console.log("📚 Sending message to backend with book context...", {
        message: input,
        bookName: bookName
      })
      // Send message to the backend including the book context.
      const response = await fetch('https://dark-matter-nexbo.onrender.com/bookchat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          bookName: bookName
        })
      })
      console.log("📡 API Response status: ", response.status)
      
      const data = await response.json()
      console.log("✅ API Response: ", data)
      
      // The backend returns an object with the key "answer"
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.answer,
        isUser: false,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, aiMessage])
    } catch (error) {
      console.error('Error sending message:', error)
    }
  }

  return (
    <>
      {/* Chat Toggle Button */}
      <Button
        type="button"
        size="icon"
        className={cn(
          "fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg",
          "bg-white/10 backdrop-blur-md border border-white/20",
          "hover:bg-white/20 hover:border-white/30",
          "transition-all duration-300 z-50",
          "group"
        )}
        onClick={() => setIsOpen(prev => !prev)}
      >
        {isOpen ? (
          <X className="h-6 w-6 text-white" />
        ) : (
          <>
            <MessageCircle className="h-6 w-6 text-white" />
            {messages.length === 1 && (
              <span className="absolute -top-1 -right-1 h-3 w-3 bg-emerald-500 rounded-full group-hover:animate-pulse" />
            )}
          </>
        )}
      </Button>

      {/* Chat Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className={cn(
              "fixed bottom-24 right-6 w-96 h-[600px] rounded-2xl shadow-xl z-40",
              "bg-black/60 backdrop-blur-xl border border-white/20",
              "flex flex-col overflow-hidden"
            )}
          >
            {/* Chat Header */}
            <div className="p-4 border-b border-white/10">
              <h3 className="text-lg font-semibold text-white">Need Help?</h3>
              <p className="text-sm text-white/70">Ask questions about this lesson</p>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.length === 0 ? (
                <div className="flex items-center justify-center h-full">
                  <p className="text-white/50 text-sm text-center">
                    Start a conversation by asking a question about the lesson
                  </p>
                </div>
              ) : (
                messages.map(message => (
                  <div
                    key={message.id}
                    className={cn(
                      "flex flex-col max-w-[80%] rounded-xl p-3",
                      message.isUser
                        ? "ml-auto bg-white/10 border border-white/20"
                        : "bg-white/5 border border-white/10"
                    )}
                  >
                    {message.isUser ? (
                      <p className="text-sm text-white/90">{message.text}</p>
                    ) : (
                      // Use the formatter for AI messages
                      formatAIResponse(message.text)
                    )}
                    <span className="text-xs text-white/40 mt-1">
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                ))
              )}
            </div>

            {/* Chat Input */}
            <form onSubmit={sendMessage} className="p-4 border-t border-white/10">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  placeholder="Type your message..."
                  className={cn(
                    "flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-2",
                    "text-white placeholder:text-white/40",
                    "focus:outline-none focus:ring-2 focus:ring-white/20"
                  )}
                />
                <Button
                  type="submit"
                  size="icon"
                  disabled={!input.trim()}
                  className={cn(
                    "bg-white/10 border border-white/20",
                    "hover:bg-white/20 hover:border-white/30",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                >
                  <Send className="h-4 w-4 text-white" />
                </Button>
              </div>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

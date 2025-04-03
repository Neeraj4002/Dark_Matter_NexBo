'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ShimmerButton } from "@/components/ui/shimmer-button"
import { auth, signInWithGoogle, resetPassword } from '@/lib/firebase'
import { createUserWithEmailAndPassword, signInWithEmailAndPassword } from 'firebase/auth'
import Image from 'next/image'
import { ArrowLeft } from 'lucide-react'

interface AuthOverlayProps {
  isOpen: boolean
  onClose: () => void
  initialView?: 'signin' | 'signup'
}

export function AuthOverlay({ isOpen, onClose, initialView = 'signin' }: AuthOverlayProps) {
  const [view, setView] = useState(initialView)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [rememberMe, setRememberMe] = useState(false)
  const [showForgotPassword, setShowForgotPassword] = useState(false)
  const [resetSent, setResetSent] = useState(false)

  const validateForm = () => {
    if (!email || !password) {
      setError('Please fill in all fields')
      return false
    }
    if (view === 'signup' && !fullName) {
      setError('Please enter your full name')
      return false
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters')
      return false
    }
    return true
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!validateForm()) return

    setIsLoading(true)
    try {
      if (view === 'signup') {
        await createUserWithEmailAndPassword(auth, email, password)
      } else {
        await signInWithEmailAndPassword(auth, email, password)
        if (rememberMe) {
          localStorage.setItem('rememberMe', 'true')
        }
      }
      window.location.href = '/home'
    } catch (err: any) {
      const errorMessage = err.code === 'auth/invalid-credential' 
        ? 'Invalid email or password'
        : err.code === 'auth/email-already-in-use'
        ? 'Email already in use. Please sign in instead.'
        : 'Authentication failed. Please try again.'
      setError(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }

  const handleGoogleSignIn = async () => {
    setError('')
    setIsLoading(true)
    try {
      const { user, error } = await signInWithGoogle()
      if (error) throw new Error(error)
      if (user) window.location.href = '/home'
    } catch (err: any) {
      setError('Google sign in failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleForgotPassword = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!email) {
      setError('Please enter your email address')
      return
    }
    setIsLoading(true)
    try {
      const { error } = await resetPassword(email)
      if (error) throw new Error(error)
      setResetSent(true)
    } catch (err) {
      setError('Failed to send reset email. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleBack = () => {
    if (showForgotPassword) {
      setShowForgotPassword(false)
      setResetSent(false)
    } else {
      onClose()
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
          />

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div className="relative w-full max-w-md overflow-hidden rounded-2xl bg-black/40 p-8 shadow-xl border border-white/10">
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent" />
              
              {/* Back Button */}
              <button
                onClick={handleBack}
                className="absolute top-4 left-4 p-2 text-white/60 hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>

              <div className="relative space-y-6">
                {/* Title */}
                <div className="text-center">
                  <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-200 via-slate-100 to-slate-200">
                    {showForgotPassword ? 'Reset Password' : view === 'signin' ? 'Welcome Back' : 'Create Account'}
                  </h2>
                  <p className="mt-2 text-sm text-slate-400">
                    {showForgotPassword 
                      ? 'Enter your email to receive a reset link'
                      : view === 'signin' 
                      ? 'Sign in to continue your journey' 
                      : 'Join NexBo to start your investment journey'}
                  </p>
                </div>

                {/* Google Sign In Button */}
                {!showForgotPassword && (
                  <ShimmerButton
                    onClick={handleGoogleSignIn}
                    disabled={isLoading}
                    className="w-full py-3 flex items-center justify-center space-x-3"
                    background="rgba(255, 255, 255, 0.05)"
                  >
                    <Image
                      src="/google.svg"
                      alt="Google"
                      width={20}
                      height={20}
                    />
                    <span className="text-white">
                      Continue with Google
                    </span>
                  </ShimmerButton>
                )}

                {/* Divider */}
                {!showForgotPassword && (
                  <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                      <div className="w-full border-t border-white/10"></div>
                    </div>
                    <div className="relative flex justify-center text-sm">
                      <span className="px-2 text-slate-400 bg-black/40">or</span>
                    </div>
                  </div>
                )}

                {/* Form */}
                <form onSubmit={showForgotPassword ? handleForgotPassword : handleSubmit} className="space-y-4">
                  {view === 'signup' && !showForgotPassword && (
                    <input
                      type="text"
                      placeholder="Full Name"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-400 text-white"
                    />
                  )}
                  <input
                    type="email"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-400 text-white"
                  />
                  {!showForgotPassword && (
                    <input
                      type="password"
                      placeholder="Password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-400 text-white"
                    />
                  )}

                  {error && (
                    <p className="text-sm text-red-400 text-center">{error}</p>
                  )}

                  {resetSent && (
                    <p className="text-sm text-green-400 text-center">
                      Password reset link sent! Check your email.
                    </p>
                  )}

                  {!showForgotPassword && view === 'signin' && (
                    <div className="flex items-center justify-between text-sm">
                      <label className="flex items-center space-x-2 text-slate-400">
                        <input
                          type="checkbox"
                          checked={rememberMe}
                          onChange={(e) => setRememberMe(e.target.checked)}
                          className="rounded border-white/10 bg-white/5"
                        />
                        <span>Remember me</span>
                      </label>
                      <button
                        type="button"
                        onClick={() => setShowForgotPassword(true)}
                        className="text-slate-400 hover:text-slate-300"
                      >
                        Forgot password?
                      </button>
                    </div>
                  )}

                  <ShimmerButton
                    type="submit"
                    disabled={isLoading}
                    className="w-full py-3"
                  >
                    {isLoading ? (
                      <div className="flex items-center justify-center">
                        <div className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                      </div>
                    ) : (
                      <span className="text-white">
                        {showForgotPassword ? 'Send Reset Link' : view === 'signin' ? 'Sign In' : 'Create Account'}
                      </span>
                    )}
                  </ShimmerButton>

                  {/* Toggle View */}
                  {!showForgotPassword && (
                    <div className="text-center">
                      <button
                        type="button"
                        onClick={() => {
                          setView(view === 'signin' ? 'signup' : 'signin')
                          setError('')
                        }}
                        className="text-sm text-slate-400 hover:text-slate-300 transition-colors"
                      >
                        {view === 'signin' 
                          ? "Don't have an account? Sign up"
                          : 'Already have an account? Sign in'}
                      </button>
                    </div>
                  )}
                </form>
              </div>

              <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent" />
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
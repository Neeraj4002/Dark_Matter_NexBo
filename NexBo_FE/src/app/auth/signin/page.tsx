'use client'

import { useState, useEffect } from 'react'
import { auth, signInWithEmailAndPassword } from '@/lib/firebase'
import { ShimmerButton } from "@/components/ui/shimmer-button"
import Link from 'next/link'

export default function SignIn() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!mounted) return

    if (!email || !password) {
      setError('Please fill all fields')
      return
    }
    
    setError('')
    setLoading(true)
    
    try {
      await signInWithEmailAndPassword(auth, email, password)
      // No redirect for now
    } catch (err: unknown) {
      console.error("Sign in error:", err)
      setError('Failed to sign in. Please check your credentials.')
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen w-full bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-8 bg-black/40 p-8 rounded-2xl border border-white/10">
        <div className="text-center space-y-2">
          <h2 className="text-3xl md:text-4xl font-bold bg-[linear-gradient(45deg,#c0c0c0,#ffffff,#8f8f8f,#ffffff)] bg-clip-text text-transparent bg-[length:200%] animate-gradient">
            Welcome Back
          </h2>
          <p className="text-sm text-slate-400">Sign in to continue your journey</p>
        </div>

        {error && <p className="text-red-500 text-sm text-center">{error}</p>}

        <form className="space-y-6" onSubmit={handleSignIn}>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full px-4 py-3 bg-black/40 border border-white/10 rounded-lg 
            focus:outline-none focus:ring-2 focus:ring-slate-400
            text-white caret-white font-medium"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full px-4 py-3 bg-black/40 border border-white/10 rounded-lg 
            focus:outline-none focus:ring-2 focus:ring-slate-400
            text-white caret-white font-medium"
          />
          
          <ShimmerButton 
            type="submit"
            className="w-full" 
            disabled={loading}
          >
            <span className="text-lg font-medium text-[#F0F0F0]">
              {loading ? 'Signing In...' : 'Sign In'}
            </span>
          </ShimmerButton>
        </form>

        <p className="text-center text-sm text-slate-400">
          Don&apos;t have an account?{' '}
          <Link href="/auth/signup" className="text-[#c0c0c0] hover:text-white underline">
            Sign Up
          </Link>
        </p>
      </div>
    </div>
  )
}
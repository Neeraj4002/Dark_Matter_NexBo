import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { AuthProvider } from '@/context/auth-context'
import { ThemeProvider } from "@/context/theme-context";

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'NexBo - Your AI Financial Assistant',
  description: 'Get personalized financial advice and investment guidance with NexBo',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthProvider>
          <ThemeProvider>
            {children}
          </ThemeProvider>
        </AuthProvider>
      </body>
    </html>
  )
}

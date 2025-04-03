/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  safelist: [
    'text-emerald-400',
    'text-amber-400',
    'text-rose-400',
    'border-emerald-500/30',
    'border-amber-500/30',
    'border-rose-500/30',
    'bg-emerald-500/20',
    'bg-amber-500/20',
    'bg-rose-500/20',
    'hover:border-emerald-500/50',
    'hover:border-amber-500/50',
    'hover:border-rose-500/50',
    'hover:shadow-emerald-500/20',
    'hover:shadow-amber-500/20',
    'hover:shadow-rose-500/20',
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        pearl: {
          50: 'var(--pearl-white)',
          100: '#fefefe',
          200: '#fdfdfd',
        },
        steel: {
          100: '#e6e8ea',
          200: '#c8ccd0',
        },
        ivory: {
          50: '#fffffb',
          100: '#fffff7',
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: 0 },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: 0 },
        },
        "shimmer-slide": {
          to: {
            transform: "translate(calc(100vw - 100%), 0)",
          },
        },
        "spin-around": {
          "0%": {
            transform: "translateZ(0) rotate(0)",
          },
          "15%, 35%": {
            transform: "translateZ(0) rotate(90deg)",
          },
          "65%, 85%": {
            transform: "translateZ(0) rotate(270deg)",
          },
          "100%": {
            transform: "translateZ(0) rotate(360deg)",
          },
        },
        'gradient': {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        scroll: {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-50%)' },
        },
        blink: {
          '0%, 100%': { 
            opacity: '1',
            textShadow: '0 0 20px rgba(255, 255, 255, 0.8)',
            filter: 'brightness(1.2)'
          },
          '50%': { 
            opacity: '0.85',
            textShadow: '0 0 10px rgba(255, 255, 255, 0.5)',
            filter: 'brightness(1)'
          },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "shimmer-slide": "shimmer-slide var(--speed) ease-in-out infinite alternate",
        "spin-around": "spin-around calc(var(--speed) * 2) infinite linear",
        'gradient': 'gradient 8s linear infinite',
        scroll: 'scroll 25s linear infinite',
        blink: 'blink 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      backgroundImage: {
        'square-pattern': `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 800'%3E%3Cg stroke-width='3.5' stroke='hsla(0, 0%25, 100%25, 1.00)' fill='none'%3E%3Crect width='400' height='400' x='0' y='0'%3E%3C/rect%3E%3Crect width='400' height='400' x='400' y='0'%3E%3C/rect%3E%3Crect width='400' height='400' x='800' y='0'%3E%3C/rect%3E%3Crect width='400' height='400' x='0' y='400'%3E%3C/rect%3E%3Crect width='400' height='400' x='400' y='400'%3E%3C/rect%3E%3Crect width='400' height='400' x='800' y='400'%3E%3C/rect%3E%3Crect width='400' height='400' x='0' y='800'%3E%3C/rect%3E%3Crect width='400' height='400' x='400' y='800'%3E%3C/rect%3E%3Crect width='400' height='400' x='800' y='800'%3E%3C/rect%3E%3C/g%3E%3C/svg%3E")`,
        'square-pattern-light': `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 800'%3E%3Cg stroke-width='3.5' stroke='hsla(215, 16%25, 47%25, 1.00)' fill='none'%3E%3Crect width='400' height='400' x='0' y='0'%3E%3C/rect%3E%3Crect width='400' height='400' x='400' y='0'%3E%3C/rect%3E%3Crect width='400' height='400' x='800' y='0'%3E%3C/rect%3E%3Crect width='400' height='400' x='0' y='400'%3E%3C/rect%3E%3Crect width='400' height='400' x='400' y='400'%3E%3C/rect%3E%3Crect width='400' height='400' x='800' y='400'%3E%3C/rect%3E%3Crect width='400' height='400' x='0' y='800'%3E%3C/rect%3E%3Crect width='400' height='400' x='400' y='800'%3E%3C/rect%3E%3Crect width='400' height='400' x='800' y='800'%3E%3C/rect%3E%3C/g%3E%3C/svg%3E")`,
        "grid-pattern": `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 800'%3E%3Cg stroke-width='3.5' stroke='hsla(0, 0%25, 100%25, 1.00)' fill='none'%3E%3Crect width='400' height='400' x='0' y='0' opacity='0.15'%3E%3C/rect%3E%3Ccircle r='10.85' cx='0' cy='0' fill='hsla(0, 0%25, 100%25, 1.00)' stroke='none'%3E%3C/circle%3E%3C/g%3E%3C/svg%3E")`,
        "grid-pattern-light": `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 800'%3E%3Cg stroke-width='3.5' stroke='hsla(215, 16%25, 47%25, 1.00)' fill='none'%3E%3Crect width='400' height='400' x='0' y='0' opacity='0.15'%3E%3C/rect%3E%3Ccircle r='10.85' cx='0' cy='0' fill='hsla(215, 16%25, 47%25, 1.00)' stroke='none'%3E%3C/circle%3E%3C/g%3E%3C/svg%3E")`,
      },
    },
  },
  plugins: [import('@tailwindcss/typography'),
    import("tailwindcss-animate"),
    ,
  ],
};
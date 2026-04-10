/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#101014',
        surface: '#1b1b22',
        'surface-alt': '#25252e',
        text: '#e8e6e3',
        'text-muted': '#8a8a96',
        accent: '#c4a35a',
        'accent-dim': '#9e8347',
        mint: '#5ae6c4',
        rose: '#e6556f',
        highlight: '#f0e6cf',
      },
      fontFamily: {
        title: ['"JetBrains Mono"', 'monospace'],
        body: ['"DM Sans"', 'sans-serif'],
      },
      boxShadow: {
        glow: '0 0 32px rgba(196, 163, 90, 0.12)',
        'glow-mint': '0 0 24px rgba(90, 230, 196, 0.1)',
        card: '0 4px 24px rgba(0, 0, 0, 0.3)',
        'card-hover': '0 8px 32px rgba(0, 0, 0, 0.5)',
      },
    },
  },
  plugins: [],
}

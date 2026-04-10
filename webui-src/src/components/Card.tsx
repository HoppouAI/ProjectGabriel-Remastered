interface CardProps {
  children: React.ReactNode
  className?: string
  hover?: boolean
}

export default function Card({ children, className = '', hover = false }: CardProps) {
  return (
    <div className={`bg-surface border border-white/[0.06] rounded-xl ${
      hover ? 'transition-all hover:-translate-y-0.5 hover:shadow-card-hover' : ''
    } ${className}`}>
      {children}
    </div>
  )
}

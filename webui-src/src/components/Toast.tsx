import { AnimatePresence, motion } from 'framer-motion'

export interface ToastItem {
  id: number
  message: string
  level: string
}

const levelColors: Record<string, string> = {
  success: 'bg-mint/10 border-mint/30 text-mint',
  error: 'bg-rose/10 border-rose/30 text-rose',
  warning: 'bg-accent/10 border-accent/30 text-accent',
  info: 'bg-surface border-white/10 text-text',
}

export default function Toast({ toasts }: { toasts: ToastItem[] }) {
  return (
    <div className="fixed top-16 right-4 z-[100] flex flex-col gap-2 max-w-sm">
      <AnimatePresence>
        {toasts.map(t => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 40 }}
            className={`px-4 py-2.5 rounded-lg border text-sm font-body shadow-card ${
              levelColors[t.level] || levelColors.info
            }`}
          >
            {t.message}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}

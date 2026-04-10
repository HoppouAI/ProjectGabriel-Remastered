interface ModalProps {
  open: boolean
  onClose: () => void
  title: string
  children: React.ReactNode
}

export default function Modal({ open, onClose, title, children }: ModalProps) {
  if (!open) return null
  return (
    <div
      className="fixed inset-0 z-[90] flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="bg-surface border border-white/[0.06] rounded-xl w-full max-w-lg mx-4 shadow-card-hover animate-[fadeIn_0.15s_ease-out]">
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.06]">
          <h3 className="font-title text-base font-semibold text-text">{title}</h3>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text transition-colors text-xl leading-none"
          >
            &times;
          </button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  )
}

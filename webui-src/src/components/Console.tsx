import { useRef, useEffect, useMemo } from 'react'
import type { ConsoleEntry } from '../lib/types'

const typeStyles: Record<string, string> = {
  transcription: 'text-mint',
  response: 'text-text',
  thinking: 'text-accent-dim italic',
  tool_call: 'text-accent',
  tool_response: 'text-accent-dim',
  error: 'text-rose',
  info: 'text-text-muted',
  warn: 'text-accent',
  system: 'text-accent-dim',
}

const typeLabels: Record<string, string> = {
  transcription: 'USER',
  response: 'AI',
  thinking: 'THINK',
  tool_call: 'TOOL',
  tool_response: 'RESULT',
  error: 'ERROR',
  info: 'INFO',
  warn: 'WARN',
  system: 'SYS',
}

interface ConsoleProps {
  logs: ConsoleEntry[]
}

export default function Console({ logs }: ConsoleProps) {
  const endRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const autoScrollRef = useRef(true)

  const handleScroll = () => {
    const el = containerRef.current
    if (!el) return
    autoScrollRef.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 40
  }

  useEffect(() => {
    if (autoScrollRef.current) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs])

  const rendered = useMemo(() => logs.map((entry, i) => {
    const style = typeStyles[entry.type] || 'text-text-muted'
    const label = typeLabels[entry.type] || entry.type.toUpperCase()
    return (
      <div key={i} className={`font-title text-[13px] leading-relaxed ${style}`}>
        <span className="text-text-muted/60 mr-2 text-xs select-none">[{label}]</span>
        <span className="break-words whitespace-pre-wrap">{entry.content}</span>
      </div>
    )
  }), [logs])

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="console-scroll h-[420px] bg-background/60 rounded-lg border border-white/[0.04] p-3 space-y-0.5"
    >
      {rendered.length === 0 && (
        <div className="text-text-muted text-sm italic">Waiting for activity...</div>
      )}
      {rendered}
      <div ref={endRef} />
    </div>
  )
}

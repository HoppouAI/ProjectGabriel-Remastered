import { useRef, useEffect, useMemo, useState, useCallback } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ConsoleEntry } from '../lib/types'
import {
  TbBrain, TbTool, TbChevronDown, TbChevronRight,
  TbInfoCircle, TbAlertTriangle, TbX, TbSettings,
} from 'react-icons/tb'

/* ── Grouped message model ─────────────────────── */

type AssistantRole = 'assistant'
type UserRole = 'user'
type SystemRole = 'system'

interface ToolCall { name: string; args: string; result?: string }

interface AssistantMsg {
  role: AssistantRole
  thinking: string
  toolCalls: ToolCall[]
  response: string
  streaming: boolean
}

interface UserMsg { role: UserRole; content: string }
interface SystemMsg { role: SystemRole; content: string; level: 'info' | 'error' | 'warn' | 'system' }

type ChatMsg = AssistantMsg | UserMsg | SystemMsg

const ASSISTANT_TYPES = new Set(['thinking', 'tool_call', 'tool_response', 'response'])
const SYSTEM_TYPES = new Set(['info', 'error', 'warn', 'system'])

function groupLogs(logs: ConsoleEntry[]): ChatMsg[] {
  const messages: ChatMsg[] = []
  // Use array wrapper so TS doesn't narrow to 'never' through closures
  const ctx: { cur: AssistantMsg | null } = { cur: null }

  const flushAssistant = () => {
    if (ctx.cur) {
      messages.push(ctx.cur)
      ctx.cur = null
    }
  }

  const ensureAssistant = (): AssistantMsg => {
    if (!ctx.cur) {
      ctx.cur = { role: 'assistant', thinking: '', toolCalls: [], response: '', streaming: false }
    }
    return ctx.cur
  }

  for (let i = 0; i < logs.length; i++) {
    const entry = logs[i]
    const isLast = i === logs.length - 1

    if (entry.type === 'transcription') {
      flushAssistant()
      // Merge consecutive transcription entries
      const last = messages[messages.length - 1]
      if (last?.role === 'user') {
        last.content += entry.content
      } else {
        messages.push({ role: 'user', content: entry.content })
      }
    } else if (entry.type === 'thinking') {
      if (ctx.cur?.response) flushAssistant()
      ensureAssistant().thinking += entry.content
      if (isLast) ctx.cur!.streaming = true
    } else if (entry.type === 'tool_call') {
      if (ctx.cur?.response) flushAssistant()
      const match = entry.content.match(/^(\w+)\((.*)?\)$/s)
      const name = match?.[1] || entry.content
      const args = match?.[2] || ''
      ensureAssistant().toolCalls.push({ name, args })
    } else if (entry.type === 'tool_response') {
      const a = ensureAssistant()
      const pending = [...a.toolCalls].reverse().find((tc: ToolCall) => !tc.result)
      if (pending) {
        const arrowIdx = entry.content.indexOf('→')
        pending.result = arrowIdx >= 0 ? entry.content.slice(arrowIdx + 1).trim() : entry.content
      }
    } else if (entry.type === 'response') {
      const a = ensureAssistant()
      a.response += entry.content
      if (isLast) a.streaming = true
    } else if (SYSTEM_TYPES.has(entry.type)) {
      flushAssistant()
      messages.push({
        role: 'system',
        content: entry.content,
        level: entry.type as SystemMsg['level'],
      })
    }
  }

  flushAssistant()
  return messages
}

/* ── Sub-components ─────────────────────────────── */

function ThinkingBlock({ text, streaming }: { text: string; streaming: boolean }) {
  const [expanded, setExpanded] = useState(false)

  if (!text) return null

  return (
    <div className="mb-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-accent-dim hover:text-accent transition-colors group"
      >
        <TbBrain size={14} className={streaming ? 'animate-pulse' : ''} />
        <span className={streaming ? 'shimmer' : ''}>{streaming ? 'Thinking...' : 'Thought process'}</span>
        {expanded ? <TbChevronDown size={12} /> : <TbChevronRight size={12} />}
      </button>
      {expanded && (
        <div className="mt-1.5 ml-5 pl-3 border-l-2 border-accent/15 text-xs text-text-muted/70 leading-relaxed whitespace-pre-wrap">
          {text}
        </div>
      )}
    </div>
  )
}

function ToolCallBlock({ calls }: { calls: ToolCall[] }) {
  const [expanded, setExpanded] = useState(false)

  if (calls.length === 0) return null

  const allDone = calls.every(tc => tc.result !== undefined)
  const label = allDone
    ? `Used ${calls.length} tool${calls.length > 1 ? 's' : ''}`
    : `Calling tool...`

  return (
    <div className="mb-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-accent/80 hover:text-accent transition-colors"
      >
        <TbTool size={14} className={!allDone ? 'animate-spin-slow' : ''} />
        <span className={!allDone ? 'shimmer' : ''}>{label}</span>
        {expanded ? <TbChevronDown size={12} /> : <TbChevronRight size={12} />}
      </button>
      {expanded && (
        <div className="mt-1.5 ml-5 space-y-1.5">
          {calls.map((tc, i) => (
            <div key={i} className="text-xs border-l-2 border-accent/20 pl-3 py-1">
              <div className="flex items-center gap-1.5">
                <span className="text-accent font-title font-medium">{tc.name}</span>
                {tc.args && (
                  <span className="text-text-muted/50 truncate max-w-[200px]">({tc.args})</span>
                )}
              </div>
              {tc.result !== undefined && (
                <div className="mt-0.5 text-text-muted/60 whitespace-pre-wrap break-words">
                  → {tc.result}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function MarkdownContent({ text }: { text: string }) {
  return (
    <div className="prose-chat text-sm text-text leading-relaxed break-words">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">{children}</a>
          ),
          code: ({ className, children }) => {
            const isBlock = className?.startsWith('language-')
            if (isBlock) {
              return (
                <pre className="my-2 bg-background/80 border border-white/[0.06] rounded-lg px-3 py-2 overflow-x-auto text-xs">
                  <code className="text-text/90">{children}</code>
                </pre>
              )
            }
            return <code className="bg-white/[0.06] px-1 py-0.5 rounded text-[13px] text-accent/90">{children}</code>
          },
          ul: ({ children }) => <ul className="list-disc pl-4 mb-2 last:mb-0 space-y-0.5">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 last:mb-0 space-y-0.5">{children}</ol>,
          li: ({ children }) => <li className="text-text/90">{children}</li>,
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-accent/30 pl-3 my-2 text-text-muted italic">{children}</blockquote>
          ),
          h1: ({ children }) => <h1 className="text-lg font-semibold text-text mb-2 mt-3">{children}</h1>,
          h2: ({ children }) => <h2 className="text-base font-semibold text-text mb-1.5 mt-2">{children}</h2>,
          h3: ({ children }) => <h3 className="text-sm font-semibold text-text mb-1 mt-2">{children}</h3>,
          table: ({ children }) => (
            <div className="my-2 overflow-x-auto"><table className="text-xs border border-white/[0.08] rounded">{children}</table></div>
          ),
          th: ({ children }) => <th className="px-2 py-1 bg-surface text-left text-text-muted font-title border-b border-white/[0.08]">{children}</th>,
          td: ({ children }) => <td className="px-2 py-1 border-b border-white/[0.04] text-text/80">{children}</td>,
        }}
      >
        {text}
      </Markdown>
    </div>
  )
}

const systemIcons: Record<string, React.ReactNode> = {
  info: <TbInfoCircle size={13} />,
  error: <TbX size={13} />,
  warn: <TbAlertTriangle size={13} />,
  system: <TbSettings size={13} />,
}

const systemColors: Record<string, string> = {
  info: 'text-text-muted/50 border-white/[0.04]',
  error: 'text-rose/70 border-rose/10',
  warn: 'text-accent/60 border-accent/10',
  system: 'text-accent-dim/60 border-accent/10',
}

/* ── Main Console ───────────────────────────────── */

interface ConsoleProps {
  logs: ConsoleEntry[]
}

export default function Console({ logs }: ConsoleProps) {
  const endRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const autoScrollRef = useRef(true)

  const handleScroll = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    autoScrollRef.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 60
  }, [])

  useEffect(() => {
    if (autoScrollRef.current) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs])

  const messages = useMemo(() => groupLogs(logs), [logs])

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="console-scroll h-full overflow-y-auto py-4 space-y-4"
    >
      {messages.length === 0 && (
        <div className="flex flex-col items-center justify-center h-full gap-2 text-text-muted/30">
          <TbBrain size={32} />
          <span className="text-sm">Waiting for activity...</span>
        </div>
      )}

      {messages.map((msg, i) => {
        if (msg.role === 'user') {
          return (
            <div key={i} className="flex justify-end">
              <div className="max-w-[80%] bg-mint/10 border border-mint/15 rounded-2xl rounded-br-md px-4 py-2.5">
                <p className="text-sm text-mint/90 whitespace-pre-wrap break-words">{msg.content}</p>
              </div>
            </div>
          )
        }

        if (msg.role === 'system') {
          return (
            <div key={i} className="flex justify-center px-4">
              <div className={`flex items-center gap-1.5 text-[11px] border rounded-full px-3 py-1 ${systemColors[msg.level]}`}>
                {systemIcons[msg.level]}
                <span className="truncate max-w-[400px]">{msg.content}</span>
              </div>
            </div>
          )
        }

        // Assistant message
        const a = msg as AssistantMsg
        return (
          <div key={i} className="flex gap-3">
            {/* Avatar */}
            <div className="shrink-0 mt-1">
              <div className="w-7 h-7 rounded-full bg-accent/15 flex items-center justify-center">
                <TbBrain size={15} className="text-accent" />
              </div>
            </div>
            {/* Content */}
            <div className="flex-1 min-w-0">
              <span className="text-xs font-title text-text-muted/40 mb-1 block">Gabriel</span>
              <ThinkingBlock text={a.thinking} streaming={a.streaming && !!a.thinking && !a.response} />
              <ToolCallBlock calls={a.toolCalls} />
              {a.response ? (
                <MarkdownContent text={a.response} />
              ) : a.streaming && !a.thinking && a.toolCalls.length === 0 ? (
                <div className="flex items-center gap-1.5 text-text-muted/40 text-sm">
                  <span className="inline-flex gap-0.5">
                    <span className="w-1.5 h-1.5 bg-text-muted/40 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-1.5 h-1.5 bg-text-muted/40 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-1.5 h-1.5 bg-text-muted/40 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </span>
                </div>
              ) : null}
            </div>
          </div>
        )
      })}

      <div ref={endRef} />
    </div>
  )
}

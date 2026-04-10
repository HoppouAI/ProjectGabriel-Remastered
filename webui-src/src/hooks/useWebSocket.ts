import { useEffect, useRef, useCallback, useState } from 'react'
import type { AppState, ConsoleEntry } from '../lib/types'

interface WSMessage {
  type: 'state' | 'log' | 'toast'
  data?: AppState | ConsoleEntry
  message?: string
  level?: string
}

const SKIP_LOG_TYPES = new Set(['music_update', 'turn_complete', 'user_turn_complete'])

export function useWebSocket(onToast: (msg: string, level: string) => void) {
  const [state, setState] = useState<AppState | null>(null)
  const [logs, setLogs] = useState<ConsoleEntry[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>(null)

  const connect = useCallback(() => {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${proto}//${location.host}/ws`)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      setLogs(prev => [...prev, { type: 'info', content: 'WebSocket connected' }])
    }

    ws.onmessage = (event) => {
      const msg: WSMessage = JSON.parse(event.data)
      if (msg.type === 'state') {
        setState(msg.data as AppState)
      } else if (msg.type === 'toast') {
        onToast(msg.message || '', msg.level || 'info')
      } else if (msg.type === 'log') {
        const entry = msg.data as ConsoleEntry
        if (!SKIP_LOG_TYPES.has(entry.type)) {
          setLogs(prev => {
            const extra = entry.extra as Record<string, unknown> | undefined
            // Handle streaming entries
            if (extra?.streaming && prev.length > 0) {
              const last = prev[prev.length - 1]
              if (last.type === entry.type) {
                return [
                  ...prev.slice(0, -1),
                  { ...last, content: last.content + entry.content },
                ]
              }
            }
            const next = [...prev, entry]
            return next.length > 300 ? next.slice(-300) : next
          })
        }
        // Forward music updates to state
        if (entry.type === 'music_update') {
          setState(prev => {
            if (!prev) return prev
            const extra = entry.extra as Record<string, unknown> | undefined
            if (!extra) return prev
            return {
              ...prev,
              music_progress: {
                is_playing: extra.playing as boolean,
                song_name: (extra.song_name as string) || null,
                position: (extra.position as number) || 0,
                duration: (extra.duration as number) || 0,
              },
            }
          })
        }
      }
    }

    ws.onclose = () => {
      setConnected(false)
      setLogs(prev => [...prev, { type: 'error', content: 'WebSocket disconnected' }])
      reconnectRef.current = setTimeout(connect, 2500)
    }
  }, [onToast])

  useEffect(() => {
    connect()
    return () => {
      wsRef.current?.close()
      if (reconnectRef.current) clearTimeout(reconnectRef.current)
    }
  }, [connect])

  const clearLogs = useCallback(() => {
    setLogs([{ type: 'info', content: 'Console cleared' }])
  }, [])

  return { state, logs, connected, clearLogs }
}

import { useEffect, useRef, useCallback, useState } from 'react'
import type { AppState, ConsoleEntry } from '../lib/types'

interface WSMessage {
  type: 'state' | 'log' | 'toast'
  data?: AppState | ConsoleEntry
  message?: string
  level?: string
}

const SKIP_LOG_TYPES = new Set(['music_update', 'turn_complete', 'user_turn_complete'])

// reconnect tuning. exponential backoff with cap + jitter so we dont hammer
// a downed server, but recover fast when its back.
const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 15000
// client ping cadence. backend just does receive_text in a loop so it'll
// silently swallow these. we mostly care about the send() throwing when
// the socket is actually dead (wifi switched, NAT timeout, laptop slept).
const HEARTBEAT_MS = 25000
// if we havent heard ANYTHING from the server for this long, assume the
// socket is half open and force a reconnect. state broadcast pushes every
// ~1s so 40s of silence is definitively dead.
const LIVENESS_TIMEOUT_MS = 40000

export function useWebSocket(onToast: (msg: string, level: string) => void) {
  const [state, setState] = useState<AppState | null>(null)
  const [logs, setLogs] = useState<ConsoleEntry[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const heartbeatRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const livenessRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const lastMessageRef = useRef<number>(Date.now())
  const attemptsRef = useRef<number>(0)
  const mountedRef = useRef<boolean>(true)
  // stash onToast in a ref so connect() doesnt depend on it. parent re-renders
  // with new onToast identity would otherwise tear down + recreate the socket.
  const onToastRef = useRef(onToast)
  useEffect(() => { onToastRef.current = onToast }, [onToast])

  const clearTimers = useCallback(() => {
    if (reconnectRef.current) { clearTimeout(reconnectRef.current); reconnectRef.current = null }
    if (heartbeatRef.current) { clearInterval(heartbeatRef.current); heartbeatRef.current = null }
    if (livenessRef.current) { clearInterval(livenessRef.current); livenessRef.current = null }
  }, [])

  const scheduleReconnect = useCallback((connectFn: () => void) => {
    if (!mountedRef.current) return
    if (reconnectRef.current) return
    const n = attemptsRef.current
    const base = Math.min(RECONNECT_BASE_MS * Math.pow(2, n), RECONNECT_MAX_MS)
    // +/- 25% jitter so a swarm of tabs doesnt all reconnect on the same tick
    const jitter = base * 0.25 * (Math.random() * 2 - 1)
    const delay = Math.max(250, base + jitter)
    attemptsRef.current = Math.min(n + 1, 10)
    reconnectRef.current = setTimeout(() => {
      reconnectRef.current = null
      connectFn()
    }, delay)
  }, [])

  const forceReconnect = useCallback((reason: string) => {
    const ws = wsRef.current
    if (ws && ws.readyState !== WebSocket.CLOSED && ws.readyState !== WebSocket.CLOSING) {
      try { ws.close() } catch { /* ignore */ }
    }
    setLogs(prev => [...prev, { type: 'warn', content: `WebSocket: ${reason}, reconnecting...` }])
  }, [])

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    // already have a live or pending socket, dont stack another one
    const existing = wsRef.current
    if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
      return
    }

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${proto}//${location.host}/ws`)
    wsRef.current = ws
    lastMessageRef.current = Date.now()

    ws.onopen = () => {
      attemptsRef.current = 0
      setConnected(true)
      setLogs(prev => [...prev, { type: 'info', content: 'WebSocket connected' }])
      // start heartbeat. backend ignores the frame but send() throws if the
      // socket is half open, which trips our reconnect path.
      if (heartbeatRef.current) clearInterval(heartbeatRef.current)
      heartbeatRef.current = setInterval(() => {
        const sock = wsRef.current
        if (!sock || sock.readyState !== WebSocket.OPEN) return
        try {
          sock.send(JSON.stringify({ type: 'ping', t: Date.now() }))
        } catch {
          forceReconnect('heartbeat send failed')
        }
      }, HEARTBEAT_MS)
      // liveness watchdog. if nothing arrives for too long, the socket is
      // probably half open even though readyState still says OPEN.
      if (livenessRef.current) clearInterval(livenessRef.current)
      livenessRef.current = setInterval(() => {
        if (Date.now() - lastMessageRef.current > LIVENESS_TIMEOUT_MS) {
          forceReconnect(`no traffic for ${Math.round(LIVENESS_TIMEOUT_MS / 1000)}s`)
        }
      }, 5000)
    }

    ws.onmessage = (event) => {
      lastMessageRef.current = Date.now()
      const msg: WSMessage = JSON.parse(event.data)
      if (msg.type === 'state') {
        setState(msg.data as AppState)
      } else if (msg.type === 'toast') {
        onToastRef.current(msg.message || '', msg.level || 'info')
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

    ws.onerror = () => {
      // some failure modes never fire onclose. force it so the cleanup path
      // (clear timers, schedule reconnect) actually runs.
      try { ws.close() } catch { /* ignore */ }
    }

    ws.onclose = () => {
      setConnected(false)
      if (heartbeatRef.current) { clearInterval(heartbeatRef.current); heartbeatRef.current = null }
      if (livenessRef.current) { clearInterval(livenessRef.current); livenessRef.current = null }
      // dont spam the log if we're tearing down on unmount
      if (mountedRef.current) {
        setLogs(prev => [...prev, { type: 'error', content: 'WebSocket disconnected' }])
        scheduleReconnect(connect)
      }
    }
  }, [forceReconnect, scheduleReconnect])

  useEffect(() => {
    mountedRef.current = true
    connect()

    // tab came back into focus -> if disconnected, reconnect immediately
    // instead of waiting out the backoff. browsers throttle setTimeout in
    // background tabs so without this the gap after wake can be huge.
    const onVisibility = () => {
      if (document.visibilityState === 'visible') {
        const ws = wsRef.current
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          if (reconnectRef.current) { clearTimeout(reconnectRef.current); reconnectRef.current = null }
          attemptsRef.current = 0
          connect()
        }
      }
    }
    const onOnline = () => {
      const ws = wsRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        if (reconnectRef.current) { clearTimeout(reconnectRef.current); reconnectRef.current = null }
        attemptsRef.current = 0
        connect()
      }
    }
    document.addEventListener('visibilitychange', onVisibility)
    window.addEventListener('online', onOnline)

    return () => {
      mountedRef.current = false
      document.removeEventListener('visibilitychange', onVisibility)
      window.removeEventListener('online', onOnline)
      clearTimers()
      const ws = wsRef.current
      if (ws) {
        try { ws.close() } catch { /* ignore */ }
      }
    }
  }, [connect, clearTimers])

  const clearLogs = useCallback(() => {
    setLogs([{ type: 'info', content: 'Console cleared' }])
  }, [])

  const addLog = useCallback((entry: ConsoleEntry) => {
    setLogs(prev => {
      const next = [...prev, entry]
      return next.length > 300 ? next.slice(-300) : next
    })
  }, [])

  return { state, logs, connected, clearLogs, addLog }
}

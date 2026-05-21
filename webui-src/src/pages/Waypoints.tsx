import { useState, useEffect, useCallback } from 'react'
import { TbMapPin, TbPlus, TbTrash, TbRefresh, TbAlertTriangle, TbWalk } from 'react-icons/tb'
import { api } from '../lib/api'

interface Waypoint {
  name: string
  x: number
  y: number
  z: number
  yaw: number
  note: string
  created_at: number
  updated_at: number
}

interface WaypointsResponse {
  waypoints: Waypoint[]
  world: string
}

interface MappingState {
  running: boolean
  pose: { x: number; y: number; z: number; yaw: number } | null
  world: string
}

interface Props {
  onToast: (msg: string, level?: string) => void
}

export default function Waypoints({ onToast }: Props) {
  const [waypoints, setWaypoints] = useState<Waypoint[]>([])
  const [world, setWorld] = useState<string>('')
  const [mapping, setMapping] = useState<MappingState | null>(null)
  const [loading, setLoading] = useState(false)
  const [showAdd, setShowAdd] = useState(false)
  const [name, setName] = useState('')
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [wps, m] = await Promise.all([
        api<WaypointsResponse>('/api/waypoints'),
        api<MappingState>('/api/mapping/state'),
      ])
      setWaypoints(wps.waypoints)
      setWorld(wps.world ?? '')
      setMapping(m)
    } catch (err) {
      onToast(`load failed: ${(err as Error).message}`, 'error')
    } finally { setLoading(false) }
  }, [onToast])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 2000)
    return () => clearInterval(id)
  }, [refresh])

  const add = async () => {
    if (!name.trim()) {
      onToast('name required', 'warn')
      return
    }
    setBusy(true)
    try {
      await api('/api/waypoints', 'POST', { name: name.trim(), note: note.trim() })
      onToast(`added "${name.trim()}"`, 'success')
      setName('')
      setNote('')
      setShowAdd(false)
      refresh()
    } catch (err) {
      onToast((err as Error).message, 'error')
    } finally { setBusy(false) }
  }

  const remove = async (n: string) => {
    if (!confirm(`Delete waypoint "${n}"?`)) return
    try {
      await api(`/api/waypoints/${encodeURIComponent(n)}`, 'DELETE')
      onToast(`deleted "${n}"`, 'info')
      refresh()
    } catch (err) {
      onToast((err as Error).message, 'error')
    }
  }

  const walkTo = async (n: string) => {
    try {
      const r = await api<{ found: boolean; reason?: string; driving?: boolean; full?: unknown[] }>(
        '/api/mapping/goto', 'POST', { waypoint: n },
      )
      if (!r.found) {
        onToast(`no path: ${r.reason || 'unknown'}`, 'error')
        return
      }
      onToast(`walking to "${n}" (${r.full?.length ?? 0} cells)`, 'success')
    } catch (err) {
      onToast((err as Error).message, 'error')
    }
  }

  const hasPose = !!mapping?.pose
  const canAdd = hasPose && mapping?.running

  return (
    <div className="max-w-[1100px] mx-auto py-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-[18px] font-title font-bold text-text flex items-center gap-2">
            <TbMapPin size={18} className="text-accent" />
            Waypoints
          </h1>
          <p className="text-[12px] text-text-muted mt-0.5">
            world <span className="text-text font-mono">{world || '-'}</span>
            <span className="mx-2 text-text-muted/40">|</span>
            {waypoints.length} saved
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={refresh}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] bg-white/5 text-text-muted hover:bg-white/10 transition disabled:opacity-40"
          >
            <TbRefresh size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
          <button
            onClick={() => setShowAdd(s => !s)}
            disabled={!canAdd}
            title={!canAdd ? 'mapping must be running with a valid pose' : ''}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium bg-accent/15 text-accent hover:bg-accent/25 transition disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <TbPlus size={14} /> Add at Current Pose
          </button>
        </div>
      </div>

      {!canAdd && (
        <div className="flex items-start gap-2 mb-4 p-3 rounded-lg bg-yellow-400/5 border border-yellow-400/20 text-[12px] text-yellow-200/90">
          <TbAlertTriangle size={14} className="mt-0.5 shrink-0" />
          <span>
            Start mapping from the <b>Mapping</b> tab first so a current pose is available.
          </span>
        </div>
      )}

      {showAdd && canAdd && (
        <div className="mb-4 p-4 rounded-xl bg-surface/80 border border-white/10 space-y-2">
          <div className="text-[12px] text-text-muted">
            Saving at pose: <span className="text-text font-mono">
              {mapping?.pose?.x.toFixed(2)}, {mapping?.pose?.y.toFixed(2)}, {mapping?.pose?.z.toFixed(2)}
            </span>
          </div>
          <input
            type="text"
            placeholder="name (e.g. couch, sky_room)"
            value={name}
            onChange={e => setName(e.target.value)}
            autoFocus
            onKeyDown={e => { if (e.key === 'Enter') add() }}
            className="w-full px-3 py-2 rounded-md bg-background border border-white/10 text-text text-[13px] focus:outline-none focus:border-accent/50"
          />
          <input
            type="text"
            placeholder="note (optional)"
            value={note}
            onChange={e => setNote(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') add() }}
            className="w-full px-3 py-2 rounded-md bg-background border border-white/10 text-text text-[13px] focus:outline-none focus:border-accent/50"
          />
          <div className="flex gap-2 justify-end">
            <button
              onClick={() => { setShowAdd(false); setName(''); setNote('') }}
              className="px-3 py-1.5 rounded-md text-[12px] text-text-muted hover:bg-white/5 transition"
            >
              Cancel
            </button>
            <button
              onClick={add}
              disabled={busy || !name.trim()}
              className="px-3 py-1.5 rounded-md text-[12px] font-medium bg-mint/15 text-mint hover:bg-mint/25 transition disabled:opacity-40"
            >
              Save
            </button>
          </div>
        </div>
      )}

      <div className="rounded-xl bg-surface/60 border border-white/10 overflow-hidden">
        {waypoints.length === 0 ? (
          <div className="p-8 text-center text-text-muted text-[13px]">
            no waypoints yet
          </div>
        ) : (
          <table className="w-full text-[13px]">
            <thead className="bg-white/[0.03] text-[11px] text-text-muted/70 uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-2 font-medium">Name</th>
                <th className="text-left px-4 py-2 font-medium">Position</th>
                <th className="text-left px-4 py-2 font-medium">Yaw</th>
                <th className="text-left px-4 py-2 font-medium">Note</th>
                <th className="text-left px-4 py-2 font-medium">Created</th>
                <th className="w-20" />
              </tr>
            </thead>
            <tbody>
              {waypoints.map(w => (
                <tr key={w.name} className="border-t border-white/5 hover:bg-white/[0.02]">
                  <td className="px-4 py-2 font-mono text-text">{w.name}</td>
                  <td className="px-4 py-2 font-mono text-text-muted text-[12px]">
                    {w.x.toFixed(2)}, {w.y.toFixed(2)}, {w.z.toFixed(2)}
                  </td>
                  <td className="px-4 py-2 font-mono text-text-muted text-[12px]">{w.yaw.toFixed(0)}&deg;</td>
                  <td className="px-4 py-2 text-text-muted">{w.note || <span className="text-text-muted/30">-</span>}</td>
                  <td className="px-4 py-2 text-text-muted/60 text-[11px]">
                    {new Date(w.created_at * 1000).toLocaleString()}
                  </td>
                  <td className="px-4 py-2">
                    <div className="flex items-center gap-1 justify-end">
                      <button
                        onClick={() => walkTo(w.name)}
                        disabled={!mapping?.running}
                        className="p-1.5 rounded-md text-text-muted/50 hover:text-mint hover:bg-mint/10 transition disabled:opacity-30"
                        title={mapping?.running ? 'walk here' : 'mapping must be running'}
                      >
                        <TbWalk size={14} />
                      </button>
                      <button
                        onClick={() => remove(w.name)}
                        className="p-1.5 rounded-md text-text-muted/50 hover:text-rose hover:bg-rose/10 transition"
                        title="delete"
                      >
                        <TbTrash size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

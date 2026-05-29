import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import Card from '../components/Card'
import { api } from '../lib/api'
import {
  TbEye, TbRefresh, TbDeviceFloppy, TbRestore,
  TbChevronDown, TbChevronRight, TbAdjustmentsHorizontal,
  TbSearch, TbX, TbLayoutSidebarRightCollapse, TbLayoutSidebarRightExpand,
} from 'react-icons/tb'

interface Props {
  onToast: (msg: string, level?: string) => void
}

interface VisionState {
  enabled: boolean
  has_frame: boolean
  fps?: number
  target_id?: number | null
  target_area?: number
  osc_look_h?: number
  osc_forward?: number
  osc_strafe?: number
  sprinting?: boolean
  detections?: number
  frame_w?: number
  frame_h?: number
  message?: string
}

interface YoloConfigResponse {
  config: Record<string, number>
  tracker_running: boolean
}

interface Field {
  key: string
  label: string
  min: number
  max: number
  step: number
  help: string
}

interface Group {
  id: string
  label: string
  fields: Field[]
}

// sliders grouped by what they actually affect, makes the sidebar way less of a wall
const GROUPS: Group[] = [
  {
    id: 'detection',
    label: 'Detection',
    fields: [
      { key: 'confidence_threshold', label: 'Confidence', min: 0.05, max: 0.95, step: 0.01,
        help: 'min YOLO confidence to count as a detection. higher = fewer false positives' },
      { key: 'iou_threshold', label: 'IoU Threshold', min: 0.1, max: 0.9, step: 0.01,
        help: 'NMS overlap cutoff. lower = more aggressive box merging' },
      { key: 'max_detections', label: 'Max Detections', min: 1, max: 30, step: 1,
        help: 'cap on YOLO results per frame' },
    ],
  },
  {
    id: 'target',
    label: 'Target Selection',
    fields: [
      { key: 'center_distance_weight', label: 'Center Distance Weight', min: 0.0, max: 3.0, step: 0.1,
        help: 'how much being off-center hurts a target score' },
      { key: 'area_weight', label: 'Area Weight', min: 0.0, max: 3.0, step: 0.1,
        help: 'how much being big helps a target score' },
      { key: 'lock_timeout', label: 'Lock Timeout (s)', min: 0.5, max: 30.0, step: 0.5,
        help: 'seconds before we drop a lost target' },
      { key: 'reacquire_threshold', label: 'Reacquire Threshold', min: 0.1, max: 5.0, step: 0.1,
        help: 'how much better a new target must score before switching' },
    ],
  },
  {
    id: 'distance',
    label: 'Distance & Movement',
    fields: [
      { key: 'target_area', label: 'Target Area', min: 0.005, max: 0.2, step: 0.005,
        help: 'desired bounding box size, drives forward/back movement' },
      { key: 'too_close_area', label: 'Too Close Area', min: 0.01, max: 0.5, step: 0.005,
        help: 'past this size the bot will back up' },
      { key: 'sprint_area', label: 'Sprint Area', min: 0.001, max: 0.05, step: 0.001,
        help: 'below this size the bot starts sprinting' },
      { key: 'forward_scale_min', label: 'Forward Scale Min', min: 0.0, max: 1.0, step: 0.05,
        help: 'lowest forward movement value' },
      { key: 'forward_scale_max', label: 'Forward Scale Max', min: 0.0, max: 1.0, step: 0.05,
        help: 'highest forward movement value' },
      { key: 'backup_scale', label: 'Backup Scale', min: 0.0, max: 1.0, step: 0.05,
        help: 'how hard the bot backs up when too close' },
    ],
  },
  {
    id: 'smoothing',
    label: 'Smoothing & Turn',
    fields: [
      { key: 'deadzone', label: 'Deadzone', min: 0.0, max: 0.3, step: 0.01,
        help: 'no input under this normalized offset, keeps it from twitching' },
      { key: 'smoothing_alpha', label: 'Smoothing Alpha', min: 0.05, max: 1.0, step: 0.05,
        help: 'EMA factor. lower = smoother but laggier' },
      { key: 'turn_gain', label: 'Turn Gain', min: 0.5, max: 4.0, step: 0.1,
        help: 'multiplier on horizontal offset before clamping' },
      { key: 'max_turn_rate', label: 'Max Turn Rate', min: 0.01, max: 0.5, step: 0.01,
        help: 'cap on how fast look_h can change per tick' },
    ],
  },
  {
    id: 'strafe',
    label: 'Strafe',
    fields: [
      { key: 'strafe_threshold', label: 'Strafe Threshold', min: 0.0, max: 1.0, step: 0.05,
        help: 'horizontal offset before strafing kicks in' },
      { key: 'strafe_scale', label: 'Strafe Scale', min: 0.0, max: 1.0, step: 0.05,
        help: 'magnitude of strafe input' },
    ],
  },
]

const COLLAPSE_KEY = 'vision.groupsCollapsed'
const SIDEBAR_KEY = 'vision.sidebarOpen'

function Stat({ label, value, tone }: { label: string; value: string | number; tone?: string }) {
  const color = tone === 'good' ? 'text-mint' : tone === 'warn' ? 'text-amber-400'
    : tone === 'bad' ? 'text-rose' : 'text-text'
  return (
    <div className="flex justify-between text-[12px] py-1">
      <span className="text-text-muted/70">{label}</span>
      <span className={`font-mono ${color}`}>{value}</span>
    </div>
  )
}

function Slider({ f, value, dirty, onChange }: {
  f: Field; value: number; dirty: boolean; onChange: (v: number) => void
}) {
  return (
    <div className="group/slider">
      <div className="flex justify-between items-baseline mb-1 gap-2">
        <label className="text-[11px] text-text-muted truncate flex items-center gap-1.5">
          {dirty && <span className="w-1.5 h-1.5 rounded-full bg-accent shrink-0" title="modified" />}
          {f.label}
        </label>
        <input
          type="number"
          min={f.min}
          max={f.max}
          step={f.step}
          value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="w-[68px] text-right font-mono text-[11px] bg-white/[0.04] border border-white/[0.06] rounded px-1.5 py-0.5 text-text focus:outline-none focus:border-accent/40"
        />
      </div>
      <input
        type="range"
        min={f.min}
        max={f.max}
        step={f.step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full accent-accent h-1"
      />
      <p className="text-[10px] text-text-muted/40 mt-0.5 leading-tight opacity-0 group-hover/slider:opacity-100 transition-opacity">
        {f.help}
      </p>
    </div>
  )
}

export default function Vision({ onToast }: Props) {
  const [state, setState] = useState<VisionState | null>(null)
  const [cfg, setCfg] = useState<Record<string, number>>({})
  const [original, setOriginal] = useState<Record<string, number>>({})
  const [trackerRunning, setTrackerRunning] = useState(false)
  const [saving, setSaving] = useState(false)
  const [streamKey, setStreamKey] = useState(0)
  const [search, setSearch] = useState('')

  // persist sidebar + collapsed groups in localStorage so the user's layout sticks
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    try { return localStorage.getItem(SIDEBAR_KEY) !== '0' } catch { return true }
  })
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>(() => {
    try {
      const raw = localStorage.getItem(COLLAPSE_KEY)
      return raw ? JSON.parse(raw) : {}
    } catch { return {} }
  })

  useEffect(() => { try { localStorage.setItem(SIDEBAR_KEY, sidebarOpen ? '1' : '0') } catch { /* ignore */ } }, [sidebarOpen])
  useEffect(() => { try { localStorage.setItem(COLLAPSE_KEY, JSON.stringify(collapsed)) } catch { /* ignore */ } }, [collapsed])

  const pollRef = useRef<number | null>(null)

  const loadConfig = useCallback(async () => {
    try {
      const r = await api<YoloConfigResponse>('/api/vision/yolo-config')
      setCfg({ ...r.config })
      setOriginal({ ...r.config })
      setTrackerRunning(r.tracker_running)
    } catch (e) {
      onToast((e as Error).message, 'error')
    }
  }, [onToast])

  useEffect(() => {
    loadConfig()
    const poll = async () => {
      try {
        const s = await api<VisionState>('/api/vision/state')
        setState(s)
      } catch { /* ignore */ }
    }
    poll()
    pollRef.current = window.setInterval(poll, 500)
    return () => { if (pollRef.current) window.clearInterval(pollRef.current) }
  }, [loadConfig])

  const setVal = (k: string, v: number) => setCfg(prev => ({ ...prev, [k]: v }))
  const dirty = JSON.stringify(cfg) !== JSON.stringify(original)

  // count of changed keys per group, drives the badge
  const dirtyByGroup = useMemo(() => {
    const out: Record<string, number> = {}
    for (const g of GROUPS) {
      let n = 0
      for (const f of g.fields) {
        if (cfg[f.key] !== undefined && cfg[f.key] !== original[f.key]) n++
      }
      out[g.id] = n
    }
    return out
  }, [cfg, original])

  const totalDirty = Object.values(dirtyByGroup).reduce((a, b) => a + b, 0)

  const filteredGroups = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return GROUPS
    return GROUPS.map(g => ({
      ...g,
      fields: g.fields.filter(f =>
        f.label.toLowerCase().includes(q) ||
        f.key.toLowerCase().includes(q) ||
        f.help.toLowerCase().includes(q)
      ),
    })).filter(g => g.fields.length > 0)
  }, [search])

  const save = async () => {
    setSaving(true)
    try {
      const r = await api<{ success: boolean; reloaded: boolean; rejected: string[]; config: Record<string, number> }>(
        '/api/vision/yolo-config', 'POST', { config: cfg }
      )
      setOriginal({ ...r.config })
      setCfg({ ...r.config })
      if (r.reloaded) onToast('YOLO config saved and hot-reloaded', 'success')
      else onToast('YOLO config saved (tracker not running, will pick up on start)', 'info')
      if (r.rejected?.length) onToast(`Ignored: ${r.rejected.join(', ')}`, 'warn')
    } catch (e) {
      onToast((e as Error).message, 'error')
    }
    setSaving(false)
  }

  const reset = () => setCfg({ ...original })
  const toggleGroup = (id: string) => setCollapsed(p => ({ ...p, [id]: !p[id] }))
  const expandAll = () => setCollapsed({})
  const collapseAll = () => setCollapsed(Object.fromEntries(GROUPS.map(g => [g.id, true])))

  const fpsTone = (state?.fps ?? 0) >= 15 ? 'good' : (state?.fps ?? 0) >= 8 ? 'warn' : 'bad'

  return (
    <div className="max-w-[1800px] mx-auto px-4 py-4">
      <div className={`grid gap-4 transition-[grid-template-columns] duration-200 ${
        sidebarOpen ? 'grid-cols-1 xl:grid-cols-[1fr_340px]' : 'grid-cols-1'
      }`}>
        {/* LEFT: stream + stats */}
        <div className="space-y-4 min-w-0">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <TbEye className="text-accent" size={18} />
                <h2 className="font-title text-[14px] uppercase tracking-wider text-text">YOLO Vision Stream</h2>
                {state?.enabled && (
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                    trackerRunning ? 'bg-mint/10 text-mint' : 'bg-white/[0.04] text-text-muted'
                  }`}>
                    {trackerRunning ? 'TRACKING' : 'IDLE'}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setStreamKey(k => k + 1)}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/[0.04] hover:bg-white/[0.08] text-text-muted hover:text-text text-[12px] transition-all"
                  title="reconnect stream"
                >
                  <TbRefresh size={13} />
                  Reconnect
                </button>
                <button
                  onClick={() => setSidebarOpen(o => !o)}
                  className="hidden xl:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/[0.04] hover:bg-white/[0.08] text-text-muted hover:text-text text-[12px] transition-all"
                  title={sidebarOpen ? 'hide tuning panel' : 'show tuning panel'}
                >
                  {sidebarOpen ? <TbLayoutSidebarRightCollapse size={14} /> : <TbLayoutSidebarRightExpand size={14} />}
                  {sidebarOpen ? 'Hide Tuning' : 'Show Tuning'}
                </button>
              </div>
            </div>
            <div className="bg-black/60 border border-white/[0.06] rounded-lg overflow-hidden flex items-center justify-center min-h-[300px]">
              {state?.enabled && state.has_frame ? (
                <img
                  key={streamKey}
                  src={`/api/vision/stream?t=${streamKey}`}
                  alt="YOLO stream"
                  className="max-w-full max-h-[65vh]"
                />
              ) : (
                <div className="p-8 text-center text-text-muted text-[13px]">
                  {state?.enabled
                    ? 'waiting for first annotated frame... (start follow / make sure tracker is active)'
                    : 'YOLO tracker is disabled. enable yolo.enabled in config.yml to see the stream.'}
                </div>
              )}
            </div>
          </Card>

          <Card className="p-4">
            <h2 className="font-title text-[14px] uppercase tracking-wider text-text mb-3">Tracker Stats</h2>
            {!state?.enabled ? (
              <p className="text-text-muted text-[12px]">tracker disabled</p>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-0">
                <Stat label="FPS" value={(state.fps ?? 0).toFixed(1)} tone={fpsTone} />
                <Stat label="Target ID" value={state.target_id ?? 'none'} />
                <Stat label="Target Area" value={`${((state.target_area ?? 0) * 100).toFixed(1)}%`} />
                <Stat label="Detections" value={state.detections ?? 0} />
                <Stat label="Sprinting" value={state.sprinting ? 'YES' : 'no'} tone={state.sprinting ? 'good' : undefined} />
                <Stat label="OSC LookH" value={(state.osc_look_h ?? 0).toFixed(3)} />
                <Stat label="OSC Forward" value={(state.osc_forward ?? 0).toFixed(3)} />
                <Stat label="OSC Strafe" value={(state.osc_strafe ?? 0).toFixed(3)} />
                <Stat label="Frame" value={`${state.frame_w ?? 0}x${state.frame_h ?? 0}`} />
                <Stat label="Tracker" value={trackerRunning ? 'running' : 'stopped'} tone={trackerRunning ? 'good' : 'bad'} />
              </div>
            )}
          </Card>
        </div>

        {/* RIGHT: tuning sidebar */}
        {sidebarOpen && (
          <div>
            <Card className="p-0 sticky top-16 max-h-[calc(100vh-5rem)] flex flex-col overflow-hidden">
              {/* header */}
              <div className="px-4 pt-4 pb-3 border-b border-white/[0.06]">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <TbAdjustmentsHorizontal className="text-accent" size={16} />
                    <h2 className="font-title text-[13px] uppercase tracking-wider text-text">Tuning</h2>
                    {totalDirty > 0 && (
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-accent/15 text-accent">
                        {totalDirty} changed
                      </span>
                    )}
                  </div>
                  <button
                    onClick={() => setSidebarOpen(false)}
                    className="xl:flex hidden items-center justify-center w-6 h-6 rounded-md hover:bg-white/[0.06] text-text-muted hover:text-text transition-all"
                    title="hide tuning panel"
                  >
                    <TbX size={14} />
                  </button>
                </div>

                {/* search */}
                <div className="relative mb-2">
                  <TbSearch size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-text-muted/60" />
                  <input
                    type="text"
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    placeholder="filter settings..."
                    className="w-full pl-7 pr-7 py-1.5 rounded-md bg-white/[0.03] border border-white/[0.06] text-[12px] text-text placeholder:text-text-muted/50 focus:outline-none focus:border-accent/40"
                  />
                  {search && (
                    <button
                      onClick={() => setSearch('')}
                      className="absolute right-1.5 top-1/2 -translate-y-1/2 text-text-muted/60 hover:text-text"
                    >
                      <TbX size={12} />
                    </button>
                  )}
                </div>

                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1 text-[10px]">
                    <button onClick={expandAll} className="text-text-muted/70 hover:text-text px-1 py-0.5 rounded hover:bg-white/[0.04]">
                      expand all
                    </button>
                    <span className="text-text-muted/30">/</span>
                    <button onClick={collapseAll} className="text-text-muted/70 hover:text-text px-1 py-0.5 rounded hover:bg-white/[0.04]">
                      collapse all
                    </button>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={reset}
                      disabled={!dirty}
                      className="flex items-center gap-1 px-2 py-1 rounded-md bg-white/[0.04] hover:bg-white/[0.08] text-text-muted hover:text-text text-[11px] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      <TbRestore size={11} />
                      Revert
                    </button>
                    <button
                      onClick={save}
                      disabled={!dirty || saving}
                      className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-accent/15 hover:bg-accent/25 text-accent text-[11px] font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      <TbDeviceFloppy size={11} />
                      {saving ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                </div>
              </div>

              {/* groups */}
              <div className="overflow-y-auto flex-1 px-2 py-2">
                {filteredGroups.length === 0 && (
                  <p className="text-center text-text-muted text-[12px] py-8">no settings match "{search}"</p>
                )}
                {filteredGroups.map(g => {
                  // when searching, force-open. otherwise honor stored collapsed state.
                  const isOpen = search.trim() ? true : !collapsed[g.id]
                  const groupDirty = dirtyByGroup[g.id] ?? 0
                  return (
                    <div key={g.id} className="mb-1">
                      <button
                        onClick={() => toggleGroup(g.id)}
                        className="w-full flex items-center justify-between px-2 py-1.5 rounded-md hover:bg-white/[0.04] text-left transition-colors"
                      >
                        <div className="flex items-center gap-1.5">
                          {isOpen ? <TbChevronDown size={12} className="text-text-muted" /> : <TbChevronRight size={12} className="text-text-muted" />}
                          <span className="text-[12px] font-medium text-text">{g.label}</span>
                          {groupDirty > 0 && (
                            <span className="text-[9px] font-mono px-1 py-0.5 rounded bg-accent/15 text-accent">
                              {groupDirty}
                            </span>
                          )}
                        </div>
                        <span className="text-[10px] text-text-muted/40">{g.fields.length}</span>
                      </button>
                      {isOpen && (
                        <div className="space-y-2.5 px-2 pt-2 pb-3">
                          {g.fields.map(f => {
                            const v = cfg[f.key]
                            if (v === undefined) return null
                            const fieldDirty = v !== original[f.key]
                            return <Slider key={f.key} f={f} value={v} dirty={fieldDirty} onChange={nv => setVal(f.key, nv)} />
                          })}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </Card>
          </div>
        )}

        {/* floating "show tuning" tab when sidebar is hidden (xl only) */}
        {!sidebarOpen && (
          <button
            onClick={() => setSidebarOpen(true)}
            className="hidden xl:flex fixed right-0 top-1/2 -translate-y-1/2 items-center gap-2 px-3 py-3 rounded-l-lg bg-surface border border-r-0 border-white/[0.08] text-text-muted hover:text-accent hover:bg-white/[0.04] transition-all z-10 shadow-card"
            title="show tuning panel"
            style={{ writingMode: 'vertical-rl' }}
          >
            <TbAdjustmentsHorizontal size={14} className="rotate-90" />
            <span className="text-[11px] uppercase tracking-wider">Tuning</span>
            {totalDirty > 0 && (
              <span className="text-[9px] font-mono px-1 py-0.5 rounded bg-accent/20 text-accent rotate-180">
                {totalDirty}
              </span>
            )}
          </button>
        )}
      </div>
    </div>
  )
}

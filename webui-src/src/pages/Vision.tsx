import { useState, useEffect, useCallback, useRef } from 'react'
import Card from '../components/Card'
import { api } from '../lib/api'
import { TbEye, TbRefresh, TbDeviceFloppy, TbRestore } from 'react-icons/tb'

interface Props {
  onToast: (msg: string, level?: string) => void
}

interface VisionState {
  enabled: boolean
  has_frame: boolean
  vision_debug_port_running?: boolean
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

// each slider's min/max/step. anything outside this list is hidden from the UI.
const FIELDS: { key: string; label: string; min: number; max: number; step: number; help: string }[] = [
  { key: 'confidence_threshold', label: 'Confidence Threshold', min: 0.05, max: 0.95, step: 0.01,
    help: 'min YOLO confidence to count as a detection. higher = fewer false positives' },
  { key: 'iou_threshold', label: 'IoU Threshold', min: 0.1, max: 0.9, step: 0.01,
    help: 'NMS overlap cutoff. lower = more aggressive box merging' },
  { key: 'target_area', label: 'Target Area (% of frame)', min: 0.005, max: 0.2, step: 0.005,
    help: 'desired bounding box size, drives forward/back movement' },
  { key: 'too_close_area', label: 'Too Close Area', min: 0.01, max: 0.5, step: 0.005,
    help: 'past this size the bot will back up' },
  { key: 'sprint_area', label: 'Sprint Area', min: 0.001, max: 0.05, step: 0.001,
    help: 'below this size the bot starts sprinting' },
  { key: 'deadzone', label: 'Deadzone', min: 0.0, max: 0.3, step: 0.01,
    help: 'no input under this normalized offset, keeps it from twitching' },
  { key: 'smoothing_alpha', label: 'Smoothing Alpha', min: 0.05, max: 1.0, step: 0.05,
    help: 'EMA factor. lower = smoother but laggier' },
  { key: 'turn_gain', label: 'Turn Gain', min: 0.5, max: 4.0, step: 0.1,
    help: 'multiplier on horizontal offset before clamping' },
  { key: 'max_turn_rate', label: 'Max Turn Rate / frame', min: 0.01, max: 0.5, step: 0.01,
    help: 'cap on how fast look_h can change per tick' },
  { key: 'center_distance_weight', label: 'Center Distance Weight', min: 0.0, max: 3.0, step: 0.1,
    help: 'how much being off-center hurts a target score' },
  { key: 'area_weight', label: 'Area Weight', min: 0.0, max: 3.0, step: 0.1,
    help: 'how much being big helps a target score' },
  { key: 'lock_timeout', label: 'Lock Timeout (s)', min: 0.5, max: 30.0, step: 0.5,
    help: 'seconds before we drop a lost target' },
  { key: 'reacquire_threshold', label: 'Reacquire Threshold', min: 0.1, max: 5.0, step: 0.1,
    help: 'how much better a new target must score before switching' },
  { key: 'forward_scale_min', label: 'Forward Scale Min', min: 0.0, max: 1.0, step: 0.05,
    help: 'lowest forward movement value' },
  { key: 'forward_scale_max', label: 'Forward Scale Max', min: 0.0, max: 1.0, step: 0.05,
    help: 'highest forward movement value' },
  { key: 'backup_scale', label: 'Backup Scale', min: 0.0, max: 1.0, step: 0.05,
    help: 'how hard the bot backs up when too close' },
  { key: 'strafe_threshold', label: 'Strafe Threshold', min: 0.0, max: 1.0, step: 0.05,
    help: 'horizontal offset before strafing kicks in (unused in some modes)' },
  { key: 'strafe_scale', label: 'Strafe Scale', min: 0.0, max: 1.0, step: 0.05,
    help: 'magnitude of strafe input' },
  { key: 'max_detections', label: 'Max Detections', min: 1, max: 30, step: 1,
    help: 'cap on YOLO results per frame' },
]

function Stat({ label, value, tone }: { label: string; value: string | number; tone?: string }) {
  const color = tone === 'good' ? 'text-mint' : tone === 'warn' ? 'text-amber-400' : tone === 'bad' ? 'text-rose' : 'text-text'
  return (
    <div className="flex justify-between text-[12px] py-1">
      <span className="text-text-muted/70">{label}</span>
      <span className={`font-mono ${color}`}>{value}</span>
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

  const fpsTone = (state?.fps ?? 0) >= 15 ? 'good' : (state?.fps ?? 0) >= 8 ? 'warn' : 'bad'

  return (
    <div className="max-w-[1600px] mx-auto px-4 py-4">
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_360px] gap-4">
        {/* LEFT: stream + sliders */}
        <div className="space-y-4">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <TbEye className="text-accent" size={18} />
                <h2 className="font-title text-[14px] uppercase tracking-wider text-text">YOLO Vision Stream</h2>
              </div>
              <button
                onClick={() => setStreamKey(k => k + 1)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/[0.04] hover:bg-white/[0.08] text-text-muted hover:text-text text-[12px] transition-all"
                title="reconnect stream"
              >
                <TbRefresh size={13} />
                Reconnect
              </button>
            </div>
            <div className="bg-black/60 border border-white/[0.06] rounded-lg overflow-hidden flex items-center justify-center min-h-[300px]">
              {state?.enabled && state.has_frame ? (
                <img
                  key={streamKey}
                  src={`/api/vision/stream?t=${streamKey}`}
                  alt="YOLO stream"
                  className="max-w-full max-h-[60vh]"
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
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-title text-[14px] uppercase tracking-wider text-text">Model Settings</h2>
              <div className="flex items-center gap-2">
                <button
                  onClick={reset}
                  disabled={!dirty}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/[0.04] hover:bg-white/[0.08] text-text-muted hover:text-text text-[12px] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <TbRestore size={13} />
                  Revert
                </button>
                <button
                  onClick={save}
                  disabled={!dirty || saving}
                  className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-accent/15 hover:bg-accent/25 text-accent text-[12px] font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <TbDeviceFloppy size={13} />
                  {saving ? 'Saving...' : dirty ? 'Save & Apply' : 'Saved'}
                </button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3">
              {FIELDS.map(f => {
                const v = cfg[f.key]
                if (v === undefined) return null
                return (
                  <div key={f.key}>
                    <div className="flex justify-between items-baseline mb-1">
                      <label className="text-[12px] text-text-muted">{f.label}</label>
                      <input
                        type="number"
                        min={f.min}
                        max={f.max}
                        step={f.step}
                        value={v}
                        onChange={e => setVal(f.key, Number(e.target.value))}
                        className="w-20 text-right font-mono text-[12px] bg-white/[0.04] border border-white/[0.06] rounded px-1.5 py-0.5 text-text focus:outline-none focus:border-accent/40"
                      />
                    </div>
                    <input
                      type="range"
                      min={f.min}
                      max={f.max}
                      step={f.step}
                      value={v}
                      onChange={e => setVal(f.key, Number(e.target.value))}
                      className="w-full accent-accent"
                    />
                    <p className="text-[10px] text-text-muted/50 mt-0.5 leading-tight">{f.help}</p>
                  </div>
                )
              })}
            </div>
          </Card>
        </div>

        {/* RIGHT: live stats */}
        <div>
          <Card className="p-4 sticky top-16">
            <h2 className="font-title text-[14px] uppercase tracking-wider text-text mb-3">Tracker Stats</h2>
            {!state?.enabled ? (
              <p className="text-text-muted text-[12px]">tracker disabled</p>
            ) : (
              <div className="space-y-1">
                <Stat label="FPS" value={(state.fps ?? 0).toFixed(1)} tone={fpsTone} />
                <Stat label="Target ID" value={state.target_id ?? 'none'} />
                <Stat label="Target Area" value={`${((state.target_area ?? 0) * 100).toFixed(1)}%`} />
                <Stat label="Detections" value={state.detections ?? 0} />
                <Stat label="Sprinting" value={state.sprinting ? 'YES' : 'no'} tone={state.sprinting ? 'good' : undefined} />
                <div className="h-px bg-white/[0.06] my-2" />
                <Stat label="OSC LookH" value={(state.osc_look_h ?? 0).toFixed(3)} />
                <Stat label="OSC Forward" value={(state.osc_forward ?? 0).toFixed(3)} />
                <Stat label="OSC Strafe" value={(state.osc_strafe ?? 0).toFixed(3)} />
                <div className="h-px bg-white/[0.06] my-2" />
                <Stat label="Frame" value={`${state.frame_w ?? 0}x${state.frame_h ?? 0}`} />
                <Stat label="Tracker Running" value={trackerRunning ? 'yes' : 'no'} tone={trackerRunning ? 'good' : 'bad'} />
                {state.vision_debug_port_running && (
                  <p className="text-[10px] text-text-muted/60 pt-2">
                    standalone debug also on :{(window.location.host.split(':')[0])}:8767/vision
                  </p>
                )}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}

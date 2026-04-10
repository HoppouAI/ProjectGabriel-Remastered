import { useState, useCallback, useEffect } from 'react'
import Console from '../components/Console'
import { api } from '../lib/api'
import { formatNumber, formatTime, truncate } from '../lib/utils'
import type { AppState, ConsoleEntry, Personality } from '../lib/types'
import {
  RiRefreshLine, RiDeleteBinLine, RiSendPlaneLine,
  RiPlayFill, RiPauseFill, RiStopFill, RiVolumeUpFill,
  RiVipCrownLine, RiSettings3Line,
  RiArrowUpLine,
} from 'react-icons/ri'
import {
  TbMicrophone, TbMicrophoneOff, TbLayoutSidebarLeftCollapse,
  TbLayoutSidebarLeftExpand, TbBrain, TbActivity,
} from 'react-icons/tb'

interface Props {
  state: AppState | null
  logs: ConsoleEntry[]
  clearLogs: () => void
  onToast: (msg: string, level?: string) => void
}

export default function Dashboard({ state, logs, clearLogs, onToast }: Props) {
  const [text, setText] = useState('')
  const [sysText, setSysText] = useState('')
  const [volume, setVolume] = useState(100)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [inputMode, setInputMode] = useState<'message' | 'system'>('message')

  useEffect(() => {
    if (state?.music_progress?.is_playing) return
  }, [state?.music_progress])

  const act = useCallback(async (endpoint: string, body?: unknown) => {
    try {
      const res = await api<{ status?: string; detail?: string }>(endpoint, 'POST', body)
      onToast(res.status || res.detail || 'Done', 'success')
    } catch (e: unknown) {
      onToast((e as Error).message, 'error')
    }
  }, [onToast])

  const music = state?.music_progress
  const usage = state?.usage_metadata

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (inputMode === 'message') {
      if (text.trim()) { act('/api/send-text', { text }); setText('') }
    } else {
      if (sysText.trim()) { act('/api/send-system-instruction', { instruction: sysText }); setSysText('') }
    }
  }

  return (
    <div className="flex h-[calc(100vh-72px)] gap-0">
      {/* Left sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} shrink-0 transition-all duration-200 overflow-hidden border-r border-white/[0.06]`}>
        <div className="w-64 h-full flex flex-col overflow-y-auto console-scroll p-3 gap-2.5">
          {/* Quick actions - icon row */}
          <div className="flex items-center gap-1 p-1.5 bg-surface/50 rounded-lg">
            <IconBtn
              icon={<RiRefreshLine size={15} />}
              label="Reconnect"
              onClick={() => act('/api/reconnect')}
            />
            <IconBtn
              icon={state?.mic_muted ? <TbMicrophoneOff size={15} /> : <TbMicrophone size={15} />}
              label={state?.mic_muted ? 'Unmute' : 'Mute'}
              onClick={() => act('/api/toggle-mute')}
              variant={state?.mic_muted ? 'danger' : 'default'}
            />
            <IconBtn
              icon={<RiDeleteBinLine size={15} />}
              label="Clear Session"
              onClick={() => act('/api/clear-session')}
            />
            <IconBtn
              icon={<RiDeleteBinLine size={15} />}
              label="Clear Console"
              onClick={clearLogs}
              variant="muted"
            />
          </div>

          {/* Session + VRChat */}
          <SidebarSection label="Session">
            <div className="space-y-1.5">
              <StatusRow
                label="Session"
                value={state?.session_handle?.exists ? 'Active' : 'Inactive'}
                active={state?.session_handle?.exists}
              />
              {state?.session_handle?.age_minutes !== undefined && (
                <StatusRow label="Uptime" value={`${Math.round(state.session_handle.age_minutes)}m`} />
              )}
              <StatusRow
                label="VRChat"
                value={state?.vrchat?.is_in_world ? 'In World' : 'Offline'}
                active={state?.vrchat?.is_in_world}
              />
              {(state?.vrchat?.player_count ?? 0) > 0 && (
                <StatusRow label="Players" value={String(state?.vrchat?.player_count)} />
              )}
            </div>
          </SidebarSection>

          {/* Usage */}
          <SidebarSection label="Usage">
            <div className="grid grid-cols-2 gap-x-3 gap-y-1">
              <UsageRow label="Prompt" value={formatNumber(usage?.prompt_tokens)} />
              <UsageRow label="Response" value={formatNumber(usage?.response_tokens)} />
              <UsageRow label="Total" value={formatNumber(usage?.total_tokens)} />
              <UsageRow label="Tools" value={formatNumber(usage?.tool_calls)} />
            </div>
          </SidebarSection>

          {/* Personality */}
          <PersonalitySelector
            personalities={state?.personalities || []}
            current={state?.current_personality || null}
            onSwitch={(id) => act('/api/switch-personality', { personality_id: id })}
          />

          {/* Now playing */}
          {music?.is_playing && (
            <SidebarSection label="Now Playing">
              <p className="text-xs text-text truncate mb-2">{music.song_name || 'Unknown'}</p>
              <div className="h-1 bg-background rounded-full overflow-hidden mb-1">
                <div
                  className="h-full bg-accent rounded-full transition-all"
                  style={{ width: music.duration > 0 ? `${(music.position / music.duration) * 100}%` : '0%' }}
                />
              </div>
              <div className="flex items-center justify-between text-[10px] text-text-muted/50 font-title mb-2">
                <span>{formatTime(music.position)}</span>
                <span>{formatTime(music.duration)}</span>
              </div>
              <div className="flex items-center gap-1">
                <button onClick={() => act('/api/pause-music')} className="p-1 hover:text-accent transition-colors text-xs"><RiPauseFill /></button>
                <button onClick={() => act('/api/resume-music')} className="p-1 hover:text-accent transition-colors text-xs"><RiPlayFill /></button>
                <button onClick={() => act('/api/stop-music')} className="p-1 hover:text-rose transition-colors text-xs"><RiStopFill /></button>
                <div className="ml-auto flex items-center gap-1">
                  <RiVolumeUpFill className="text-text-muted/40 text-[10px]" />
                  <input
                    type="range" min={0} max={100} value={volume}
                    onChange={e => { setVolume(+e.target.value); act('/api/set-volume', { volume: +e.target.value }) }}
                    className="w-14 accent-accent h-0.5"
                  />
                  <span className="text-text-muted/40 text-[10px] font-title">{volume}%</span>
                </div>
              </div>
            </SidebarSection>
          )}

          {/* Recent memories */}
          <SidebarSection label="Recent Memories">
            <div className="space-y-1.5">
              {state?.recent_memories?.memories?.length ? (
                state.recent_memories.memories.slice(0, 4).map((m, i) => (
                  <div key={i} className="text-[11px] border-l-2 border-accent/20 pl-2 py-0.5">
                    <span className="text-text/70">{truncate(m.content, 60)}</span>
                    <span className="block text-text-muted/30 text-[9px] font-title">{m.category}</span>
                  </div>
                ))
              ) : (
                <p className="text-text-muted/40 text-[11px] italic">No memories yet</p>
              )}
            </div>
          </SidebarSection>
        </div>
      </div>

      {/* Main content area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Topbar */}
        <div className="flex items-center gap-3 px-4 py-2 border-b border-white/[0.06] shrink-0">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1.5 hover:bg-white/[0.06] rounded-lg transition-colors text-text-muted hover:text-text"
            title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {sidebarOpen ? <TbLayoutSidebarLeftCollapse size={18} /> : <TbLayoutSidebarLeftExpand size={18} />}
          </button>

          <div className="flex-1" />

          {/* Status indicators */}
          <div className="flex items-center gap-4 text-xs font-title">
            {usage && (
              <div className="flex items-center gap-1.5 text-text-muted/60">
                <TbBrain size={14} />
                <span>{formatNumber(usage.total_tokens)} tokens</span>
              </div>
            )}
            {state?.session_handle?.exists && (
              <div className="flex items-center gap-1.5 text-mint/70">
                <TbActivity size={14} />
                <span>Live</span>
              </div>
            )}
          </div>
        </div>

        {/* Console / Chat area */}
        <div className="flex-1 min-h-0 overflow-hidden">
          <div className="h-full max-w-4xl mx-auto px-4 py-3">
            <Console logs={logs} />
          </div>
        </div>

        {/* Bottom input area */}
        <div className="shrink-0 border-t border-white/[0.06] px-4 py-3">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={handleSubmit}>
              <div className="flex items-center gap-1.5 bg-surface border border-white/[0.08] rounded-xl px-3 py-1.5 focus-within:border-accent/30 transition-colors">
                {inputMode === 'message' ? (
                  <input
                    value={text}
                    onChange={e => setText(e.target.value)}
                    placeholder="Send a message to the model..."
                    className="flex-1 bg-transparent py-1.5 text-sm text-text placeholder:text-text-muted/40 focus:outline-none min-w-0"
                  />
                ) : (
                  <input
                    value={sysText}
                    onChange={e => setSysText(e.target.value)}
                    placeholder="Send a system instruction..."
                    className="flex-1 bg-transparent py-1.5 text-sm text-text placeholder:text-text-muted/40 focus:outline-none min-w-0"
                  />
                )}
                <button
                  type="button"
                  onClick={() => setInputMode(inputMode === 'message' ? 'system' : 'message')}
                  title={inputMode === 'message' ? 'Switch to system instruction' : 'Switch to message'}
                  className={`p-1.5 rounded-lg transition-colors shrink-0 ${
                    inputMode === 'system'
                      ? 'text-accent bg-accent/10'
                      : 'text-text-muted/50 hover:text-text-muted hover:bg-white/[0.04]'
                  }`}
                >
                  <RiSettings3Line size={16} />
                </button>
                <button
                  type="submit"
                  className="p-2 bg-accent text-background rounded-lg hover:bg-accent-dim transition-colors shrink-0"
                >
                  <RiArrowUpLine size={16} />
                </button>
              </div>
              <div className="flex items-center justify-between mt-1.5 px-1">
                <span className="text-[10px] text-text-muted/30 font-title">
                  {inputMode === 'message' ? 'Message' : 'System Instruction'} mode
                </span>
                <span className="text-[10px] text-text-muted/30 font-title">
                  Enter to send
                </span>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}

/* -- Sidebar subcomponents -- */

function SidebarSection({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="bg-surface/40 rounded-lg p-3 border border-white/[0.04]">
      <h3 className="font-title text-[10px] text-text-muted/50 uppercase tracking-wider mb-2">{label}</h3>
      {children}
    </div>
  )
}

function IconBtn({ icon, label, onClick, variant = 'default' }: {
  icon: React.ReactNode; label: string; onClick: () => void; variant?: 'default' | 'danger' | 'muted'
}) {
  const styles = {
    default: 'text-accent/70 hover:text-accent hover:bg-accent/15',
    danger: 'text-rose/80 hover:text-rose hover:bg-rose/15',
    muted: 'text-text-muted/40 hover:text-text-muted hover:bg-white/[0.06]',
  }
  return (
    <button
      onClick={onClick}
      title={label}
      className={`p-2 rounded-lg transition-colors flex-1 flex items-center justify-center ${styles[variant]}`}
    >
      {icon}
    </button>
  )
}

function StatusRow({ label, value, active }: { label: string; value: string; active?: boolean }) {
  return (
    <div className="flex items-center justify-between text-[11px]">
      <span className="text-text-muted/50 font-title">{label}</span>
      <span className={`font-title ${active ? 'text-mint' : active === false ? 'text-text-muted/40' : 'text-text/70'}`}>{value}</span>
    </div>
  )
}

function UsageRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-[11px]">
      <span className="text-text-muted/40 font-title">{label}</span>
      <span className="text-text/60 font-title tabular-nums">{value}</span>
    </div>
  )
}

function PersonalitySelector({ personalities, current, onSwitch }: {
  personalities: Personality[]; current: string | null; onSwitch: (id: string) => void
}) {
  if (!personalities.length) return null
  return (
    <SidebarSection label="Personality">
      <div className="flex flex-wrap gap-1">
        {personalities.map(p => (
          <button
            key={p.id}
            onClick={() => onSwitch(p.id)}
            className={`px-2 py-0.5 rounded text-[11px] font-title transition-colors ${
              current === p.id
                ? 'bg-accent/20 text-accent border border-accent/30'
                : 'bg-background/40 text-text-muted/50 hover:text-text-muted border border-transparent hover:border-white/[0.06]'
            }`}
          >
            {p.name}
          </button>
        ))}
      </div>
    </SidebarSection>
  )
}

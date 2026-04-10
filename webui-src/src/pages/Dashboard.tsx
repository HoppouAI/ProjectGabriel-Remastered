import { useState, useCallback, useEffect } from 'react'
import Card from '../components/Card'
import Console from '../components/Console'
import { api } from '../lib/api'
import { formatNumber, formatTime, truncate } from '../lib/utils'
import type { AppState, ConsoleEntry, Personality } from '../lib/types'
import {
  RiRefreshLine, RiDeleteBinLine, RiSendPlaneLine,
  RiPlayFill, RiPauseFill, RiStopFill, RiVolumeUpFill,
  RiVipCrownLine, RiSettings3Line,
} from 'react-icons/ri'
import { TbMicrophone, TbMicrophoneOff } from 'react-icons/tb'

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

  return (
    <div className="flex flex-col gap-3 h-[calc(100vh-72px)]">
      {/* Top bar: session controls + system instruction */}
      <div className="flex flex-wrap items-center gap-2 shrink-0">
        <ToolbarBtn icon={<RiRefreshLine />} label="Reconnect" onClick={() => act('/api/reconnect')} />
        <ToolbarBtn
          icon={state?.mic_muted ? <TbMicrophoneOff className="text-rose" /> : <TbMicrophone />}
          label={state?.mic_muted ? 'Unmute' : 'Mute'}
          onClick={() => act('/api/toggle-mute')}
        />
        <ToolbarBtn icon={<RiDeleteBinLine />} label="Clear Session" onClick={() => act('/api/clear-session')} />
        <ToolbarBtn icon={<RiDeleteBinLine />} label="Clear Console" onClick={clearLogs} accent={false} />

        <div className="h-5 w-px bg-white/[0.08] mx-1 hidden sm:block" />

        {/* System instruction inline */}
        <form
          className="flex gap-1.5 flex-1 min-w-[200px]"
          onSubmit={(e) => { e.preventDefault(); if (sysText.trim()) { act('/api/send-system-instruction', { instruction: sysText }); setSysText('') } }}
        >
          <div className="relative flex-1">
            <RiSettings3Line className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted/50 text-xs" />
            <input
              value={sysText}
              onChange={e => setSysText(e.target.value)}
              placeholder="System instruction..."
              className="w-full bg-background/60 border border-white/[0.08] rounded-lg pl-8 pr-3 py-1.5 text-xs text-text placeholder:text-text-muted/40 focus:outline-none focus:border-accent/40 font-title"
            />
          </div>
          <button type="submit" className="px-2.5 py-1.5 bg-surface-alt text-text-muted rounded-lg text-xs font-title hover:text-accent hover:bg-accent/10 transition-colors shrink-0">
            Inject
          </button>
        </form>
      </div>

      {/* Main content: console + sidebar */}
      <div className="grid grid-cols-12 gap-3 flex-1 min-h-0">
        {/* Console */}
        <div className="col-span-12 lg:col-span-8 flex flex-col min-h-0">
          <Card className="p-3 flex flex-col flex-1 min-h-0">
            <div className="flex items-center gap-2 mb-2 shrink-0">
              <span className="font-title text-sm text-text-muted">Console</span>
              <span className="text-text-muted/40 text-xs font-title">{logs.length} entries</span>
            </div>
            <div className="flex-1 min-h-0">
              <Console logs={logs} />
            </div>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-3 overflow-y-auto console-scroll min-h-0 pb-1">
          {/* Usage stats - compact row */}
          <Card className="p-3">
            <h3 className="font-title text-[10px] text-text-muted/60 uppercase tracking-wider mb-2">Usage</h3>
            <div className="grid grid-cols-4 gap-1">
              <MiniStat label="Prompt" value={formatNumber(usage?.prompt_tokens)} />
              <MiniStat label="Response" value={formatNumber(usage?.response_tokens)} />
              <MiniStat label="Total" value={formatNumber(usage?.total_tokens)} />
              <MiniStat label="Tools" value={formatNumber(usage?.tool_calls)} />
            </div>
          </Card>

          {/* Session + VRChat side by side */}
          <div className="grid grid-cols-2 gap-3">
            {state?.session_handle && (
              <Card className="p-3">
                <h3 className="font-title text-[10px] text-text-muted/60 uppercase tracking-wider mb-1.5">Session</h3>
                <div className={`font-title text-sm ${state.session_handle.exists ? 'text-mint' : 'text-text-muted'}`}>
                  {state.session_handle.exists ? 'Active' : 'None'}
                </div>
                {state.session_handle.age_minutes !== undefined && (
                  <div className="text-text-muted/50 text-[10px] font-title">{Math.round(state.session_handle.age_minutes)}m uptime</div>
                )}
              </Card>
            )}
            {state?.vrchat && (
              <Card className="p-3">
                <h3 className="font-title text-[10px] text-text-muted/60 uppercase tracking-wider mb-1.5">VRChat</h3>
                <div className={`font-title text-sm ${state.vrchat.is_in_world ? 'text-mint' : 'text-text-muted'}`}>
                  {state.vrchat.is_in_world ? 'In World' : 'Offline'}
                </div>
                {state.vrchat.player_count > 0 && (
                  <div className="text-text-muted/50 text-[10px] font-title">{state.vrchat.player_count} players</div>
                )}
              </Card>
            )}
          </div>

          {/* Personality */}
          <PersonalitySelector
            personalities={state?.personalities || []}
            current={state?.current_personality || null}
            onSwitch={(id) => act('/api/switch-personality', { personality_id: id })}
          />

          {/* Now playing - only shown when music is active */}
          {music?.is_playing && (
            <Card className="p-3">
              <h3 className="font-title text-[10px] text-text-muted/60 uppercase tracking-wider mb-1.5">Now Playing</h3>
              <p className="text-sm text-text truncate mb-2">{music.song_name || 'Unknown'}</p>
              <div className="h-1 bg-background/60 rounded-full overflow-hidden mb-1">
                <div
                  className="h-full bg-accent rounded-full transition-all"
                  style={{ width: music.duration > 0 ? `${(music.position / music.duration) * 100}%` : '0%' }}
                />
              </div>
              <div className="flex items-center justify-between text-[10px] text-text-muted/50 font-title mb-2">
                <span>{formatTime(music.position)}</span>
                <span>{formatTime(music.duration)}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <button onClick={() => act('/api/pause-music')} className="p-1 hover:text-accent transition-colors text-sm"><RiPauseFill /></button>
                <button onClick={() => act('/api/resume-music')} className="p-1 hover:text-accent transition-colors text-sm"><RiPlayFill /></button>
                <button onClick={() => act('/api/stop-music')} className="p-1 hover:text-rose transition-colors text-sm"><RiStopFill /></button>
                <div className="ml-auto flex items-center gap-1">
                  <RiVolumeUpFill className="text-text-muted/40 text-[10px]" />
                  <input
                    type="range" min={0} max={100} value={volume}
                    onChange={e => { setVolume(+e.target.value); act('/api/set-volume', { volume: +e.target.value }) }}
                    className="w-16 accent-accent h-0.5"
                  />
                  <span className="text-text-muted/40 text-[10px] font-title w-6 text-right">{volume}%</span>
                </div>
              </div>
            </Card>
          )}

          {/* Recent memories */}
          <Card className="p-3">
            <h3 className="font-title text-[10px] text-text-muted/60 uppercase tracking-wider mb-2">Recent Memories</h3>
            <div className="space-y-1.5">
              {state?.recent_memories?.memories?.length ? (
                state.recent_memories.memories.slice(0, 5).map((m, i) => (
                  <div key={i} className="text-[11px] border-l-2 border-accent/20 pl-2 py-0.5">
                    <span className="text-text/80">{truncate(m.content, 70)}</span>
                    <span className="block text-text-muted/40 text-[9px] font-title mt-0.5">{m.category}</span>
                  </div>
                ))
              ) : (
                <p className="text-text-muted/50 text-xs italic">No recent memories</p>
              )}
            </div>
          </Card>
        </div>
      </div>

      {/* Bottom: chat input */}
      <Card className="p-3 shrink-0">
        <form
          className="flex gap-2"
          onSubmit={(e) => { e.preventDefault(); if (text.trim()) { act('/api/send-text', { text }); setText('') } }}
        >
          <input
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Send a message..."
            className="flex-1 bg-background/60 border border-white/[0.08] rounded-lg px-4 py-2 text-sm text-text placeholder:text-text-muted/40 focus:outline-none focus:border-accent/40"
          />
          <button type="submit" className="px-4 py-2 bg-accent text-background font-title text-sm rounded-lg hover:bg-accent-dim transition-colors flex items-center gap-1.5 shrink-0">
            <RiSendPlaneLine />
            Send
          </button>
        </form>
      </Card>
    </div>
  )
}

/* -- Subcomponents -- */

function ToolbarBtn({ icon, label, onClick, accent = true }: {
  icon: React.ReactNode; label: string; onClick: () => void; accent?: boolean
}) {
  return (
    <button
      onClick={onClick}
      title={label}
      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs transition-colors shrink-0 ${
        accent
          ? 'bg-accent/10 text-accent hover:bg-accent/20'
          : 'bg-white/[0.04] text-text-muted hover:text-text hover:bg-white/[0.08]'
      }`}
    >
      {icon}
      <span className="hidden sm:inline font-title">{label}</span>
    </button>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <div className="font-title text-sm text-text">{value}</div>
      <div className="text-text-muted/50 text-[9px] font-title uppercase">{label}</div>
    </div>
  )
}

function PersonalitySelector({ personalities, current, onSwitch }: {
  personalities: Personality[]; current: string | null; onSwitch: (id: string) => void
}) {
  if (!personalities.length) return null
  return (
    <Card className="p-3">
      <h3 className="font-title text-[10px] text-text-muted/60 uppercase tracking-wider mb-2">
        <RiVipCrownLine className="inline mr-1 text-accent/60" />
        Personality
      </h3>
      <div className="flex flex-wrap gap-1">
        {personalities.map(p => (
          <button
            key={p.id}
            onClick={() => onSwitch(p.id)}
            className={`px-2 py-0.5 rounded text-[11px] font-title transition-colors ${
              current === p.id
                ? 'bg-accent/20 text-accent border border-accent/30'
                : 'bg-background/40 text-text-muted/70 hover:text-text border border-transparent hover:border-white/[0.06]'
            }`}
          >
            {p.name}
          </button>
        ))}
      </div>
    </Card>
  )
}

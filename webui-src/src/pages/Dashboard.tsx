import { useState, useCallback, useEffect } from 'react'
import Card from '../components/Card'
import Console from '../components/Console'
import { api } from '../lib/api'
import { formatNumber, formatTime, truncate } from '../lib/utils'
import type { AppState, ConsoleEntry, Personality } from '../lib/types'
import {
  RiRefreshLine, RiDeleteBinLine, RiSendPlaneLine,
  RiPlayFill, RiPauseFill, RiStopFill, RiVolumeUpFill,
  RiUserVoiceLine, RiBrainLine, RiTimeLine,
  RiVipCrownLine,
} from 'react-icons/ri'
import { TbMicrophone, TbMicrophoneOff, TbPlayerSkipForward } from 'react-icons/tb'

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

  /* Keep volume slider in sync with actual API */
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
    <div className="grid grid-cols-12 gap-4">
      {/* Main column */}
      <div className="col-span-12 lg:col-span-8 flex flex-col gap-4">
        {/* Session toolbar */}
        <Card className="p-3 flex flex-wrap items-center gap-2">
          <ToolbarBtn icon={<RiRefreshLine />} label="Reconnect" onClick={() => act('/api/reconnect')} />
          <ToolbarBtn
            icon={state?.mic_muted ? <TbMicrophoneOff className="text-rose" /> : <TbMicrophone />}
            label={state?.mic_muted ? 'Unmute' : 'Mute'}
            onClick={() => act('/api/toggle-mute')}
          />
          <ToolbarBtn icon={<RiDeleteBinLine />} label="Clear Session" onClick={() => act('/api/clear-session')} />
          <ToolbarBtn icon={<RiDeleteBinLine />} label="Clear Console" onClick={clearLogs} accent={false} />
          <div className="flex-1" />
          {/* Send text */}
          <form
            className="flex gap-1.5"
            onSubmit={(e) => { e.preventDefault(); if (text.trim()) { act('/api/send-text', { text }); setText('') } }}
          >
            <input
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder="Send text..."
              className="bg-background/60 border border-white/[0.08] rounded-lg px-3 py-1.5 text-sm text-text placeholder:text-text-muted/50 w-48 focus:outline-none focus:border-accent/40"
            />
            <button type="submit" className="text-accent hover:text-accent-dim transition-colors p-1.5">
              <RiSendPlaneLine />
            </button>
          </form>
        </Card>

        {/* Console */}
        <Card className="p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="font-title text-sm text-text-muted">Console</span>
            <span className="text-text-muted/40 text-xs font-title">{logs.length} entries</span>
          </div>
          <Console logs={logs} />
        </Card>

        {/* System instruction */}
        <Card className="p-3">
          <form
            className="flex gap-2"
            onSubmit={(e) => { e.preventDefault(); if (sysText.trim()) { act('/api/send-system-instruction', { instruction: sysText }); setSysText('') } }}
          >
            <input
              value={sysText}
              onChange={e => setSysText(e.target.value)}
              placeholder="Send system instruction..."
              className="flex-1 bg-background/60 border border-white/[0.08] rounded-lg px-3 py-1.5 text-sm text-text placeholder:text-text-muted/50 focus:outline-none focus:border-accent/40"
            />
            <button type="submit" className="px-3 py-1.5 bg-accent/15 text-accent rounded-lg text-sm font-title hover:bg-accent/25 transition-colors">
              Inject
            </button>
          </form>
        </Card>
      </div>

      {/* Sidebar */}
      <div className="col-span-12 lg:col-span-4 flex flex-col gap-4">
        {/* Stats */}
        <Card className="p-4">
          <h3 className="font-title text-xs text-text-muted/80 uppercase tracking-wider mb-3">Usage Stats</h3>
          <div className="grid grid-cols-2 gap-3">
            <StatItem icon={<RiSendPlaneLine />} label="Prompt" value={formatNumber(usage?.prompt_tokens)} />
            <StatItem icon={<RiUserVoiceLine />} label="Response" value={formatNumber(usage?.response_tokens)} />
            <StatItem icon={<RiBrainLine />} label="Total" value={formatNumber(usage?.total_tokens)} />
            <StatItem icon={<TbPlayerSkipForward />} label="Tool Calls" value={formatNumber(usage?.tool_calls)} />
          </div>
        </Card>

        {/* Session */}
        {state?.session_handle && (
          <Card className="p-4">
            <h3 className="font-title text-xs text-text-muted/80 uppercase tracking-wider mb-3">Session</h3>
            <div className="space-y-1.5 text-sm">
              <div className="flex justify-between">
                <span className="text-text-muted">Handle</span>
                <span className={state.session_handle.exists ? 'text-mint' : 'text-text-muted'}>
                  {state.session_handle.exists ? 'Active' : 'None'}
                </span>
              </div>
              {state.session_handle.age_minutes !== undefined && (
                <div className="flex justify-between">
                  <span className="text-text-muted">Age</span>
                  <span className="text-text">{Math.round(state.session_handle.age_minutes)}m</span>
                </div>
              )}
            </div>
          </Card>
        )}

        {/* Personality */}
        <PersonalitySelector
          personalities={state?.personalities || []}
          current={state?.current_personality || null}
          onSwitch={(id) => act('/api/switch-personality', { personality_id: id })}
        />

        {/* Now playing */}
        <Card className="p-4">
          <h3 className="font-title text-xs text-text-muted/80 uppercase tracking-wider mb-3">Now Playing</h3>
          {music?.is_playing ? (
            <div className="space-y-3">
              <p className="text-sm text-text truncate">{music.song_name || 'Unknown'}</p>
              {/* Progress bar */}
              <div className="h-1 bg-surface-alt rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent rounded-full transition-all"
                  style={{ width: music.duration > 0 ? `${(music.position / music.duration) * 100}%` : '0%' }}
                />
              </div>
              <div className="flex items-center justify-between text-xs text-text-muted font-title">
                <span>{formatTime(music.position)}</span>
                <span>{formatTime(music.duration)}</span>
              </div>
              {/* Controls */}
              <div className="flex items-center gap-2">
                <button onClick={() => act('/api/pause-music')} className="p-1.5 hover:text-accent transition-colors"><RiPauseFill /></button>
                <button onClick={() => act('/api/resume-music')} className="p-1.5 hover:text-accent transition-colors"><RiPlayFill /></button>
                <button onClick={() => act('/api/stop-music')} className="p-1.5 hover:text-rose transition-colors"><RiStopFill /></button>
                <div className="ml-auto flex items-center gap-1.5">
                  <RiVolumeUpFill className="text-text-muted text-xs" />
                  <input
                    type="range" min={0} max={100} value={volume}
                    onChange={e => { setVolume(+e.target.value); act('/api/set-volume', { volume: +e.target.value }) }}
                    className="w-20 accent-accent h-1"
                  />
                  <span className="text-text-muted text-xs font-title w-7 text-right">{volume}%</span>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-text-muted text-sm italic">Nothing playing</p>
          )}
        </Card>

        {/* VRChat */}
        {state?.vrchat && (
          <Card className="p-4">
            <h3 className="font-title text-xs text-text-muted/80 uppercase tracking-wider mb-3">VRChat</h3>
            <div className="space-y-1.5 text-sm">
              <div className="flex justify-between">
                <span className="text-text-muted">Status</span>
                <span className={state.vrchat.is_in_world ? 'text-mint' : 'text-text-muted'}>
                  {state.vrchat.is_in_world ? 'In World' : 'Offline'}
                </span>
              </div>
              {state.vrchat.player_count > 0 && (
                <div className="flex justify-between">
                  <span className="text-text-muted">Players</span>
                  <span className="text-text">{state.vrchat.player_count}</span>
                </div>
              )}
            </div>
          </Card>
        )}

        {/* Recent memories */}
        <Card className="p-4">
          <h3 className="font-title text-xs text-text-muted/80 uppercase tracking-wider mb-3">Recent Memories</h3>
          <div className="space-y-2 max-h-48 overflow-y-auto console-scroll">
            {state?.recent_memories?.memories?.length ? (
              state.recent_memories.memories.slice(0, 5).map((m, i) => (
                <div key={i} className="text-xs border-l-2 border-accent/30 pl-2 py-0.5">
                  <span className="text-text">{truncate(m.content, 80)}</span>
                  <span className="block text-text-muted/60 mt-0.5">{m.category}</span>
                </div>
              ))
            ) : (
              <p className="text-text-muted text-sm italic">No recent memories</p>
            )}
          </div>
        </Card>
      </div>
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
      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-sm transition-colors ${
        accent
          ? 'bg-accent/10 text-accent hover:bg-accent/20'
          : 'bg-white/[0.04] text-text-muted hover:text-text hover:bg-white/[0.08]'
      }`}
    >
      {icon}
      <span className="hidden sm:inline font-title text-xs">{label}</span>
    </button>
  )
}

function StatItem({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 p-2 rounded-lg bg-background/40">
      <span className="text-accent text-sm">{icon}</span>
      <div>
        <div className="font-title text-sm text-text">{value}</div>
        <div className="text-text-muted text-[10px] font-title uppercase">{label}</div>
      </div>
    </div>
  )
}

function PersonalitySelector({ personalities, current, onSwitch }: {
  personalities: Personality[]; current: string | null; onSwitch: (id: string) => void
}) {
  if (!personalities.length) return null
  return (
    <Card className="p-4">
      <h3 className="font-title text-xs text-text-muted/80 uppercase tracking-wider mb-3">
        <RiVipCrownLine className="inline mr-1" />
        Personality
      </h3>
      <div className="flex flex-wrap gap-1.5">
        {personalities.map(p => (
          <button
            key={p.id}
            onClick={() => onSwitch(p.id)}
            className={`px-2.5 py-1 rounded-lg text-xs font-title transition-colors ${
              current === p.id
                ? 'bg-accent/20 text-accent border border-accent/30'
                : 'bg-background/40 text-text-muted hover:text-text border border-transparent hover:border-white/[0.08]'
            }`}
          >
            {p.name}
          </button>
        ))}
      </div>
    </Card>
  )
}

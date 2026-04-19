import { useState, useCallback, useEffect } from 'react'
import {
  RiUserLine, RiMapPinLine, RiWifiLine, RiWifiOffLine,
  RiCloseLine, RiShieldLine, RiEyeOffLine,
  RiProhibitedLine, RiVolumeMuteLine, RiCursorLine,
  RiRefreshLine,
} from 'react-icons/ri'
import { TbUsers, TbMessageCircleOff } from 'react-icons/tb'
import { api } from '../lib/api'
import type { AppState, VRChatUser } from '../lib/types'

interface Props {
  state: AppState | null
  onToast: (msg: string, level?: string) => void
}

type ModeType = 'block' | 'hideAvatar' | 'showAvatar' | 'mute' | 'unmute' | 'muteChat' | 'unmuteChat' | 'interactOff' | 'interactOn'

// each toggle tracks what moderation type it controls
const TOGGLES: {
  label: string
  modType: ModeType   // the moderation type to toggle (send this to both POST and PUT endpoints)
  icon: React.ReactNode
  color: string
}[] = [
  {
    label: 'Blocked',
    modType: 'block',
    icon: <RiProhibitedLine size={14} />,
    color: 'text-rose-400 border-rose-400/30 bg-rose-400/10',
  },
  {
    label: 'Voice Muted',
    modType: 'mute',
    icon: <RiVolumeMuteLine size={14} />,
    color: 'text-amber-400 border-amber-400/30 bg-amber-400/10',
  },
  {
    label: 'Chat Muted',
    modType: 'muteChat',
    icon: <TbMessageCircleOff size={14} />,
    color: 'text-amber-400 border-amber-400/30 bg-amber-400/10',
  },
  {
    label: 'Avatar Hidden',
    modType: 'hideAvatar',
    icon: <RiEyeOffLine size={14} />,
    color: 'text-blue-400 border-blue-400/30 bg-blue-400/10',
  },
  {
    label: 'Interact Off',
    modType: 'interactOff',
    icon: <RiCursorLine size={14} />,
    color: 'text-purple-400 border-purple-400/30 bg-purple-400/10',
  },
]

export default function Players({ state, onToast }: Props) {
  const vrchat = state?.vrchat
  const players = vrchat?.players ?? []
  const isInWorld = vrchat?.is_in_world ?? false
  const location = vrchat?.location ?? null
  const playerCount = vrchat?.player_count ?? 0

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [userInfo, setUserInfo] = useState<Record<string, VRChatUser>>({})
  const [loadingId, setLoadingId] = useState<string | null>(null)
  const [activeMods, setActiveMods] = useState<Record<string, Set<ModeType>>>({})
  const [modsLoading, setModsLoading] = useState<string | null>(null)
  const [toggling, setToggling] = useState<string | null>(null)

  const loadUser = useCallback(async (id: string, force = false) => {
    if ((!force && userInfo[id]) || loadingId === id || !id) return
    setLoadingId(id)
    try {
      const u = await api<VRChatUser>(`/api/vrchat/user/${id}`)
      setUserInfo(prev => ({ ...prev, [id]: u }))
    } catch { /* no profile pic, thats fine */ }
    setLoadingId(null)
  }, [userInfo, loadingId])

  const loadModerations = useCallback(async (id: string, force = false) => {
    if (!id) return
    if (!force && activeMods[id] !== undefined) return
    setModsLoading(id)
    try {
      const res = await api<{ active: string[] }>(`/api/vrchat/moderations/${id}`)
      setActiveMods(prev => ({ ...prev, [id]: new Set(res.active as ModeType[]) }))
    } catch {
      // 403 or no auth - treat as no active mods, not an error
      setActiveMods(prev => ({ ...prev, [id]: new Set() }))
    }
    setModsLoading(null)
  }, [activeMods])

  useEffect(() => {
    if (selectedId) {
      loadUser(selectedId)
      loadModerations(selectedId)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId])

  const selectPlayer = (id: string) => {
    setSelectedId(prev => prev === id ? null : id)
  }

  const refreshSelected = () => {
    if (!selectedId) return
    loadUser(selectedId, true)
    loadModerations(selectedId, true)
  }

  const toggleMod = async (userId: string, toggle: typeof TOGGLES[0]) => {
    const key = `${userId}-${toggle.modType}`
    const isActive = activeMods[userId]?.has(toggle.modType) ?? false
    setToggling(key)
    try {
      if (isActive) {
        // send the active type to remove it
        await api(`/api/vrchat/unmoderate`, 'PUT', { moderated: userId, type: toggle.modType })
        onToast(`${toggle.label} removed`, 'success')
      } else {
        // send the same type to apply it
        await api(`/api/vrchat/moderate`, 'POST', { moderated: userId, type: toggle.modType })
        onToast(`${toggle.label} applied`, 'success')
      }
      // force refresh mods to sync with VRChat's actual state
      await loadModerations(userId, true)
    } catch (e: unknown) {
      onToast((e as Error).message ?? 'Moderation failed', 'error')
    }
    setToggling(null)
  }

  const selectedPlayer = players.find(p => p.id === selectedId)
  const selectedUser = selectedId ? userInfo[selectedId] : null

  return (
    <div className="flex h-[calc(100vh-3rem)]">
      {/* Left: player list */}
      <div className="flex flex-col w-80 shrink-0 border-r border-white/[0.06] px-3 py-4">
        <div className="flex items-center gap-2 mb-3">
          <TbUsers size={15} className="text-accent" />
          <span className="font-title text-[13px] font-semibold text-text">Instance Players</span>
          <span className="ml-auto px-1.5 py-0.5 bg-accent/15 text-accent rounded text-[11px] font-title tabular-nums">
            {playerCount}
          </span>
          <div className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-title ${
            isInWorld ? 'bg-mint/10 text-mint' : 'bg-white/[0.04] text-text-muted/40'
          }`}>
            {isInWorld ? <RiWifiLine size={10} /> : <RiWifiOffLine size={10} />}
            {isInWorld ? 'Live' : 'Offline'}
          </div>
        </div>

        {location && (
          <div className="flex items-start gap-1.5 bg-surface/50 border border-white/[0.06] rounded-lg px-2.5 py-2 mb-3">
            <RiMapPinLine size={12} className="text-accent/60 mt-0.5 shrink-0" />
            <p className="text-[10px] text-text/60 font-mono break-all leading-relaxed">{location}</p>
          </div>
        )}

        <div className="flex-1 overflow-y-auto console-scroll space-y-1">
          {!isInWorld ? (
            <div className="flex flex-col items-center justify-center h-full text-text-muted/30 gap-2 py-8">
              <RiWifiOffLine size={28} />
              <p className="text-xs font-title text-center">Not in a VRChat world</p>
            </div>
          ) : players.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-text-muted/30 gap-2 py-8">
              <TbUsers size={28} />
              <p className="text-xs font-title">No players detected</p>
            </div>
          ) : players.map((p, i) => {
            const info = userInfo[p.id]
            const thumb = info?.profilePicOverride || info?.currentAvatarThumbnailImageUrl
            const isSelected = selectedId === p.id
            const hasMods = (activeMods[p.id]?.size ?? 0) > 0

            return (
              <button
                key={p.id || i}
                onClick={() => selectPlayer(p.id)}
                className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg transition-colors text-left ${
                  isSelected
                    ? 'bg-accent/10 border border-accent/20'
                    : 'bg-white/[0.02] border border-white/[0.05] hover:bg-white/[0.04] hover:border-white/[0.08]'
                }`}
              >
                <div className="w-8 h-8 rounded-full shrink-0 overflow-hidden bg-accent/10 border border-accent/20 flex items-center justify-center">
                  {thumb ? (
                    <img src={thumb} alt="" className="w-full h-full object-cover" onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
                  ) : (
                    <RiUserLine size={14} className="text-accent/60" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-[12px] text-text/90 font-medium truncate">{p.name}</p>
                  {info?.statusDescription && (
                    <p className="text-[10px] text-text-muted/40 truncate">{info.statusDescription}</p>
                  )}
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  {hasMods && (
                    <span className="w-1.5 h-1.5 rounded-full bg-rose-400/60" title="Has active moderations" />
                  )}
                  <span className="text-[9px] text-text-muted/25 font-title tabular-nums">#{i + 1}</span>
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Right: detail panel */}
      <div className="flex-1 flex flex-col min-w-0">
        {!selectedPlayer ? (
          <div className="flex flex-col items-center justify-center h-full text-text-muted/30 gap-2">
            <RiUserLine size={32} />
            <p className="text-sm font-title">Select a player to view details</p>
          </div>
        ) : (
          <div className="flex flex-col h-full overflow-y-auto console-scroll p-6 gap-5 max-w-xl">
            {/* Profile header */}
            <div className="flex items-start gap-4">
              <div className="w-20 h-20 rounded-xl overflow-hidden bg-accent/10 border border-accent/20 flex items-center justify-center shrink-0">
                {selectedUser?.profilePicOverride || selectedUser?.currentAvatarThumbnailImageUrl ? (
                  <img
                    src={selectedUser.profilePicOverride || selectedUser.currentAvatarThumbnailImageUrl}
                    alt=""
                    className="w-full h-full object-cover"
                  />
                ) : loadingId === selectedPlayer.id ? (
                  <div className="w-5 h-5 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
                ) : (
                  <RiUserLine size={28} className="text-accent/40" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h2 className="font-title text-[16px] font-semibold text-text">{selectedPlayer.name}</h2>
                  {selectedUser?.isFriend && (
                    <span className="px-1.5 py-0.5 bg-mint/10 text-mint rounded text-[10px] font-title">Friend</span>
                  )}
                </div>
                {selectedUser?.statusDescription && (
                  <p className="text-[12px] text-text-muted/60 mb-1">{selectedUser.statusDescription}</p>
                )}
                <div className="flex items-center gap-2 flex-wrap">
                  {selectedUser?.status && (
                    <span className="text-[10px] font-title text-text-muted/40 capitalize">{selectedUser.status}</span>
                  )}
                  {selectedUser?.last_platform && (
                    <span className="text-[10px] font-title text-text-muted/30">{selectedUser.last_platform}</span>
                  )}
                  <span className="text-[10px] font-mono text-text-muted/25">{selectedPlayer.id}</span>
                </div>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <button
                  onClick={refreshSelected}
                  disabled={!!loadingId || !!modsLoading}
                  title="Refresh profile & moderations"
                  className="p-1.5 text-text-muted/40 hover:text-accent rounded-lg hover:bg-white/[0.04] transition-colors disabled:opacity-30"
                >
                  <RiRefreshLine size={15} />
                </button>
                <button
                  onClick={() => setSelectedId(null)}
                  className="p-1.5 text-text-muted/40 hover:text-text-muted rounded-lg hover:bg-white/[0.04] transition-colors"
                >
                  <RiCloseLine size={16} />
                </button>
              </div>
            </div>

            {/* Bio */}
            {selectedUser?.bio && (
              <div className="bg-surface/40 border border-white/[0.05] rounded-lg p-3">
                <p className="text-[10px] font-title text-text-muted/40 uppercase tracking-wider mb-1.5">Bio</p>
                <p className="text-[12px] text-text/70 leading-relaxed">{selectedUser.bio}</p>
              </div>
            )}

            {/* Moderation toggles */}
            <div className="bg-surface/40 border border-white/[0.05] rounded-lg p-4">
              <div className="flex items-center gap-2 mb-4">
                <RiShieldLine size={13} className="text-accent/60" />
                <p className="text-[11px] font-title font-semibold text-text/70 uppercase tracking-wider">Moderation</p>
                {modsLoading === selectedPlayer.id && (
                  <div className="ml-auto w-3 h-3 border border-accent/30 border-t-accent rounded-full animate-spin" />
                )}
              </div>

              <div className="grid grid-cols-2 gap-2">
                {TOGGLES.map(toggle => {
                  const isActive = activeMods[selectedPlayer.id]?.has(toggle.modType) ?? false
                  const key = `${selectedPlayer.id}-${toggle.modType}`
                  const isToggling = toggling === key

                  return (
                    <button
                      key={toggle.modType}
                      onClick={() => toggleMod(selectedPlayer.id, toggle)}
                      disabled={isToggling || modsLoading === selectedPlayer.id}
                      title={isActive ? `Remove: ${toggle.label}` : `Apply: ${toggle.label}`}
                      className={`flex items-center gap-2 px-3 py-2.5 rounded-lg border text-[12px] font-title transition-all disabled:opacity-40 ${
                        isActive
                          ? toggle.color
                          : 'text-text-muted/50 border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.06] hover:text-text-muted/70'
                      }`}
                    >
                      {isToggling ? (
                        <div className="w-3.5 h-3.5 border border-current border-t-transparent rounded-full animate-spin shrink-0" />
                      ) : (
                        <span className="shrink-0">{toggle.icon}</span>
                      )}
                      <span className="flex-1 text-left">{toggle.label}</span>
                      {/* pill toggle indicator */}
                      <span className={`relative w-7 h-4 rounded-full shrink-0 transition-colors ${isActive ? 'bg-current' : 'bg-white/[0.10]'}`}>
                        <span className={`absolute top-0.5 w-3 h-3 rounded-full transition-all ${isActive ? 'right-0.5 bg-background/80' : 'left-0.5 bg-white/30'}`} />
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

import { HiOutlineWifi, HiOutlineStatusOffline } from 'react-icons/hi'
import { RiDashboardLine, RiBrainLine, RiMusicLine } from 'react-icons/ri'
import { TbMicrophoneOff, TbUsers } from 'react-icons/tb'

type Tab = 'dashboard' | 'memories' | 'music' | 'players'

interface NavbarProps {
  appName: string
  isConnected: boolean
  isMuted: boolean
  activeTab: Tab
  onTabChange: (tab: Tab) => void
}

const tabs: { id: Tab; label: string; icon: typeof RiDashboardLine }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: RiDashboardLine },
  { id: 'memories', label: 'Memories', icon: RiBrainLine },
  { id: 'music', label: 'Music', icon: RiMusicLine },
  { id: 'players', label: 'Players', icon: TbUsers },
]

export default function Navbar({ appName, isConnected, isMuted, activeTab, onTabChange }: NavbarProps) {
  return (
    <header className="sticky top-0 z-50 bg-surface/90 backdrop-blur-xl border-b border-white/[0.06]">
      <div className="flex items-center h-12 px-4">
        {/* Brand */}
        <div className="flex items-baseline gap-2 shrink-0 mr-6">
          <span className="font-title font-bold text-text text-[15px] tracking-tight">
            {appName}
          </span>
          <span className="text-text-muted/40 text-[10px] font-title uppercase tracking-widest">
            Panel
          </span>
        </div>

        {/* Divider */}
        <div className="w-px h-5 bg-white/[0.08] mr-4 shrink-0" />

        {/* Tabs */}
        <nav className="flex items-center gap-0.5">
          {tabs.map(t => {
            const active = activeTab === t.id
            return (
              <button
                key={t.id}
                onClick={() => onTabChange(t.id)}
                className={`relative flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium transition-all duration-150 ${
                  active
                    ? 'text-text bg-white/[0.07]'
                    : 'text-text-muted/60 hover:text-text-muted hover:bg-white/[0.03]'
                }`}
              >
                <t.icon size={14} className={active ? 'text-accent' : ''} />
                {t.label}
              </button>
            )
          })}
        </nav>

        <div className="flex-1" />

        {/* Status badges */}
        <div className="flex items-center gap-2">
          {isMuted && (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-rose/10 text-rose text-[11px] font-title">
              <TbMicrophoneOff size={13} />
              <span>Muted</span>
            </div>
          )}
          <div className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-title ${
            isConnected
              ? 'bg-mint/10 text-mint'
              : 'bg-white/[0.04] text-text-muted/50'
          }`}>
            {isConnected ? (
              <>
                <HiOutlineWifi size={13} />
                <span>Connected</span>
              </>
            ) : (
              <>
                <HiOutlineStatusOffline size={13} />
                <span>Offline</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}

import { HiOutlineWifi, HiOutlineStatusOffline } from 'react-icons/hi'
import { RiDashboardLine, RiBrainLine, RiMusicLine } from 'react-icons/ri'
import { TbMicrophoneOff } from 'react-icons/tb'

type Tab = 'dashboard' | 'memories' | 'music'

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
]

export default function Navbar({ appName, isConnected, isMuted, activeTab, onTabChange }: NavbarProps) {
  return (
    <header className="sticky top-0 z-50 bg-surface/80 backdrop-blur-md border-b border-white/[0.06]">
      <div className="max-w-[1600px] mx-auto px-4 h-14 flex items-center gap-6">
        {/* Logo */}
        <div className="flex items-center gap-2.5 shrink-0">
          <span className="font-title font-bold text-accent text-lg tracking-tight">
            {appName}
          </span>
          <span className="text-text-muted text-xs font-title">Control Panel</span>
        </div>

        {/* Tabs */}
        <nav className="flex gap-1 ml-4">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => onTabChange(t.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === t.id
                  ? 'bg-accent/15 text-accent'
                  : 'text-text-muted hover:text-text hover:bg-white/[0.04]'
              }`}
            >
              <t.icon className="text-base" />
              {t.label}
            </button>
          ))}
        </nav>

        {/* Status */}
        <div className="ml-auto flex items-center gap-3">
          {isMuted && (
            <span className="flex items-center gap-1 text-rose text-xs font-title">
              <TbMicrophoneOff />
              Muted
            </span>
          )}
          <span className={`flex items-center gap-1.5 text-xs font-title ${
            isConnected ? 'text-mint' : 'text-text-muted'
          }`}>
            {isConnected ? (
              <>
                <HiOutlineWifi className="text-sm" />
                <span className="w-1.5 h-1.5 rounded-full bg-mint animate-pulse-dot" />
                Connected
              </>
            ) : (
              <>
                <HiOutlineStatusOffline className="text-sm" />
                Disconnected
              </>
            )}
          </span>
        </div>
      </div>
    </header>
  )
}

import { useState, useCallback } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Memories from './pages/Memories'
import Music from './pages/Music'
import Toast, { type ToastItem } from './components/Toast'

type Tab = 'dashboard' | 'memories' | 'music'

let toastId = 0

export default function App() {
  const [tab, setTab] = useState<Tab>('dashboard')
  const [toasts, setToasts] = useState<ToastItem[]>([])

  const addToast = useCallback((message: string, level: string = 'info') => {
    const id = ++toastId
    setToasts(prev => [...prev, { id, message, level }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000)
  }, [])

  const { state, logs, clearLogs } = useWebSocket(addToast)

  return (
    <div className="min-h-screen bg-background">
      <Navbar
        appName={state?.app_name || 'Gabriel'}
        isConnected={state?.is_connected ?? false}
        isMuted={state?.mic_muted ?? false}
        activeTab={tab}
        onTabChange={setTab}
      />
      <main className={tab === 'dashboard' ? '' : 'max-w-[1600px] mx-auto px-4 py-4'}>
        {tab === 'dashboard' && (
          <Dashboard state={state} logs={logs} clearLogs={clearLogs} onToast={addToast} />
        )}
        {tab === 'memories' && <Memories onToast={addToast} />}
        {tab === 'music' && <Music onToast={addToast} />}
      </main>
      <Toast toasts={toasts} />
    </div>
  )
}

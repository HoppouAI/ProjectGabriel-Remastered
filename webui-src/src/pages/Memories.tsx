import { useState, useEffect, useCallback } from 'react'
import Card from '../components/Card'
import Modal from '../components/Modal'
import { api } from '../lib/api'
import { formatDate, truncate } from '../lib/utils'
import type { Memory, MemoryStats } from '../lib/types'
import {
  RiSearchLine, RiAddLine, RiDeleteBinLine, RiPushpinLine,
  RiEditLine, RiRefreshLine, RiBrainLine,
} from 'react-icons/ri'

interface Props {
  onToast: (msg: string, level?: string) => void
}

const TYPES = ['all', 'long_term', 'short_term', 'quick_note']

export default function Memories({ onToast }: Props) {
  const [memories, setMemories] = useState<Memory[]>([])
  const [stats, setStats] = useState<MemoryStats | null>(null)
  const [search, setSearch] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const [loading, setLoading] = useState(false)
  const [editMem, setEditMem] = useState<Memory | null>(null)
  const [createOpen, setCreateOpen] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (search) params.set('search', search)
      if (typeFilter !== 'all') params.set('type', typeFilter)
      params.set('limit', '100')
      const qs = params.toString()
      const [mems, st] = await Promise.all([
        api<Memory[]>(`/api/memories?${qs}`),
        api<MemoryStats>('/api/memories/stats'),
      ])
      setMemories(mems)
      setStats(st)
    } catch (e: unknown) {
      onToast((e as Error).message, 'error')
    }
    setLoading(false)
  }, [search, typeFilter, onToast])

  useEffect(() => { load() }, [load])

  const deleteMem = async (key: string) => {
    try {
      await api(`/api/memories/${encodeURIComponent(key)}`, 'DELETE')
      onToast('Memory deleted', 'success')
      load()
    } catch (e: unknown) {
      onToast((e as Error).message, 'error')
    }
  }

  const pinMem = async (key: string) => {
    try {
      await api(`/api/memories/${encodeURIComponent(key)}/pin`, 'POST')
      onToast('Pin toggled', 'success')
      load()
    } catch (e: unknown) {
      onToast((e as Error).message, 'error')
    }
  }

  return (
    <div className="space-y-4">
      {/* Stats row */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatCard label="Total" value={stats.total} color="text-accent" />
          <StatCard label="Long Term" value={stats.long_term} color="text-mint" />
          <StatCard label="Short Term" value={stats.short_term} color="text-text" />
          <StatCard label="Quick Notes" value={stats.quick_note} color="text-text-muted" />
        </div>
      )}

      {/* Toolbar */}
      <Card className="p-3 flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[200px]">
          <RiSearchLine className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search memories..."
            className="w-full bg-background/60 border border-white/[0.08] rounded-lg pl-9 pr-3 py-1.5 text-sm text-text placeholder:text-text-muted/50 focus:outline-none focus:border-accent/40"
          />
        </div>
        <div className="flex gap-1">
          {TYPES.map(t => (
            <button
              key={t}
              onClick={() => setTypeFilter(t)}
              className={`px-2.5 py-1 rounded-lg text-xs font-title transition-colors ${
                typeFilter === t
                  ? 'bg-accent/20 text-accent'
                  : 'text-text-muted hover:text-text hover:bg-white/[0.04]'
              }`}
            >
              {t === 'all' ? 'All' : t.replace('_', ' ')}
            </button>
          ))}
        </div>
        <button onClick={() => setCreateOpen(true)} className="flex items-center gap-1 px-2.5 py-1.5 bg-accent/15 text-accent rounded-lg text-xs font-title hover:bg-accent/25 transition-colors">
          <RiAddLine /> Add
        </button>
        <button onClick={load} className="p-1.5 text-text-muted hover:text-text transition-colors">
          <RiRefreshLine className={loading ? 'animate-spin' : ''} />
        </button>
      </Card>

      {/* Memory list */}
      <Card className="divide-y divide-white/[0.04]">
        {memories.length === 0 ? (
          <div className="p-8 text-center text-text-muted">
            <RiBrainLine className="mx-auto text-3xl mb-2 text-text-muted/40" />
            <p className="text-sm">No memories found</p>
          </div>
        ) : memories.map(m => (
          <div key={m.key} className="px-4 py-3 hover:bg-white/[0.02] transition-colors group">
            <div className="flex items-start gap-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <TypeBadge type={m.memory_type} />
                  <span className="text-text-muted/60 text-[10px] font-title">{m.category}</span>
                  <span className="text-text-muted/40 text-[10px] font-title ml-auto">{formatDate(m.created_at)}</span>
                </div>
                <p className="text-sm text-text">{truncate(m.content, 200)}</p>
                {m.tags?.length > 0 && (
                  <div className="flex gap-1 mt-1.5">
                    {m.tags.slice(0, 5).map(tag => (
                      <span key={tag} className="px-1.5 py-0.5 bg-surface-alt rounded text-[10px] text-text-muted font-title">
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                <ActionBtn icon={<RiPushpinLine />} onClick={() => pinMem(m.key)} title="Pin" />
                <ActionBtn icon={<RiEditLine />} onClick={() => setEditMem(m)} title="Edit" />
                <ActionBtn icon={<RiDeleteBinLine />} onClick={() => deleteMem(m.key)} title="Delete" className="hover:text-rose" />
              </div>
            </div>
          </div>
        ))}
      </Card>

      {/* Edit Modal */}
      {editMem && (
        <EditModal memory={editMem} onClose={() => setEditMem(null)} onSaved={() => { setEditMem(null); load() }} onToast={onToast} />
      )}

      {/* Create Modal */}
      {createOpen && (
        <CreateModal onClose={() => setCreateOpen(false)} onSaved={() => { setCreateOpen(false); load() }} onToast={onToast} />
      )}
    </div>
  )
}

/* -- Subcomponents -- */

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <Card className="p-3 text-center">
      <div className={`font-title text-2xl font-bold ${color}`}>{value}</div>
      <div className="text-text-muted text-[10px] font-title uppercase mt-1">{label}</div>
    </Card>
  )
}

function TypeBadge({ type }: { type: string }) {
  const styles: Record<string, string> = {
    long_term: 'bg-mint/15 text-mint',
    short_term: 'bg-accent/15 text-accent',
    quick_note: 'bg-white/[0.06] text-text-muted',
  }
  return (
    <span className={`px-1.5 py-0.5 rounded text-[10px] font-title ${styles[type] || styles.quick_note}`}>
      {type.replace('_', ' ')}
    </span>
  )
}

function ActionBtn({ icon, onClick, title, className = '' }: {
  icon: React.ReactNode; onClick: () => void; title: string; className?: string
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`p-1 rounded hover:bg-white/[0.06] text-text-muted transition-colors text-sm ${className}`}
    >
      {icon}
    </button>
  )
}

/* -- Edit Modal -- */
function EditModal({ memory, onClose, onSaved, onToast }: {
  memory: Memory; onClose: () => void; onSaved: () => void; onToast: (msg: string, level?: string) => void
}) {
  const [content, setContent] = useState(memory.content)
  const [category, setCategory] = useState(memory.category)
  const [tags, setTags] = useState(memory.tags?.join(', ') || '')
  const [saving, setSaving] = useState(false)

  const save = async () => {
    setSaving(true)
    try {
      await api(`/api/memories/${encodeURIComponent(memory.key)}`, 'PUT', {
        content,
        category,
        tags: tags.split(',').map(t => t.trim()).filter(Boolean),
      })
      onToast('Memory updated', 'success')
      onSaved()
    } catch (e: unknown) {
      onToast((e as Error).message, 'error')
    }
    setSaving(false)
  }

  return (
    <Modal open title="Edit Memory" onClose={onClose}>
      <div className="space-y-3">
        <Field label="Content">
          <textarea value={content} onChange={e => setContent(e.target.value)} rows={4} className="form-input" />
        </Field>
        <Field label="Category">
          <input value={category} onChange={e => setCategory(e.target.value)} className="form-input" />
        </Field>
        <Field label="Tags (comma separated)">
          <input value={tags} onChange={e => setTags(e.target.value)} className="form-input" />
        </Field>
        <div className="flex justify-end gap-2 pt-2">
          <button onClick={onClose} className="px-3 py-1.5 text-text-muted text-sm rounded-lg hover:bg-white/[0.04]">Cancel</button>
          <button onClick={save} disabled={saving} className="px-4 py-1.5 bg-accent text-background font-title text-sm rounded-lg hover:bg-accent-dim disabled:opacity-50 transition-colors">
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </Modal>
  )
}

/* -- Create Modal -- */
function CreateModal({ onClose, onSaved, onToast }: {
  onClose: () => void; onSaved: () => void; onToast: (msg: string, level?: string) => void
}) {
  const [content, setContent] = useState('')
  const [category, setCategory] = useState('general')
  const [memType, setMemType] = useState('long_term')
  const [tags, setTags] = useState('')
  const [saving, setSaving] = useState(false)

  const save = async () => {
    if (!content.trim()) { onToast('Content required', 'warning'); return }
    setSaving(true)
    try {
      await api('/api/memories', 'POST', {
        content,
        category,
        memory_type: memType,
        tags: tags.split(',').map(t => t.trim()).filter(Boolean),
      })
      onToast('Memory created', 'success')
      onSaved()
    } catch (e: unknown) {
      onToast((e as Error).message, 'error')
    }
    setSaving(false)
  }

  return (
    <Modal open title="Create Memory" onClose={onClose}>
      <div className="space-y-3">
        <Field label="Content">
          <textarea value={content} onChange={e => setContent(e.target.value)} rows={4} className="form-input" placeholder="Memory content..." />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Category">
            <input value={category} onChange={e => setCategory(e.target.value)} className="form-input" />
          </Field>
          <Field label="Type">
            <select value={memType} onChange={e => setMemType(e.target.value)} className="form-input">
              <option value="long_term">Long Term</option>
              <option value="short_term">Short Term</option>
              <option value="quick_note">Quick Note</option>
            </select>
          </Field>
        </div>
        <Field label="Tags (comma separated)">
          <input value={tags} onChange={e => setTags(e.target.value)} className="form-input" placeholder="tag1, tag2" />
        </Field>
        <div className="flex justify-end gap-2 pt-2">
          <button onClick={onClose} className="px-3 py-1.5 text-text-muted text-sm rounded-lg hover:bg-white/[0.04]">Cancel</button>
          <button onClick={save} disabled={saving} className="px-4 py-1.5 bg-accent text-background font-title text-sm rounded-lg hover:bg-accent-dim disabled:opacity-50 transition-colors">
            {saving ? 'Creating...' : 'Create'}
          </button>
        </div>
      </div>
    </Modal>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs font-title text-text-muted mb-1">{label}</label>
      {children}
    </div>
  )
}

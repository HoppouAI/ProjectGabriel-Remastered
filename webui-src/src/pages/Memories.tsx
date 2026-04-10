import { useState, useEffect, useCallback, useMemo } from 'react'
import Card from '../components/Card'
import Modal from '../components/Modal'
import { api } from '../lib/api'
import { formatDate, truncate } from '../lib/utils'
import type { Memory, MemoryStats } from '../lib/types'
import {
  RiSearchLine, RiAddLine, RiDeleteBinLine, RiPushpinLine,
  RiEditLine, RiRefreshLine, RiBrainLine, RiSortDesc, RiSortAsc,
} from 'react-icons/ri'

interface Props {
  onToast: (msg: string, level?: string) => void
}

const TYPES = ['all', 'long_term', 'short_term', 'quick_note']
type SortKey = 'newest' | 'oldest' | 'category'

export default function Memories({ onToast }: Props) {
  const [memories, setMemories] = useState<Memory[]>([])
  const [stats, setStats] = useState<MemoryStats | null>(null)
  const [search, setSearch] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const [sortBy, setSortBy] = useState<SortKey>('newest')
  const [loading, setLoading] = useState(false)
  const [editMem, setEditMem] = useState<Memory | null>(null)
  const [createOpen, setCreateOpen] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (search) params.set('search', search)
      if (typeFilter !== 'all') params.set('memory_type', typeFilter)
      params.set('limit', '200')
      const qs = params.toString()
      const [memsResp, st] = await Promise.all([
        api<{ memories: Memory[]; count: number }>(`/api/memories?${qs}`),
        api<MemoryStats>('/api/memories/stats'),
      ])
      setMemories(memsResp.memories)
      setStats(st)
    } catch (e: unknown) {
      onToast((e as Error).message, 'error')
    }
    setLoading(false)
  }, [search, typeFilter, onToast])

  useEffect(() => { load() }, [load])

  const sorted = useMemo(() => {
    const arr = [...memories]
    if (sortBy === 'oldest') arr.reverse()
    else if (sortBy === 'category') arr.sort((a, b) => a.category.localeCompare(b.category))
    return arr
  }, [memories, sortBy])

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

  const cycleSortBy = () => {
    const order: SortKey[] = ['newest', 'oldest', 'category']
    setSortBy(order[(order.indexOf(sortBy) + 1) % order.length])
  }

  return (
    <div className="flex flex-col h-[calc(100vh-3.5rem)]">
      {/* Stats row */}
      {stats && (
        <div className="grid grid-cols-4 gap-3 px-4 pt-4 pb-2 shrink-0">
          <StatCard label="Total" value={stats.total} color="text-accent" />
          <StatCard label="Long Term" value={stats.long_term} color="text-mint" />
          <StatCard label="Short Term" value={stats.short_term} color="text-text" />
          <StatCard label="Quick Notes" value={stats.quick_note} color="text-text-muted" />
        </div>
      )}

      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-2 border-b border-white/[0.06] shrink-0">
        <div className="relative flex-1 min-w-[180px]">
          <RiSearchLine className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted/50 text-sm" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search memories..."
            className="w-full bg-white/[0.03] border border-white/[0.08] rounded-lg pl-9 pr-3 py-1.5 text-sm text-text placeholder:text-text-muted/40 focus:outline-none focus:border-accent/30"
          />
        </div>
        <div className="flex gap-0.5 bg-white/[0.03] rounded-lg p-0.5">
          {TYPES.map(t => (
            <button
              key={t}
              onClick={() => setTypeFilter(t)}
              className={`px-2.5 py-1 rounded-md text-xs font-title transition-colors ${
                typeFilter === t
                  ? 'bg-accent/20 text-accent'
                  : 'text-text-muted hover:text-text'
              }`}
            >
              {t === 'all' ? 'All' : t.replace('_', ' ')}
            </button>
          ))}
        </div>
        <button
          onClick={cycleSortBy}
          title={`Sort: ${sortBy}`}
          className="flex items-center gap-1 px-2 py-1.5 text-text-muted hover:text-text text-xs font-title rounded-lg hover:bg-white/[0.04] transition-colors"
        >
          {sortBy === 'oldest' ? <RiSortAsc size={14} /> : <RiSortDesc size={14} />}
          {sortBy}
        </button>
        <button onClick={() => setCreateOpen(true)} className="flex items-center gap-1 px-2.5 py-1.5 bg-accent/15 text-accent rounded-lg text-xs font-title hover:bg-accent/25 transition-colors">
          <RiAddLine size={14} /> New
        </button>
        <button onClick={load} className="p-1.5 text-text-muted hover:text-text transition-colors">
          <RiRefreshLine size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Memory list -- scrollable */}
      <div className="flex-1 overflow-y-auto console-scroll px-4 py-2">
        {sorted.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-text-muted/60">
            <RiBrainLine className="text-4xl mb-2" />
            <p className="text-sm">No memories found</p>
          </div>
        ) : (
          <div className="grid gap-2">
            {sorted.map(m => (
              <div
                key={m.key}
                className="bg-white/[0.02] border border-white/[0.05] rounded-lg px-3.5 py-2.5 hover:bg-white/[0.04] hover:border-white/[0.08] transition-colors group"
              >
                <div className="flex items-start gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <TypeBadge type={m.memory_type} />
                      <span className="text-text-muted/50 text-[10px] font-title">{m.category}</span>
                      <span className="text-text-muted/30 text-[10px] font-title ml-auto">{formatDate(m.created_at)}</span>
                    </div>
                    <p className="text-[13px] text-text/90 leading-relaxed">{truncate(m.content, 180)}</p>
                    {m.tags?.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {m.tags.slice(0, 5).map(tag => (
                          <span key={tag} className="px-1.5 py-0.5 bg-white/[0.04] rounded text-[10px] text-text-muted/60 font-title">
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                    <ActionBtn icon={<RiPushpinLine />} onClick={() => pinMem(m.key)} title="Pin" />
                    <ActionBtn icon={<RiEditLine />} onClick={() => setEditMem(m)} title="Edit" />
                    <ActionBtn icon={<RiDeleteBinLine />} onClick={() => deleteMem(m.key)} title="Delete" className="hover:text-rose" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

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
    <div className="bg-white/[0.03] border border-white/[0.05] rounded-lg p-3 text-center">
      <div className={`font-title text-xl font-bold ${color}`}>{value}</div>
      <div className="text-text-muted/60 text-[10px] font-title uppercase mt-0.5">{label}</div>
    </div>
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

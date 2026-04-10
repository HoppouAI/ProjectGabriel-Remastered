import { useState, useEffect, useCallback, useRef } from 'react'
import Card from '../components/Card'
import { api } from '../lib/api'
import type { MusicFile } from '../lib/types'
import {
  RiPlayFill, RiDeleteBinLine, RiUploadCloud2Line,
  RiFolderMusicLine, RiMusicLine, RiRefreshLine,
  RiFolderOpenLine,
} from 'react-icons/ri'

interface Props {
  onToast: (msg: string, level?: string) => void
}

export default function Music({ onToast }: Props) {
  const [files, setFiles] = useState<MusicFile[]>([])
  const [folders, setFolders] = useState<string[]>([])
  const [activeFolder, setActiveFolder] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [sevenZip, setSevenZip] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [f, fl, sz] = await Promise.all([
        api<MusicFile[]>('/api/music-files'),
        api<string[]>('/api/music-folders'),
        api<{ available: boolean }>('/api/sevenzip-status'),
      ])
      setFiles(f)
      setFolders(fl)
      setSevenZip(sz.available)
    } catch (e: unknown) {
      onToast((e as Error).message, 'error')
    }
    setLoading(false)
  }, [onToast])

  useEffect(() => { load() }, [load])

  const filtered = activeFolder
    ? files.filter(f => f.folder === activeFolder)
    : files

  const play = async (name: string) => {
    try {
      await api('/api/play-music', 'POST', { filename: name })
      onToast('Playing: ' + name, 'success')
    } catch (e: unknown) {
      onToast((e as Error).message, 'error')
    }
  }

  const deleteFile = async (path: string) => {
    try {
      await api(`/api/music-files/${encodeURIComponent(path)}`, 'DELETE')
      onToast('Deleted', 'success')
      load()
    } catch (e: unknown) {
      onToast((e as Error).message, 'error')
    }
  }

  const upload = async (fileList: FileList) => {
    if (!fileList.length) return
    setUploading(true)
    const form = new FormData()
    for (let i = 0; i < fileList.length; i++) {
      form.append('files', fileList[i])
    }
    try {
      const resp = await fetch('/api/music-upload', { method: 'POST', body: form })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || 'Upload failed')
      onToast(`Uploaded ${data.uploaded?.length || fileList.length} file(s)`, 'success')
      load()
    } catch (e: unknown) {
      onToast((e as Error).message, 'error')
    }
    setUploading(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files.length) upload(e.dataTransfer.files)
  }

  const openFolder = async () => {
    try {
      await api('/api/open-music-folder', 'POST')
    } catch {
      // ignored
    }
  }

  return (
    <div className="space-y-4">
      {/* Upload zone */}
      <Card
        className={`p-8 border-2 border-dashed transition-colors cursor-pointer ${
          dragOver ? 'border-accent bg-accent/[0.04]' : 'border-white/[0.08] hover:border-accent/30'
        }`}
      >
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          className="flex flex-col items-center gap-3"
        >
          <RiUploadCloud2Line className="text-4xl text-accent/60" />
          <div className="text-center">
            <p className="text-sm text-text">Drop music files here or click to upload</p>
            <p className="text-xs text-text-muted mt-1">
              Supports .mp3, .wav, .ogg, .flac, .m4a
              {sevenZip && ', .7z, .zip, .rar'}
            </p>
          </div>
          {uploading && (
            <div className="text-xs text-accent font-title animate-pulse">Uploading...</div>
          )}
        </div>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept=".mp3,.wav,.ogg,.flac,.m4a,.7z,.zip,.rar"
          onChange={e => e.target.files && upload(e.target.files)}
          className="hidden"
        />
      </Card>

      {/* Toolbar */}
      <Card className="p-3 flex items-center gap-2">
        <div className="flex gap-1 flex-1 flex-wrap">
          <button
            onClick={() => setActiveFolder(null)}
            className={`px-2.5 py-1 rounded-lg text-xs font-title transition-colors ${
              !activeFolder ? 'bg-accent/20 text-accent' : 'text-text-muted hover:text-text'
            }`}
          >
            All ({files.length})
          </button>
          {folders.map(f => (
            <button
              key={f}
              onClick={() => setActiveFolder(f)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-title transition-colors ${
                activeFolder === f ? 'bg-accent/20 text-accent' : 'text-text-muted hover:text-text'
              }`}
            >
              <RiFolderMusicLine className="text-xs" />
              {f || 'root'}
            </button>
          ))}
        </div>
        <button onClick={openFolder} className="p-1.5 text-text-muted hover:text-accent transition-colors" title="Open folder">
          <RiFolderOpenLine />
        </button>
        <button onClick={load} className="p-1.5 text-text-muted hover:text-text transition-colors" title="Refresh">
          <RiRefreshLine className={loading ? 'animate-spin' : ''} />
        </button>
      </Card>

      {/* File list */}
      <Card className="divide-y divide-white/[0.04]">
        {filtered.length === 0 ? (
          <div className="p-8 text-center text-text-muted">
            <RiMusicLine className="mx-auto text-3xl mb-2 text-text-muted/40" />
            <p className="text-sm">No music files</p>
          </div>
        ) : filtered.map(f => (
          <div key={f.name} className="px-4 py-2.5 flex items-center gap-3 hover:bg-white/[0.02] transition-colors group">
            <RiMusicLine className="text-accent/60 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-text truncate">{f.display_name}</p>
              <p className="text-[10px] text-text-muted font-title">{f.size_mb.toFixed(1)} MB</p>
            </div>
            <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={() => play(f.name)}
                className="p-1.5 rounded hover:bg-accent/15 text-accent transition-colors"
                title="Play"
              >
                <RiPlayFill />
              </button>
              <button
                onClick={() => deleteFile(f.name)}
                className="p-1.5 rounded hover:bg-rose/15 text-text-muted hover:text-rose transition-colors"
                title="Delete"
              >
                <RiDeleteBinLine />
              </button>
            </div>
          </div>
        ))}
      </Card>
    </div>
  )
}

import { useEffect, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { TbPlayerPlay, TbPlayerStop, TbRobot, TbCrosshair, TbAlertTriangle, TbTrash, TbDatabase, TbSettings, TbRun, TbEdit, TbSparkles } from 'react-icons/tb'
import { api } from '../lib/api'

interface Pose { x: number; y: number; z: number; yaw: number }
interface Counts { reach: number; wall: number; iffy: number; total: number }
interface MappingState {
  running: boolean
  explore: boolean
  manual?: boolean
  world: string
  world_name: string
  pose: Pose | null
  target: [number, number, number] | null
  action: string
  counts: Counts
  decode_rate: number
  last_error: string
  settings?: { tick_hz: number; force_run: boolean; manual_wall_distance?: number; manual_wall_ratio?: number }
  follow?: { active: boolean; remaining: number; label: string }
}
interface WorldCells {
  world: string
  reach: [number, number, number][]
  wall: [number, number, number][]
  iffy: [number, number, number][]
}
interface PathResult {
  found: boolean
  reason?: string
  start?: [number, number, number]
  goal?: [number, number, number]
  full?: [number, number, number][]
  filtered?: [number, number, number][]
}
interface SavedWorld { world: string; size_kb: number; is_current: boolean }

const CELL = 0.25
const HALF = CELL / 2
// starting capacity per InstancedMesh. packCells grows past this on demand
// (big worlds can easily blow past 12k cells per type).
const CAP = 16000

interface Props {
  onToast: (msg: string, level?: string) => void
}

export default function Mapping({ onToast }: Props) {
  const mountRef = useRef<HTMLDivElement>(null)
  const [state, setState] = useState<MappingState | null>(null)
  const [busy, setBusy] = useState(false)
  const [pickEnabled, setPickEnabled] = useState(false)
  const [path, setPath] = useState<PathResult | null>(null)
  const [showWorlds, setShowWorlds] = useState(false)
  const [savedWorlds, setSavedWorlds] = useState<SavedWorld[]>([])
  const [showSettings, setShowSettings] = useState(false)
  const [tickHz, setTickHz] = useState(20)
  const [forceRun, setForceRun] = useState(false)
  const [wallDist, setWallDist] = useState(0.35)
  const [editMode, setEditMode] = useState(false)
  const [editMenu, setEditMenu] = useState<
    { x: number; y: number; serial: [number, number, number] } | null
  >(null)
  // drag-rectangle selection state
  const [dragRect, setDragRect] = useState<
    { x0: number; y0: number; x1: number; y1: number } | null
  >(null)
  const dragStateRef = useRef<{
    active: boolean
    started: boolean   // true once the cursor actually moved past the threshold
    x0: number; y0: number
  } | null>(null)
  const [bulkMenu, setBulkMenu] = useState<
    { x: number; y: number; cells: [number, number, number][] } | null
  >(null)

  const suppressClickRef = useRef(false)

  // three.js scene refs (kept in refs so React doesnt re-create them)
  const sceneRefs = useRef<{
    renderer?: THREE.WebGLRenderer
    scene?: THREE.Scene
    camera?: THREE.PerspectiveCamera
    controls?: OrbitControls
    meshReach?: THREE.InstancedMesh
    meshWall?: THREE.InstancedMesh
    meshIffy?: THREE.InstancedMesh
    pathLine?: THREE.Line
    player?: THREE.Mesh
    target?: THREE.Mesh
    plane?: THREE.Mesh
    raycaster?: THREE.Raycaster
    firstPose?: boolean
    dispose?: () => void
  }>({})

  // -----------------------------------------------------------------
  // three.js init -- once
  // -----------------------------------------------------------------
  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    const width = mount.clientWidth
    const height = mount.clientHeight

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0a0d12)
    // soft distance fade -- pushed far out so big worlds still render fully
    scene.fog = new THREE.Fog(0x0a0d12, 200, 600)

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.05, 2000)
    camera.position.set(6, 8, 6)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, 1, 0)
    controls.enableDamping = true
    controls.dampingFactor = 0.08

    scene.add(new THREE.AmbientLight(0xffffff, 0.75))
    const dir = new THREE.DirectionalLight(0xffffff, 0.55)
    dir.position.set(5, 10, 5)
    scene.add(dir)

    const grid = new THREE.GridHelper(60, 120, 0x223344, 0x152128)
    ;(grid.material as THREE.Material).transparent = true
    ;(grid.material as THREE.Material).opacity = 0.55
    scene.add(grid)
    scene.add(new THREE.AxesHelper(1.2))

    const geo = new THREE.BoxGeometry(CELL * 0.95, CELL * 0.95, CELL * 0.95)
    const matReach = new THREE.MeshLambertMaterial({ color: 0x4ade80, transparent: true, opacity: 0.85 })
    const matWall = new THREE.MeshLambertMaterial({ color: 0xf87171, transparent: true, opacity: 0.55 })
    const matIffy = new THREE.MeshLambertMaterial({ color: 0xfacc15, transparent: true, opacity: 0.7 })
    const meshReach = new THREE.InstancedMesh(geo, matReach, CAP)
    const meshWall = new THREE.InstancedMesh(geo, matWall, CAP)
    const meshIffy = new THREE.InstancedMesh(geo, matIffy, CAP)
    meshReach.count = 0; meshWall.count = 0; meshIffy.count = 0
    // disable frustum culling, three.js computes the bounding sphere from the
    // unit BoxGeometry and not from instance world transforms, so when the
    // map drifts away from origin the whole mesh gets culled and the voxel
    // grid just vanishes at certain camera angles / zoom levels.
    meshReach.frustumCulled = false
    meshWall.frustumCulled = false
    meshIffy.frustumCulled = false
    scene.add(meshReach, meshWall, meshIffy)

    const playerGeo = new THREE.ConeGeometry(0.18, 0.5, 12)
    playerGeo.rotateX(Math.PI / 2)
    const player = new THREE.Mesh(playerGeo, new THREE.MeshBasicMaterial({ color: 0x38bdf8 }))
    player.visible = false
    scene.add(player)

    const target = new THREE.Mesh(
      new THREE.BoxGeometry(CELL, CELL, CELL),
      new THREE.MeshBasicMaterial({ color: 0xfb923c, wireframe: true }),
    )
    target.visible = false
    scene.add(target)

    // invisible ground plane for raycasting clicks into the world
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(200, 200),
      new THREE.MeshBasicMaterial({ visible: false }),
    )
    plane.rotation.x = -Math.PI / 2
    scene.add(plane)

    // path line geometry, updated dynamically
    const pathGeo = new THREE.BufferGeometry()
    pathGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3))
    const pathMat = new THREE.LineBasicMaterial({ color: 0xa78bfa, linewidth: 3 })
    const pathLine = new THREE.Line(pathGeo, pathMat)
    scene.add(pathLine)

    let raf = 0
    const animate = () => {
      raf = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    const onResize = () => {
      const w = mount.clientWidth
      const h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    const ro = new ResizeObserver(onResize)
    ro.observe(mount)

    sceneRefs.current = {
      renderer, scene, camera, controls,
      meshReach, meshWall, meshIffy, pathLine,
      player, target, plane,
      raycaster: new THREE.Raycaster(),
      firstPose: false,
    }

    const dispose = () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.dispose()
      renderer.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
    sceneRefs.current.dispose = dispose

    return () => dispose()
  }, [])

  // -----------------------------------------------------------------
  // poll state (light) every ~600ms
  // -----------------------------------------------------------------
  useEffect(() => {
    let alive = true
    const tick = async () => {
      try {
        const s = await api<MappingState>('/api/mapping/state')
        if (!alive) return
        setState(s)
        if (s.settings) {
          setTickHz(prev => prev === s.settings!.tick_hz ? prev : s.settings!.tick_hz)
          setForceRun(prev => prev === s.settings!.force_run ? prev : s.settings!.force_run)
          const wd = s.settings.manual_wall_distance
          if (typeof wd === 'number') {
            setWallDist(prev => Math.abs(prev - wd) < 0.005 ? prev : wd)
          }
        }
        const refs = sceneRefs.current
        if (refs.player && s.pose) {
          refs.player.visible = true
          refs.player.position.set(s.pose.x, s.pose.y + 0.5, s.pose.z)
          refs.player.rotation.y = -s.pose.yaw * Math.PI / 180
          if (!refs.firstPose && refs.camera && refs.controls) {
            refs.firstPose = true
            refs.controls.target.set(s.pose.x, s.pose.y, s.pose.z)
            refs.camera.position.set(s.pose.x + 5, s.pose.y + 6, s.pose.z + 5)
          }
        } else if (refs.player) {
          refs.player.visible = false
        }
        if (refs.target) {
          if (s.target) {
            refs.target.visible = true
            refs.target.position.set(
              s.target[0] * CELL + HALF,
              s.target[1] * CELL + HALF,
              s.target[2] * CELL + HALF,
            )
          } else {
            refs.target.visible = false
          }
        }
      } catch { /* server might be reloading */ }
    }
    tick()
    const id = setInterval(tick, 600)
    return () => { alive = false; clearInterval(id) }
  }, [])

  // heavy world cells poll, every ~1.2s only when running
  useEffect(() => {
    if (!state?.running) return
    let alive = true
    const tick = async () => {
      try {
        const w = await api<WorldCells>('/api/mapping/world')
        if (!alive) return
        const refs = sceneRefs.current
        refs.meshReach = packCells(refs.meshReach, w.reach, refs.scene)
        refs.meshWall = packCells(refs.meshWall, w.wall, refs.scene)
        refs.meshIffy = packCells(refs.meshIffy, w.iffy, refs.scene)
      } catch { /* ignore */ }
    }
    tick()
    const id = setInterval(tick, 1200)
    return () => { alive = false; clearInterval(id) }
  }, [state?.running])

  // redraw path overlay when path changes
  useEffect(() => {
    const line = sceneRefs.current.pathLine
    if (!line) return
    const cells = path?.found ? (path.full ?? []) : []
    const positions = new Float32Array(cells.length * 3)
    cells.forEach(([sx, sy, sz], i) => {
      positions[i * 3] = sx * CELL + HALF
      positions[i * 3 + 1] = sy * CELL + HALF + 0.05
      positions[i * 3 + 2] = sz * CELL + HALF
    })
    line.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    line.geometry.computeBoundingSphere()
    line.visible = cells.length > 0
  }, [path])

  // -----------------------------------------------------------------
  // click pathfinding
  // -----------------------------------------------------------------
  const onCanvasClick = useCallback(async (e: React.MouseEvent) => {
    if (suppressClickRef.current) {
      suppressClickRef.current = false
      return
    }
    if (editMode) {
      // shift+click belongs to the rectangle selector, never opens a menu
      if (e.shiftKey) return
      // only existing voxels are editable. clicks on empty space close any
      // open menu and otherwise do nothing -- no more accidental stray
      // cells from clicking the ground plane.
      const refs = sceneRefs.current
      if (!refs.renderer || !refs.camera || !refs.raycaster) return
      const rect = refs.renderer.domElement.getBoundingClientRect()
      const x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      const y = -((e.clientY - rect.top) / rect.height) * 2 + 1
      refs.raycaster.setFromCamera(new THREE.Vector2(x, y), refs.camera)
      const meshes = [refs.meshReach, refs.meshWall, refs.meshIffy].filter(Boolean) as THREE.InstancedMesh[]
      const hits = refs.raycaster.intersectObjects(meshes, false)
      const tmp = new THREE.Matrix4()
      const pos = new THREE.Vector3()
      for (const h of hits) {
        if (h.instanceId === undefined) continue
        const mesh = h.object as THREE.InstancedMesh
        mesh.getMatrixAt(h.instanceId, tmp)
        pos.setFromMatrixPosition(tmp)
        const sx = Math.floor(pos.x / CELL)
        const sy = Math.floor(pos.y / CELL)
        const sz = Math.floor(pos.z / CELL)
        setEditMenu({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
          serial: [sx, sy, sz],
        })
        setBulkMenu(null)
        return
      }
      // blank click in edit mode -- dismiss any open menus
      setEditMenu(null)
      setBulkMenu(null)
      return
    }
    if (!pickEnabled) return
    const refs = sceneRefs.current
    if (!refs.renderer || !refs.camera || !refs.plane || !refs.raycaster) return
    const rect = refs.renderer.domElement.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1
    refs.raycaster.setFromCamera(new THREE.Vector2(x, y), refs.camera)
    const hit = refs.raycaster.intersectObject(refs.plane)[0]
    if (!hit) return
    try {
      const res = await api<PathResult>('/api/mapping/pathfind', 'POST', {
        x: hit.point.x, y: hit.point.y, z: hit.point.z,
      })
      setPath(res)
      if (!res.found) onToast(`pathfind: ${res.reason ?? 'no path'}`, 'warn')
      else onToast(`path found: ${res.full?.length ?? 0} cells`, 'success')
    } catch (err) {
      onToast(`pathfind failed: ${(err as Error).message}`, 'error')
    }
  }, [pickEnabled, editMode, onToast])

  const refreshWorld = useCallback(async () => {
    try {
      const w = await api<WorldCells>('/api/mapping/world')
      const refs = sceneRefs.current
      refs.meshReach = packCells(refs.meshReach, w.reach, refs.scene)
      refs.meshWall = packCells(refs.meshWall, w.wall, refs.scene)
      refs.meshIffy = packCells(refs.meshIffy, w.iffy, refs.scene)
    } catch { /* ignore */ }
  }, [])

  const applyCellEdit = useCallback(async (kind: 'reach' | 'wall' | 'iffy' | 'delete') => {
    if (!editMenu) return
    const [sx, sy, sz] = editMenu.serial
    try {
      await api('/api/mapping/cell', 'POST', { sx, sy, sz, kind })
      onToast(`cell ${sx},${sy},${sz} -> ${kind}`, 'success')
      setEditMenu(null)
      refreshWorld()
    } catch (err) {
      onToast(`edit failed: ${(err as Error).message}`, 'error')
    }
  }, [editMenu, onToast, refreshWorld])

  const applyBulkEdit = useCallback(async (kind: 'reach' | 'wall' | 'iffy' | 'delete') => {
    if (!bulkMenu || bulkMenu.cells.length === 0) return
    try {
      const r = await api<{ applied: number; total: number }>(
        '/api/mapping/cells/bulk', 'POST', { cells: bulkMenu.cells, kind })
      onToast(`bulk ${kind}: ${r.applied}/${r.total} cells`, 'success')
      setBulkMenu(null)
      refreshWorld()
    } catch (err) {
      onToast(`bulk edit failed: ${(err as Error).message}`, 'error')
    }
  }, [bulkMenu, onToast, refreshWorld])

  // collect every visible cell (across all 3 instanced meshes) whose
  // projected screen position lands inside the given rect. used by the
  // shift+drag selector. returns serials in voxel coords.
  const collectCellsInRect = useCallback((rect: DOMRect, x0: number, y0: number, x1: number, y1: number): [number, number, number][] => {
    const refs = sceneRefs.current
    if (!refs.camera || !refs.renderer) return []
    const out: [number, number, number][] = []
    const tmp = new THREE.Matrix4()
    const pos = new THREE.Vector3()
    const ndc = new THREE.Vector3()
    const minX = Math.min(x0, x1)
    const maxX = Math.max(x0, x1)
    const minY = Math.min(y0, y1)
    const maxY = Math.max(y0, y1)
    const meshes = [refs.meshReach, refs.meshWall, refs.meshIffy].filter(Boolean) as THREE.InstancedMesh[]
    const seen = new Set<string>()
    for (const mesh of meshes) {
      const n = mesh.count
      for (let i = 0; i < n; i++) {
        mesh.getMatrixAt(i, tmp)
        pos.setFromMatrixPosition(tmp)
        ndc.copy(pos).project(refs.camera)
        // behind the camera
        if (ndc.z < -1 || ndc.z > 1) continue
        const sx_px = (ndc.x * 0.5 + 0.5) * rect.width
        const sy_px = (-ndc.y * 0.5 + 0.5) * rect.height
        if (sx_px < minX || sx_px > maxX || sy_px < minY || sy_px > maxY) continue
        const cx = Math.floor(pos.x / CELL)
        const cy = Math.floor(pos.y / CELL)
        const cz = Math.floor(pos.z / CELL)
        const key = `${cx},${cy},${cz}`
        if (seen.has(key)) continue
        seen.add(key)
        out.push([cx, cy, cz])
      }
    }
    return out
  }, [])

  const onCanvasMouseDown = useCallback((e: React.MouseEvent) => {
    if (!editMode) return
    if (!e.shiftKey) return  // only shift+drag starts a rectangle select
    const refs = sceneRefs.current
    if (!refs.renderer || !refs.controls) return
    e.preventDefault()
    const rect = refs.renderer.domElement.getBoundingClientRect()
    dragStateRef.current = {
      active: true,
      started: false,
      x0: e.clientX - rect.left,
      y0: e.clientY - rect.top,
    }
    refs.controls.enabled = false
    setEditMenu(null)
    setBulkMenu(null)
  }, [editMode])

  const onCanvasMouseMove = useCallback((e: React.MouseEvent) => {
    const drag = dragStateRef.current
    if (!drag || !drag.active) return
    const refs = sceneRefs.current
    if (!refs.renderer) return
    const rect = refs.renderer.domElement.getBoundingClientRect()
    const x1 = e.clientX - rect.left
    const y1 = e.clientY - rect.top
    if (!drag.started) {
      // require >4px before we commit to drag mode (lets accidental shift-clicks
      // still feel like clicks)
      if (Math.abs(x1 - drag.x0) < 4 && Math.abs(y1 - drag.y0) < 4) return
      drag.started = true
    }
    setDragRect({ x0: drag.x0, y0: drag.y0, x1, y1 })
  }, [])

  const onCanvasMouseUp = useCallback((e: React.MouseEvent) => {
    const drag = dragStateRef.current
    if (!drag || !drag.active) return
    const refs = sceneRefs.current
    if (refs.controls) refs.controls.enabled = true
    dragStateRef.current = null
    if (!drag.started) {
      // shift+click without real drag, treat like nothing happened
      setDragRect(null)
      return
    }
    suppressClickRef.current = true
    if (!refs.renderer) { setDragRect(null); return }
    const rect = refs.renderer.domElement.getBoundingClientRect()
    const x1 = e.clientX - rect.left
    const y1 = e.clientY - rect.top
    const cells = collectCellsInRect(rect, drag.x0, drag.y0, x1, y1)
    setDragRect(null)
    if (cells.length === 0) {
      onToast('selection empty', 'info')
      return
    }
    setEditMenu(null)
    setBulkMenu({ x: x1, y: y1, cells })
  }, [collectCellsInRect, onToast])

  // -----------------------------------------------------------------
  // controls
  // -----------------------------------------------------------------
  const start = async (explore: boolean) => {
    setBusy(true)
    try {
      const s = await api<MappingState>('/api/mapping/start', 'POST', { explore })
      setState(s)
      onToast(explore ? 'mapping + explorer started' : 'mapping started', 'success')
    } catch (err) {
      onToast((err as Error).message, 'error')
    } finally { setBusy(false) }
  }
  const stop = async () => {
    setBusy(true)
    try {
      const s = await api<MappingState>('/api/mapping/stop', 'POST')
      setState(s)
      setPath(null)
      onToast('mapping stopped', 'info')
    } catch (err) {
      onToast((err as Error).message, 'error')
    } finally { setBusy(false) }
  }
  const toggleExplore = async () => {
    if (!state) return
    setBusy(true)
    try {
      const s = await api<MappingState>('/api/mapping/explore', 'POST', { enabled: !state.explore })
      setState(s)
    } catch (err) {
      onToast((err as Error).message, 'error')
    } finally { setBusy(false) }
  }
  const toggleManual = async () => {
    if (!state) return
    setBusy(true)
    try {
      const s = await api<MappingState>('/api/mapping/manual', 'POST', { enabled: !state.manual })
      setState(s)
      onToast(s.manual ? 'manual mapping ON' : 'manual mapping off', 'info')
    } catch (err) {
      onToast((err as Error).message, 'error')
    } finally { setBusy(false) }
  }

  const loadSavedWorlds = useCallback(async () => {
    try {
      const r = await api<{ worlds: SavedWorld[] }>('/api/mapping/worlds')
      setSavedWorlds(r.worlds)
    } catch (err) {
      onToast(`load failed: ${(err as Error).message}`, 'error')
    }
  }, [onToast])

  const openWorlds = () => {
    setShowWorlds(true)
    loadSavedWorlds()
  }

  const applySettings = async (next: { tick_hz?: number; force_run?: boolean; manual_wall_distance?: number; manual_wall_ratio?: number }) => {
    try {
      await api('/api/mapping/settings', 'POST', next)
    } catch (err) {
      onToast(`settings: ${(err as Error).message}`, 'error')
    }
  }

  const cancelGoto = async () => {
    try {
      await api('/api/mapping/cancel_goto', 'POST', {})
      onToast('walk cancelled', 'success')
    } catch (err) {
      onToast((err as Error).message, 'error')
    }
  }

  const cleanupStrays = async () => {
    // dry run first so we can tell the user what would happen
    let preview: any
    try {
      preview = await api('/api/mapping/cleanup_strays', 'POST', { min_component_size: 8, dry_run: true })
    } catch (err) {
      onToast(`cleanup preview: ${(err as Error).message}`, 'error')
      return
    }
    const wouldRemove = preview?.cells_removed ?? 0
    const comps = preview?.components_removed ?? 0
    if (wouldRemove <= 0) {
      onToast('no stray voxels found', 'success')
      return
    }
    const msg = `Found ${wouldRemove} stray voxels across ${comps} floating islands.\n\n` +
                `Keeping the main map (${preview?.largest_component ?? 0} cells) + anything near a waypoint or your current cell.\n\n` +
                `Delete them?`
    if (!confirm(msg)) return
    setBusy(true)
    try {
      const r: any = await api('/api/mapping/cleanup_strays', 'POST', { min_component_size: 8, dry_run: false })
      onToast(`removed ${r?.cells_removed ?? 0} stray voxels`, 'success')
      // refresh the world cells so the viewer drops them
      refreshWorld()
    } catch (err) {
      onToast((err as Error).message, 'error')
    } finally { setBusy(false) }
  }

  const deleteWorld = async (worldId: string, isCurrent: boolean) => {    const label = isCurrent ? 'CURRENT' : worldId
    if (!confirm(`Delete saved map for "${label}"?\n\nThis cant be undone.`)) return
    setBusy(true)
    try {
      await api(`/api/mapping/world?world=${encodeURIComponent(worldId)}`, 'DELETE')
      onToast(`deleted map for ${worldId}`, 'success')
      if (isCurrent) {
        // wipe local instanced meshes immediately so the viewer empties
        const refs = sceneRefs.current
        if (refs.meshReach) refs.meshReach.count = 0
        if (refs.meshWall) refs.meshWall.count = 0
        if (refs.meshIffy) refs.meshIffy.count = 0
        setPath(null)
      }
      loadSavedWorlds()
    } catch (err) {
      onToast((err as Error).message, 'error')
    } finally { setBusy(false) }
  }

  // -----------------------------------------------------------------
  // render
  // -----------------------------------------------------------------
  return (
    <div className="relative w-full" style={{ height: 'calc(100vh - 48px)' }}>
      <div
        ref={mountRef}
        onClick={onCanvasClick}
        onMouseDown={onCanvasMouseDown}
        onMouseMove={onCanvasMouseMove}
        onMouseUp={onCanvasMouseUp}
        className="absolute inset-0"
        style={{ cursor: (pickEnabled || editMode) ? 'crosshair' : 'grab' }}
      />

      {/* drag selection rectangle overlay */}
      {editMode && dragRect && (
        <div
          className="absolute z-20 pointer-events-none border border-accent/80 bg-accent/15"
          style={{
            left: Math.min(dragRect.x0, dragRect.x1),
            top: Math.min(dragRect.y0, dragRect.y1),
            width: Math.abs(dragRect.x1 - dragRect.x0),
            height: Math.abs(dragRect.y1 - dragRect.y0),
          }}
        />
      )}

      {/* bulk action menu (after drag-select) */}
      {editMode && bulkMenu && (
        <div
          className="absolute z-30 bg-surface/95 backdrop-blur-xl border border-white/10 rounded-lg p-2 shadow-xl flex flex-col gap-1 text-[12px] font-mono"
          style={{
            left: Math.min(bulkMenu.x + 6, (mountRef.current?.clientWidth ?? 9999) - 200),
            top: Math.min(bulkMenu.y + 6, (mountRef.current?.clientHeight ?? 9999) - 200),
            minWidth: 190,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="text-text-muted/80 px-1 pb-1 border-b border-white/5">
            <span className="text-text">{bulkMenu.cells.length}</span> cells selected
          </div>
          <button
            onClick={() => applyBulkEdit('reach')}
            className="text-left px-2 py-1 rounded hover:bg-mint/15 text-mint flex items-center gap-2"
          >
            <span className="w-3 h-3 rounded-sm" style={{ background: '#4ade80' }} />
            Set Reachable
          </button>
          <button
            onClick={() => applyBulkEdit('wall')}
            className="text-left px-2 py-1 rounded hover:bg-rose/15 text-rose flex items-center gap-2"
          >
            <span className="w-3 h-3 rounded-sm" style={{ background: '#f87171' }} />
            Set Wall
          </button>
          <button
            onClick={() => applyBulkEdit('iffy')}
            className="text-left px-2 py-1 rounded hover:bg-yellow-500/15 text-yellow-300 flex items-center gap-2"
          >
            <span className="w-3 h-3 rounded-sm" style={{ background: '#facc15' }} />
            Set Iffy
          </button>
          <button
            onClick={() => applyBulkEdit('delete')}
            className="text-left px-2 py-1 rounded hover:bg-white/10 text-text-muted flex items-center gap-2 border-t border-white/5 mt-1 pt-1.5"
          >
            <TbTrash size={12} /> Delete All
          </button>
          <button
            onClick={() => setBulkMenu(null)}
            className="text-left px-2 py-1 rounded text-text-muted/50 hover:text-text text-[11px]"
          >
            cancel
          </button>
        </div>
      )}

      {/* edit cell context menu */}
      {editMode && editMenu && (
        <div
          className="absolute z-30 bg-surface/95 backdrop-blur-xl border border-white/10 rounded-lg p-2 shadow-xl flex flex-col gap-1 text-[12px] font-mono"
          style={{
            left: Math.min(editMenu.x + 6, (mountRef.current?.clientWidth ?? 9999) - 170),
            top: Math.min(editMenu.y + 6, (mountRef.current?.clientHeight ?? 9999) - 180),
            minWidth: 160,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="text-text-muted/80 px-1 pb-1 border-b border-white/5">
            cell <span className="text-text">{editMenu.serial.join(', ')}</span>
          </div>
          <button
            onClick={() => applyCellEdit('reach')}
            className="text-left px-2 py-1 rounded hover:bg-mint/15 text-mint flex items-center gap-2"
          >
            <span className="w-3 h-3 rounded-sm" style={{ background: '#4ade80' }} />
            Reachable
          </button>
          <button
            onClick={() => applyCellEdit('wall')}
            className="text-left px-2 py-1 rounded hover:bg-rose/15 text-rose flex items-center gap-2"
          >
            <span className="w-3 h-3 rounded-sm" style={{ background: '#f87171' }} />
            Wall
          </button>
          <button
            onClick={() => applyCellEdit('iffy')}
            className="text-left px-2 py-1 rounded hover:bg-yellow-500/15 text-yellow-300 flex items-center gap-2"
          >
            <span className="w-3 h-3 rounded-sm" style={{ background: '#facc15' }} />
            Iffy
          </button>
          <button
            onClick={() => applyCellEdit('delete')}
            className="text-left px-2 py-1 rounded hover:bg-white/10 text-text-muted flex items-center gap-2 border-t border-white/5 mt-1 pt-1.5"
          >
            <TbTrash size={12} /> Delete
          </button>
          <button
            onClick={() => setEditMenu(null)}
            className="text-left px-2 py-1 rounded text-text-muted/50 hover:text-text text-[11px]"
          >
            cancel
          </button>
        </div>
      )}

      {/* top-left HUD */}
      <div className="absolute top-3 left-3 bg-surface/80 backdrop-blur-xl border border-white/10 rounded-lg p-3 text-[12px] font-mono leading-relaxed min-w-[220px]">
        <div className="flex items-center gap-2 mb-1.5">
          <span className={`w-2 h-2 rounded-full ${state?.running ? 'bg-mint animate-pulse' : 'bg-rose'}`} />
          <span className="font-title font-semibold text-text">
            {state?.running ? 'Mapping' : 'Idle'}
          </span>
          {state?.explore && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent/15 text-accent font-title uppercase tracking-wide">
              explorer
            </span>
          )}
          {state?.manual && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-mint/15 text-mint font-title uppercase tracking-wide">
              manual
            </span>
          )}
        </div>
        <div className="text-text-muted/80">
          <div>
            <span className="text-text-muted/50">world</span>{' '}
            <span className="text-text">{state?.world_name || state?.world || '-'}</span>
          </div>
          {state?.world_name && state?.world && state.world !== state.world_name && (
            <div className="text-text-muted/40 text-[10px] font-mono">{state.world}</div>
          )}
          <div>
            <span className="text-mint">{state?.counts.reach ?? 0}</span>
            <span className="text-text-muted/40 mx-1">/</span>
            <span className="text-rose">{state?.counts.wall ?? 0}</span>
            <span className="text-text-muted/40 mx-1">/</span>
            <span className="text-yellow-400">{state?.counts.iffy ?? 0}</span>
            <span className="text-text-muted/50 ml-1">cells</span>
          </div>
          {state?.pose ? (
            <div className="text-[11px]">
              {state.pose.x.toFixed(2)}, {state.pose.y.toFixed(2)}, {state.pose.z.toFixed(2)}
              <span className="text-text-muted/50"> yaw </span>
              {state.pose.yaw.toFixed(0)}&deg;
            </div>
          ) : (
            <div className="text-text-muted/50 text-[11px]">waiting for pose...</div>
          )}
          <div><span className="text-text-muted/50">action</span> <span className="text-text">{state?.action ?? '-'}</span></div>
          <div><span className="text-text-muted/50">decode</span> <span className="text-text">{((state?.decode_rate ?? 0) * 100).toFixed(0)}%</span></div>
        </div>
        {state?.last_error && (
          <div className="mt-2 flex items-start gap-1.5 text-rose text-[11px]">
            <TbAlertTriangle size={12} className="mt-0.5 shrink-0" />
            <span>{state.last_error}</span>
          </div>
        )}
      </div>

      {/* top-right legend */}
      <div className="absolute top-3 right-3 bg-surface/80 backdrop-blur-xl border border-white/10 rounded-lg p-3 text-[11px] font-mono space-y-1">
        <Legend color="#4ade80" label="reachable" />
        <Legend color="#f87171" label="wall" />
        <Legend color="#facc15" label="iffy" />
        <Legend color="#38bdf8" label="current" />
        <Legend color="#fb923c" label="target" />
        <Legend color="#a78bfa" label="path" />
      </div>

      {/* bottom control bar */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-surface/85 backdrop-blur-xl border border-white/10 rounded-xl p-2 shadow-lg">
        {!state?.running ? (
          <>
            <button
              disabled={busy}
              onClick={() => start(false)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium bg-mint/10 text-mint hover:bg-mint/20 disabled:opacity-40 transition"
            >
              <TbPlayerPlay size={14} /> Start Mapping
            </button>
            <button
              disabled={busy}
              onClick={() => start(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium bg-accent/10 text-accent hover:bg-accent/20 disabled:opacity-40 transition"
            >
              <TbRobot size={14} /> Start + Explore
            </button>
          </>
        ) : (
          <>
            <button
              disabled={busy}
              onClick={stop}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium bg-rose/10 text-rose hover:bg-rose/20 disabled:opacity-40 transition"
            >
              <TbPlayerStop size={14} /> Stop
            </button>
            <button
              disabled={busy}
              onClick={toggleExplore}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium transition ${
                state.explore
                  ? 'bg-accent/20 text-accent hover:bg-accent/30'
                  : 'bg-white/5 text-text-muted hover:bg-white/10'
              }`}
            >
              <TbRobot size={14} /> Explorer
            </button>
            <button
              disabled={busy}
              onClick={toggleManual}
              title="You drive, we listen to the forward raycast and tag walls in front of you."
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium transition ${
                state.manual
                  ? 'bg-mint/20 text-mint hover:bg-mint/30'
                  : 'bg-white/5 text-text-muted hover:bg-white/10'
              }`}
            >
              <TbPlayerPlay size={14} /> Manual
            </button>
          </>
        )}
        <div className="w-px h-5 bg-white/10 mx-1" />
        <button
          onClick={() => {
            const next = !pickEnabled
            setPickEnabled(next)
            if (!next) setPath(null)
            if (next) {
              setEditMode(false)
              setEditMenu(null)
              setBulkMenu(null)
            }
          }}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium transition ${
            pickEnabled
              ? 'bg-violet-500/20 text-violet-300 hover:bg-violet-500/30'
              : 'bg-white/5 text-text-muted hover:bg-white/10'
          }`}
        >
          <TbCrosshair size={14} /> {pickEnabled ? 'Click to pathfind' : 'Pathfind'}
        </button>
        <button
          onClick={() => { setEditMode(m => !m); setEditMenu(null); setBulkMenu(null); if (!editMode) setPickEnabled(false) }}
          title="Click an existing cell to change its type or delete it. Shift+drag to bulk-select cells. Clicks on empty space are ignored."
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium transition ${
            editMode
              ? 'bg-orange-500/20 text-orange-300 hover:bg-orange-500/30'
              : 'bg-white/5 text-text-muted hover:bg-white/10'
          }`}
        >
          <TbEdit size={14} /> {editMode ? 'Editing cells' : 'Edit Cells'}
        </button>
        {editMode && (
          <span className="text-[11px] text-text-muted/70 ml-1">
            click=single, shift+drag=bulk
          </span>
        )}
        {path?.found && (
          <span className="text-[11px] text-text-muted ml-1">
            {path.full?.length ?? 0} cells, {path.filtered?.length ?? 0} turns
          </span>
        )}
        <div className="w-px h-5 bg-white/10 mx-1" />
        <button
          onClick={() => setShowSettings(v => !v)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium transition ${showSettings ? 'bg-accent/20 text-accent' : 'bg-white/5 text-text-muted hover:bg-white/10'}`}
          title="Mapping settings"
        >
          <TbSettings size={14} /> Settings
        </button>
        <button
          onClick={openWorlds}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium bg-white/5 text-text-muted hover:bg-white/10 transition"
          title="Saved maps"
        >
          <TbDatabase size={14} /> Saved Maps
        </button>
        <button
          onClick={cleanupStrays}
          disabled={busy}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium bg-white/5 text-text-muted hover:bg-white/10 transition disabled:opacity-50"
          title="Delete tiny floating voxel islands (keeps main map + waypoints + your cell)"
        >
          <TbSparkles size={14} /> Clean Strays
        </button>
        {state?.follow?.active && (
          <button
            onClick={cancelGoto}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium bg-rose/15 text-rose hover:bg-rose/25 transition"
            title="Cancel walk"
          >
            <TbPlayerStop size={14} /> Cancel Walk
            <span className="text-[11px] opacity-70">({state.follow.remaining})</span>
          </button>
        )}
      </div>

      {/* settings panel (expandable) */}
      {showSettings && (
        <div className="absolute bottom-20 left-1/2 -translate-x-1/2 bg-surface/95 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl px-5 py-4 w-[420px] z-10">
          <div className="flex items-center justify-between mb-3">
            <div className="font-title font-semibold text-text text-[13px] flex items-center gap-2">
              <TbSettings size={14} className="text-accent" /> Mapping Settings
            </div>
            <button onClick={() => setShowSettings(false)} className="text-text-muted hover:text-text text-[16px] leading-none">&times;</button>
          </div>
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[12px] font-medium text-text-muted">Sample Rate</label>
                <span className="text-[12px] font-mono text-accent">{tickHz.toFixed(0)} Hz</span>
              </div>
              <input
                type="range" min={5} max={60} step={1}
                value={tickHz}
                onChange={e => setTickHz(parseInt(e.target.value))}
                onMouseUp={() => applySettings({ tick_hz: tickHz })}
                onTouchEnd={() => applySettings({ tick_hz: tickHz })}
                className="w-full accent-accent"
              />
              <div className="flex justify-between text-[10px] text-text-muted/50 mt-0.5">
                <span>5 (slow)</span><span>20 (default)</span><span>60 (fast)</span>
              </div>
              <div className="text-[11px] text-text-muted/70 mt-1">
                How often the pose strip is sampled and fed into the voxel map. Higher = denser/faster mapping but more CPU.
              </div>
            </div>
            <label className="flex items-center gap-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={forceRun}
                onChange={e => { setForceRun(e.target.checked); applySettings({ force_run: e.target.checked }) }}
                className="accent-accent w-4 h-4"
              />
              <TbRun size={14} className="text-text-muted" />
              <span className="text-[12px] text-text">Always sprint while exploring</span>
            </label>
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[12px] font-medium text-text-muted">Manual Wall Distance</label>
                <span className="text-[12px] font-mono text-mint">{wallDist.toFixed(2)} m</span>
              </div>
              <input
                type="range" min={0.05} max={2.0} step={0.01}
                value={wallDist}
                onChange={e => setWallDist(parseFloat(e.target.value))}
                onMouseUp={() => applySettings({ manual_wall_distance: wallDist })}
                onTouchEnd={() => applySettings({ manual_wall_distance: wallDist })}
                className="w-full accent-mint"
              />
              <div className="flex justify-between text-[10px] text-text-muted/50 mt-0.5">
                <span>0.05 (tight)</span><span>0.35 (default)</span><span>2.0 (loose)</span>
              </div>
              <div className="text-[11px] text-text-muted/70 mt-1">
                In manual mapping, the forward raycast counts a hit as a wall when distance is at or below this. Bump it up if walls arent being tagged, lower it to avoid tagging things youre walking near.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* saved maps modal */}
      {showWorlds && (
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-20" onClick={() => setShowWorlds(false)}>
          <div className="bg-surface border border-white/10 rounded-xl shadow-2xl w-[520px] max-h-[70vh] overflow-hidden flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
              <h3 className="font-title font-semibold text-text text-[14px] flex items-center gap-2">
                <TbDatabase size={16} className="text-accent" /> Saved Maps
              </h3>
              <button onClick={() => setShowWorlds(false)} className="text-text-muted hover:text-text text-[18px] leading-none">&times;</button>
            </div>
            <div className="overflow-y-auto flex-1">
              {savedWorlds.length === 0 ? (
                <div className="p-8 text-center text-text-muted text-[13px]">no saved maps</div>
              ) : (
                <table className="w-full text-[13px]">
                  <thead className="bg-white/[0.03] text-[11px] text-text-muted/70 uppercase tracking-wider">
                    <tr>
                      <th className="text-left px-4 py-2 font-medium">World ID</th>
                      <th className="text-right px-4 py-2 font-medium">Size</th>
                      <th className="w-10" />
                    </tr>
                  </thead>
                  <tbody>
                    {savedWorlds.map(w => (
                      <tr key={w.world} className="border-t border-white/5 hover:bg-white/[0.02]">
                        <td className="px-4 py-2 font-mono text-[12px]">
                          <span className={w.is_current ? 'text-mint' : 'text-text'}>{w.world}</span>
                          {w.is_current && (
                            <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-mint/15 text-mint font-title uppercase tracking-wide">
                              current
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-2 text-right text-text-muted font-mono text-[12px]">{w.size_kb.toFixed(1)} KB</td>
                        <td className="px-4 py-2">
                          <button
                            onClick={() => deleteWorld(w.world, w.is_current)}
                            disabled={busy}
                            className="p-1.5 rounded-md text-text-muted/50 hover:text-rose hover:bg-rose/10 transition disabled:opacity-30"
                            title="delete map"
                          >
                            <TbTrash size={14} />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: color }} />
      <span className="text-text-muted">{label}</span>
    </div>
  )
}

// pack a cell list into an InstancedMesh, growing the buffer if needed.
// returns the mesh to use (may be a brand new one if the old buffer was
// too small) so the caller can update its scene ref.
const _dummy = new THREE.Object3D()
function packCells(
  mesh: THREE.InstancedMesh | undefined,
  cells: [number, number, number][],
  scene?: THREE.Scene,
): THREE.InstancedMesh | undefined {
  if (!mesh) return mesh
  let target = mesh
  const capacity = mesh.instanceMatrix.count
  if (cells.length > capacity) {
    // grow with headroom so we dont reallocate on every single new cell.
    // 1.5x + 1024 padding keeps it stable as the map fills in.
    const newCap = Math.max(cells.length + 1024, Math.ceil(capacity * 1.5))
    const grown = new THREE.InstancedMesh(mesh.geometry, mesh.material, newCap)
    grown.frustumCulled = mesh.frustumCulled
    grown.count = 0
    if (scene) {
      scene.add(grown)
      scene.remove(mesh)
    }
    mesh.dispose()
    target = grown
  }
  const n = cells.length
  for (let i = 0; i < n; i++) {
    const [sx, sy, sz] = cells[i]
    _dummy.position.set(sx * CELL + HALF, sy * CELL + HALF, sz * CELL + HALF)
    _dummy.updateMatrix()
    target.setMatrixAt(i, _dummy.matrix)
  }
  target.count = n
  target.instanceMatrix.needsUpdate = true
  return target
}

"""reference-style live trail mapping in VRChat.

Combines:
    * PoseExfilReader  -- world pose (x,y,z,yaw) decoded from the screen-space
      bit shader strip in the screen corner.
    * VoxelNavManager  -- 0.25m voxel graph, Reachable / UnReachable / Iffy.
    * VoxelExplorer    -- reference Wander.WalkToTarget + discovery target.
      Picks the closest unexplored cardinal neighbor of any Reachable
      node, walks toward it via OSC, marks UnReachable on stuck.
    * Tiny FastAPI + Three.js viewer at http://localhost:8769 that polls
      /state every 300ms and renders the graph + player pose in 3D.

Run for as long as you want, then Ctrl+C. The graph saves automatically
every 5 seconds and on exit.

Usage:
    .venv\\Scripts\\python.exe scripts\\test_voxel_mapping_live.py
    .venv\\Scripts\\python.exe scripts\\test_voxel_mapping_live.py --world myroom --duration 120
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import setup_logging                              # noqa: E402
from src.config import Config                                  # noqa: E402
from src.pose_decoder import GRID_W, GRID_H, PoseExfilReader   # noqa: E402
from src.voxel_explorer import VoxelExplorer                   # noqa: E402
from src.voxel_nav import NodeType, VoxelNavManager            # noqa: E402
from src.vrchat import VRChatOSC                               # noqa: E402

from scripts.test_pose_decoder_live import scan_and_decode     # noqa: E402

logger = logging.getLogger("voxel_mapping_live")


VIEWER_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>voxel map</title>
<style>
  body{margin:0;background:#0b0f14;color:#cfd8dc;font-family:monospace;overflow:hidden}
  #hud{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.55);padding:8px 12px;border-radius:6px;font-size:12px;line-height:1.5;z-index:10}
  #hud b{color:#80deea}
  #legend{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.55);padding:8px 12px;border-radius:6px;font-size:12px;z-index:10}
  .sw{display:inline-block;width:10px;height:10px;margin-right:6px;vertical-align:middle}
</style></head><body>
<div id="hud">connecting...</div>
<div id="legend">
  <div><span class="sw" style="background:#4caf50"></span>reachable</div>
  <div><span class="sw" style="background:#e53935"></span>wall</div>
  <div><span class="sw" style="background:#fdd835"></span>iffy</div>
  <div><span class="sw" style="background:#29b6f6"></span>current</div>
  <div><span class="sw" style="background:#ff9800"></span>target</div>
</div>
<script type="importmap">
{"imports":{
  "three":"https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const CELL = 0.25, HALF = 0.125;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f14);
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 500);
camera.position.set(6,8,6);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0,1,0);

scene.add(new THREE.AmbientLight(0xffffff,0.7));
const dir = new THREE.DirectionalLight(0xffffff,0.6); dir.position.set(5,10,5); scene.add(dir);
const grid = new THREE.GridHelper(40, 80, 0x223344, 0x152128); scene.add(grid);
const axes = new THREE.AxesHelper(1); scene.add(axes);

const geo = new THREE.BoxGeometry(CELL*0.95, CELL*0.95, CELL*0.95);
const matReach = new THREE.MeshLambertMaterial({color:0x4caf50, transparent:true, opacity:0.85});
const matWall  = new THREE.MeshLambertMaterial({color:0xe53935, transparent:true, opacity:0.55});
const matIffy  = new THREE.MeshLambertMaterial({color:0xfdd835, transparent:true, opacity:0.7});
const CAP = 12000;
const meshReach = new THREE.InstancedMesh(geo, matReach, CAP);
const meshWall  = new THREE.InstancedMesh(geo, matWall, CAP);
const meshIffy  = new THREE.InstancedMesh(geo, matIffy, CAP);
meshReach.count = 0; meshWall.count = 0; meshIffy.count = 0;
scene.add(meshReach); scene.add(meshWall); scene.add(meshIffy);

const playerGeo = new THREE.ConeGeometry(0.18, 0.5, 12);
playerGeo.rotateX(Math.PI/2);
const player = new THREE.Mesh(playerGeo, new THREE.MeshBasicMaterial({color:0x29b6f6}));
scene.add(player);

const targetGeo = new THREE.BoxGeometry(CELL, CELL, CELL);
const target = new THREE.Mesh(targetGeo, new THREE.MeshBasicMaterial({
  color:0xff9800, wireframe:true,
}));
target.visible = false;
scene.add(target);

const dummy = new THREE.Object3D();
function pack(mesh, cells){
  const n = Math.min(cells.length, CAP);
  for (let i=0;i<n;i++){
    const [sx,sy,sz] = cells[i];
    dummy.position.set(sx*CELL+HALF, sy*CELL+HALF, sz*CELL+HALF);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
}

const hud = document.getElementById('hud');
let firstPose = null;
async function poll(){
  try {
    const r = await fetch('/state'); const s = await r.json();
    pack(meshReach, s.reach);
    pack(meshWall, s.wall);
    pack(meshIffy, s.iffy);
    if (s.pose){
      player.position.set(s.pose.x, s.pose.y+0.5, s.pose.z);
      player.rotation.y = -s.pose.yaw * Math.PI/180;
      if (!firstPose){
        firstPose = s.pose;
        controls.target.set(s.pose.x, s.pose.y, s.pose.z);
        camera.position.set(s.pose.x+5, s.pose.y+6, s.pose.z+5);
      }
    }
    if (s.target){
      target.visible = true;
      target.position.set(s.target[0]*CELL+HALF, s.target[1]*CELL+HALF, s.target[2]*CELL+HALF);
    } else {
      target.visible = false;
    }
    hud.innerHTML =
      `<b>world</b> ${s.world}<br>` +
      `<b>reach</b> ${s.reach.length}   <b>wall</b> ${s.wall.length}   <b>iffy</b> ${s.iffy.length}<br>` +
      (s.pose ? `<b>pose</b> ${s.pose.x.toFixed(2)}, ${s.pose.y.toFixed(2)}, ${s.pose.z.toFixed(2)}  yaw ${s.pose.yaw.toFixed(0)}\u00b0<br>` : '<b>pose</b> waiting...<br>') +
      `<b>act</b> ${s.action || '-'}`;
  } catch(e){ hud.textContent = 'disconnected'; }
}
setInterval(poll, 300); poll();

function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene,camera); }
animate();
</script></body></html>
"""


def start_viewer(nav: VoxelNavManager, state: dict, port: int) -> None:
    """Spin up a tiny FastAPI server on a daemon thread serving a Three.js
    page that polls /state for the current voxel map + player pose."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def _index():
        return VIEWER_HTML

    @app.get("/state")
    def _state():
        reach, wall, iffy = [], [], []
        with nav.graph._lock:  # noqa: SLF001
            for serial, node in nav.graph.nodes.items():
                if node.node_type == NodeType.REACHABLE:
                    reach.append(serial)
                elif node.node_type == NodeType.UNREACHABLE:
                    wall.append(serial)
                else:
                    iffy.append(serial)
        return JSONResponse({
            "world": state.get("world"),
            "pose": state.get("pose"),
            "target": state.get("target"),
            "action": state.get("action"),
            "reach": reach,
            "wall": wall,
            "iffy": iffy,
        })

    def _run():
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port,
                             log_level="warning", access_log=False)
        uvicorn.Server(cfg).run()

    threading.Thread(target=_run, daemon=True, name="voxel-viewer").start()


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="livemap",
                    help="world id label (default 'livemap')")
    ap.add_argument("--data-dir", default="data/voxel_nav")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="auto-stop after this many seconds (0 = run forever)")
    ap.add_argument("--no-explore", action="store_true",
                    help="dont autonomously explore (you drive manually)")
    ap.add_argument("--viewer", action="store_true", default=True,
                    help="start live 3D viewer at http://localhost:8769")
    ap.add_argument("--no-viewer", dest="viewer", action="store_false")
    ap.add_argument("--viewer-port", type=int, default=8769)
    args = ap.parse_args()

    print("loading config + connecting OSC...")
    config = Config()
    osc = VRChatOSC(config)

    print("looking for pose strip on screen...")
    result = scan_and_decode(8)
    if not isinstance(result, tuple) or result[0] != 0:
        print("ERROR: could not find pose strip. is VRChat focused with the shader on?")
        return 1
    _, mon_index, abs_x, abs_y, est_cell = result
    print(f"  strip at monitor {mon_index} ({abs_x},{abs_y}) cell={est_cell}")

    region = {
        "left":   abs_x,
        "top":    abs_y,
        "width":  GRID_W * est_cell,
        "height": GRID_H * est_cell,
    }
    reader = PoseExfilReader(region=region, cell_size=est_cell,
                             poll_hz=20.0, monitor_index=mon_index)
    reader.start()

    nav = VoxelNavManager(data_dir=args.data_dir, learning_mode=True)
    nav.load_world(args.world)
    print(f"loaded world '{args.world}' -- {len(nav.graph)} nodes already known")

    viewer_state = {"pose": None, "target": None, "action": "starting",
                    "world": args.world}
    if args.viewer:
        start_viewer(nav, viewer_state, args.viewer_port)
        print(f"viewer at http://localhost:{args.viewer_port}")

    explorer = None
    if not args.no_explore:
        explorer = VoxelExplorer(nav, osc, learning_mode=True)
        explorer.start()
        print("reference-style explorer started")

    stopping = False

    def _sigint(_sig, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, _sigint)

    last_pose_t = 0.0
    last_print = 0.0
    last_flush = time.time()
    start = time.time()
    try:
        print("mapping... explorer will fill in the map. Ctrl+C to stop.")
        while not stopping:
            pose = reader.get()
            if pose is not None and pose.timestamp != last_pose_t:
                last_pose_t = pose.timestamp
                grounded = bool(getattr(osc, "grounded", True))
                # observe with interpolation OFF to stay closer to reference impl
                # behavior (their VisionManager fires per frame, never
                # skips cells). leaving it off lets the explorer's stuck
                # detection trigger correctly when we cant make progress.
                nav.observe(pose.x, pose.y, pose.z,
                            grounded=grounded, interpolate=True)
                if explorer is not None:
                    explorer.tick(pose.x, pose.y, pose.z, pose.yaw)
                viewer_state["pose"] = {
                    "x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw,
                }
                if explorer is not None:
                    viewer_state["target"] = (list(explorer.state.target)
                                              if explorer.state.target else None)
                    viewer_state["action"] = explorer.state.action

            now = time.time()
            if now - last_print >= 0.5:
                last_print = now
                reachable = sum(1 for n in nav.graph.nodes.values()
                                if n.node_type == NodeType.REACHABLE)
                unreachable = sum(1 for n in nav.graph.nodes.values()
                                  if n.node_type == NodeType.UNREACHABLE)
                iffy = sum(1 for n in nav.graph.nodes.values()
                           if n.node_type == NodeType.IFFY)
                cur = nav.current.serial if nav.current else "?"
                stats = reader.stats()
                action = explorer.state.action if explorer else "manual"
                tgt = (explorer.state.target if explorer else None)
                print(f"\r[{now-start:5.1f}s] reach={reachable:4d} wall={unreachable:4d} "
                      f"iffy={iffy:3d}  cur={cur}  tgt={tgt}  "
                      f"act={action[:40]:<40s}  "
                      f"decode={stats['decode_rate']*100:5.1f}%  ",
                      end="", flush=True)

            if now - last_flush >= 5.0:
                nav.flush()
                last_flush = now

            if args.duration > 0 and now - start >= args.duration:
                stopping = True

            time.sleep(0.05)
    finally:
        print()
        if explorer is not None:
            explorer.stop()
        # belt and suspenders: zero all movement inputs
        try:
            osc.client.send_message("/input/Vertical", 0.0)
            osc.client.send_message("/input/LookHorizontal", 0.0)
            osc.client.send_message("/input/Run", 0)
        except Exception:
            pass
        reader.stop()
        nav.flush()
        print(f"saved to {args.data_dir}/{args.world}.json -- "
              f"{len(nav.graph)} total cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

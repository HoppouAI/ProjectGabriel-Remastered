"""
ProjectGabriel Supervisor
Runs both main.py and music_server.py, auto-restarts on crash.
Uses the .venv virtual environment.
"""
import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# Check if venv exists
if not VENV_PYTHON.exists():
    print(f"ERROR: Virtual environment not found at {VENV_PYTHON}")
    print("Please create one with: uv venv")
    sys.exit(1)


class ProcessSupervisor:
    def __init__(self):
        self.processes = {}
        self.running = True
        self.restart_delays = {}  # Track restart delays per process
        
    def start_process(self, name: str, script: str, restart_on_exit: bool = True):
        """Start a process and optionally auto-restart it."""
        def run():
            restart_count = 0
            while self.running:
                print(f"[Supervisor] Starting {name}...")
                
                try:
                    process = subprocess.Popen(
                        [str(VENV_PYTHON), script],
                        cwd=str(PROJECT_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    self.processes[name] = process
                    
                    # Stream output
                    for line in process.stdout:
                        print(f"[{name}] {line}", end="")
                    
                    process.wait()
                    exit_code = process.returncode
                    
                except Exception as e:
                    print(f"[Supervisor] {name} error: {e}")
                    exit_code = -1
                
                if not self.running:
                    break
                    
                if restart_on_exit:
                    restart_count += 1
                    # Exponential backoff with max 30s delay
                    delay = min(2 ** restart_count, 30)
                    print(f"[Supervisor] {name} exited (code {exit_code}), restarting in {delay}s...")
                    time.sleep(delay)
                    
                    # Reset restart count after successful run (10 minutes)
                    if restart_count > 5:
                        restart_count = 0
                else:
                    print(f"[Supervisor] {name} exited (code {exit_code})")
                    break
                    
        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        return thread
    
    def stop_all(self):
        """Stop all managed processes."""
        self.running = False
        for name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"[Supervisor] Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
    
    def run(self):
        """Main supervisor loop."""
        print("=" * 50)
        print("ProjectGabriel Supervisor")
        print(f"Python: {VENV_PYTHON}")
        print("=" * 50)
        
        # Start music server (doesn't need restart usually)
        music_thread = self.start_process("music_server", "music_server.py", restart_on_exit=True)
        
        # Give music server a moment to start
        time.sleep(1)
        
        # Start main application (always restart on crash)
        main_thread = self.start_process("main", "main.py", restart_on_exit=True)
        
        # Wait for Ctrl+C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Supervisor] Shutting down...")
            self.stop_all()
            

def main():
    supervisor = ProcessSupervisor()
    
    # Handle termination signals
    def signal_handler(sig, frame):
        print("\n[Supervisor] Received termination signal")
        supervisor.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    supervisor.run()


if __name__ == "__main__":
    main()

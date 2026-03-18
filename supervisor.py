"""
ProjectGabriel Supervisor
Runs main.py with auto-restart on crash.
Control panel (with music management) is started by main.py.
Uses the .venv virtual environment.
"""
import subprocess
import sys
import os
import time
import signal
import threading
import ctypes
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# Check if venv exists
if not VENV_PYTHON.exists():
    print(f"ERROR: Virtual environment not found at {VENV_PYTHON}")
    print("Please create one with: uv venv")
    sys.exit(1)

# ANSI codes
_RST = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[96m"
_WHITE = "\033[97m"
_YELLOW = "\033[93m"
_RED = "\033[91m"


def _enable_ansi():
    """Enable ANSI escape code processing on Windows."""
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


def _sup(msg, color=_DIM):
    """Print a supervisor message."""
    print(f"  {color}>> {msg}{_RST}")


class ProcessSupervisor:
    def __init__(self):
        self.processes = {}
        self.running = True
        self.restart_delays = {}  # Track restart delays per process
        
    def start_process(self, name: str, script: str, restart_on_exit: bool = True):
        """Start a process and optionally auto-restart it."""
        def run():
            while self.running:
                _sup(f"Starting {name}...")
                
                try:
                    env = os.environ.copy()
                    env["PYTHONIOENCODING"] = "utf-8"
                    process = subprocess.Popen(
                        [str(VENV_PYTHON), script],
                        cwd=str(PROJECT_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                        env=env,
                    )
                    self.processes[name] = process
                    
                    for line in process.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    
                    process.wait()
                    exit_code = process.returncode
                    
                except Exception as e:
                    _sup(f"{name} error: {e}", _RED)
                    exit_code = -1
                
                if not self.running:
                    break
                    
                if restart_on_exit:
                    _sup(f"{name} exited (code {exit_code}), restarting...", _YELLOW)
                else:
                    _sup(f"{name} exited (code {exit_code})")
                    break
                    
        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        return thread
    
    def stop_all(self):
        """Stop all managed processes."""
        self.running = False
        for name, process in self.processes.items():
            if process and process.poll() is None:
                _sup(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
    
    def run(self):
        """Main supervisor loop."""
        _enable_ansi()

        W = 49
        t = "P R O J E C T   G A B R I E L"
        s = "Real-time VRChat AI"
        print()
        print(f"  {_CYAN}\u2554{'\u2550' * W}\u2557{_RST}")
        print(f"  {_CYAN}\u2551{_RST}{_WHITE}{_BOLD}{t:^{W}}{_RST}{_CYAN}\u2551{_RST}")
        print(f"  {_CYAN}\u2551{_RST}{_DIM}{s:^{W}}{_RST}{_CYAN}\u2551{_RST}")
        print(f"  {_CYAN}\u255a{'\u2550' * W}\u255d{_RST}")
        print()
        
        # Start main application (control panel is started within main.py)
        main_thread = self.start_process("main", "main.py", restart_on_exit=True)
        
        # Wait for Ctrl+C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print()
            _sup("Shutting down...")
            self.stop_all()
            

def main():
    supervisor = ProcessSupervisor()
    
    # Handle termination signals
    def signal_handler(sig, frame):
        print()
        _sup("Received termination signal")
        supervisor.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    supervisor.run()


if __name__ == "__main__":
    main()

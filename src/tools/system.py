import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class SystemTools(BaseTool):
    tool_key = "system"

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="getSystemSpecs",
                description="Get the host system's hardware specs: CPU, GPU, RAM (total and current usage), and storage drives.\n**Invocation Condition:** Call when asked about your computer specs, hardware, system info, or what you are running on.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if name == "getSystemSpecs":
            return self._get_system_specs()
        return None

    def _get_system_specs(self):
        import platform
        import psutil
        specs = {}
        cpu_name = platform.processor() or "Unknown"
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            cpu_name = cpu_name.strip()
            winreg.CloseKey(key)
        except Exception:
            pass
        specs["cpu"] = cpu_name
        mem = psutil.virtual_memory()
        specs["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
        specs["ram_used_gb"] = round(mem.used / (1024 ** 3), 1)
        specs["ram_percent"] = mem.percent
        try:
            import subprocess
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=10,
            )
            gpu_lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            specs["gpu"] = gpu_lines if gpu_lines else ["Unknown"]
        except Exception:
            specs["gpu"] = ["Unknown"]
        drives = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                drives.append({
                    "mount": part.mountpoint,
                    "total_gb": round(usage.total / (1024 ** 3), 1),
                    "free_gb": round(usage.free / (1024 ** 3), 1),
                })
            except Exception:
                pass
        specs["storage"] = drives
        specs["result"] = "ok"
        return specs

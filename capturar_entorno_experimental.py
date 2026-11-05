"""Captura ESPECIFICACIONES REALES, no fabricadas, del equipo que ejecuta el experimento.
Ejecutar desde la raíz del repositorio: python capturar_entorno_experimental.py
"""
from pathlib import Path
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
import json, os, platform, subprocess, sys


def pkg(nombre):
    try: return version(nombre)
    except PackageNotFoundError: return "NO INSTALADO"


def cpu_windows():
    if platform.system() != "Windows": return platform.processor() or "Por confirmar"
    try:
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()"],
            text=True, stderr=subprocess.DEVNULL, timeout=15
        ).strip()
    except Exception:
        return platform.processor() or "Por confirmar"


def memoria():
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        if platform.system()=="Windows":
            try:
                txt=subprocess.check_output(["powershell","-NoProfile","-Command","(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],text=True,timeout=15)
                return round(int(txt.strip())/(1024**3),2)
            except Exception: pass
        return "Por confirmar (instalar psutil si es necesario)"

info = {
    "fecha_captura":datetime.now().astimezone().isoformat(timespec="seconds"),
    "sistema_operativo":platform.system(),
    "version_so":platform.release(),
    "arquitectura":platform.machine(),
    "cpu_modelo":cpu_windows(),
    "cpu_procesadores_logicos":os.cpu_count(),
    "ram_fisica_gib":memoria(),
    "python":sys.version.split(" ")[0],
    "versiones_paquetes":{nombre:pkg(nombre) for nombre in
        ["numpy","pandas","scikit-learn","pyclustering","hdbscan","matplotlib","jupyter","nbclient"]},
    "limites": "La memoria RAM disponible durante cada ejecución, el modelo de disco y el número de núcleos físicos no pueden inferirse con seguridad aquí. Corroborar manualmente."
}
salida=Path("results/reproducibilidad/entorno_ejecucion.json")
salida.parent.mkdir(parents=True,exist_ok=True)
salida.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(info,ensure_ascii=False,indent=2))
print("\nGuardado en:",salida)


from pathlib import Path
import subprocess
import pandas as pd

RAIZ = Path(__file__).resolve().parent

VARIABLES = {
    "periodos_registrados",
    "promedio_global",
    "creditos_totales",
    "cursos_aprobados",
    "cursos_desaprobados",
    "cursos_previo_inicio",
    "promedio_inicial",
    "promedio_final",
    "tendencia",
    "tasa_aprobacion",
    "variabilidad_nota",
}

IDENTIFICADORES = {
    "id_estudiante", "dni", "documento_identidad",
    "codigo_matricula", "codigo_estudiante",
    "nombre", "nombres", "apellido", "apellidos"
}

def tiene_identificadores(columnas):
    return bool(
        {str(c).strip().lower() for c in columnas}
        & IDENTIFICADORES
    )

print("=== VERIFICACION RNF-03 ===")

ruta_matriz = RAIZ / "data/processed/X_clustering.csv"
X = pd.read_csv(ruta_matriz)

matriz_ok = (
    X.shape == (244, 11)
    and set(X.columns) == VARIABLES
    and not tiene_identificadores(X.columns)
)

print("Matriz de agrupamiento:", "OK" if matriz_ok else "REVISAR")

archivos = sorted(
    (RAIZ / "results").rglob("*caracterizacion*.csv")
)

caracterizaciones_ok = len(archivos) > 0

print("Caracterizaciones encontradas:", len(archivos))

for archivo in archivos:
    columnas = pd.read_csv(archivo, nrows=0).columns
    correcto = not tiene_identificadores(columnas)
    caracterizaciones_ok &= correcto

    print(
        archivo.relative_to(RAIZ),
        "OK" if correcto else "REVISAR"
    )

salida_git = subprocess.check_output(
    ["git", "ls-files", "-z"],
    cwd=RAIZ
)

rastreados = [
    ruta.replace("\\", "/")
    for ruta in salida_git.decode("utf-8").split("\x00")
    if ruta
]

riesgos_git = []

for ruta in rastreados:
    archivo = RAIZ / ruta
    nombre = archivo.name.lower()

    
    if ruta.startswith("data/") and archivo.name != ".gitkeep":
        riesgos_git.append(ruta)
    elif ruta.startswith("results/") and archivo.exists():
        if "asignaciones" in nombre:
            riesgos_git.append(ruta)
        elif archivo.suffix.lower() == ".csv":
            columnas = pd.read_csv(archivo, nrows=0).columns
            if tiene_identificadores(columnas):
                riesgos_git.append(ruta)

print("Archivos sensibles potencialmente rastreados por Git:",
      len(riesgos_git))

for ruta in riesgos_git:
    print("ARCHIVO PARA REVISAR:", ruta)


if matriz_ok and caracterizaciones_ok and not riesgos_git:
    print("\nRNF-03: VERIFICACIONES SUPERADAS")
    print("Matriz y caracterizaciones sin identificadores.")
    print("No se detectaron archivos sensibles en las rutas revisadas por Git.")
else:
    print("\nRNF-03: REQUIERE REVISION")
    print("No publiques nuevos cambios hasta revisar los hallazgos.")

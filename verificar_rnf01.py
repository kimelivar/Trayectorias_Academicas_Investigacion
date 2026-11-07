
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent
errores = []

def verificar(condicion, descripcion):
    if condicion:
        print(f"[OK] {descripcion}")
    else:
        print(f"[ERROR] {descripcion}")
        errores.append(descripcion)

print("=== VERIFICACION RNF-01 ===\n")

ruta_registros = RAIZ / "data/interim/registros_academicos_raw.csv"

verificar(
    ruta_registros.is_file(),
    "Registros academicos estructurados disponibles"
)

if ruta_registros.is_file():
    registros = pd.read_csv(ruta_registros)
    verificar(
        len(registros) == 12546,
        "12 546 registros academicos consolidados"
    )

ruta_trayectorias = RAIZ / "data/processed/trayectorias_academicas.csv"

verificar(
    ruta_trayectorias.is_file(),
    "Archivo de trayectorias disponible"
)

if ruta_trayectorias.is_file():
    trayectorias = pd.read_csv(ruta_trayectorias)
    verificar(
        len(trayectorias) == 244,
        "244 trayectorias academicas construidas"
    )

ruta_matriz = RAIZ / "data/processed/X_clustering.csv"

verificar(
    ruta_matriz.is_file(),
    "Matriz de agrupamiento disponible"
)

if ruta_matriz.is_file():
    X = pd.read_csv(ruta_matriz)

    verificar(
        X.shape == (244, 11),
        "Matriz de 244 estudiantes y 11 variables"
    )

    verificar(
        X.select_dtypes(include="number").shape[1] == 11,
        "Las 11 caracteristicas son numericas"
    )

    verificar(
        not X.isna().any().any(),
        "Matriz sin valores faltantes"
    )

modelos = {
    "K-Means": "kmeans",
    "K-Medoids": "kmedoids",
    "Jerarquico": "jerarquico",
    "HDBSCAN": "hdbscan",
}

for nombre, carpeta in modelos.items():

    directorio = RAIZ / "results" / carpeta
    print(f"\n--- {nombre} ---")

    for tipo in ("evaluacion", "asignaciones", "caracterizacion"):

        archivos = sorted(
            directorio.glob(f"*{tipo}*.csv")
        )

        verificar(
            len(archivos) > 0,
            f"{nombre}: archivo de {tipo} disponible"
        )

        if archivos:
            datos = pd.read_csv(archivos[0])

            verificar(
                not datos.empty,
                f"{nombre}: {tipo} contiene resultados"
            )

            if tipo == "asignaciones":
                verificar(
                    len(datos) == 244,
                    f"{nombre}: 244 asignaciones registradas"
                )

directorio_comparacion = RAIZ / "results/comparacion"

archivos_comparacion = (
    list(directorio_comparacion.glob("*.csv"))
    if directorio_comparacion.is_dir()
    else []
)

verificar(
    len(archivos_comparacion) > 0,
    "Resultados de comparacion experimental disponibles"
)

print("\n=== RESULTADO FINAL ===")

if not errores:
    print("RNF-01: VERIFICACIONES SUPERADAS")
    print("Matriz, agrupaciones y resultados disponibles.")
else:
    print("RNF-01: REQUIERE REVISION")
    print("Comprobaciones pendientes:", len(errores))

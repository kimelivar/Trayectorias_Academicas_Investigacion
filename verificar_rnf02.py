
from pathlib import Path
from time import perf_counter
import hashlib
import csv

import pandas as pd
import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parent
CARPETA_NOTEBOOKS = RAIZ / "notebooks"
MATRIZ = RAIZ / "data" / "processed" / "X_clustering.csv"

NOTEBOOKS = [
    "05_kmeans.ipynb",
    "06_kmedoids.ipynb",
    "07_clustering_jerarquico.ipynb",
    "08_hdbscan.ipynb",
    "09_comparacion_modelos.ipynb",
]

LIMITE_SEGUNDOS = 600
resultados = []

X = pd.read_csv(MATRIZ)
assert X.shape == (244, 11), "La matriz no tiene forma (244, 11)"

huella_inicial = hashlib.sha256(MATRIZ.read_bytes()).hexdigest()

print("VERIFICACION RNF-02")
print("Matriz de entrada: 244 estudiantes x 11 variables")
print("Limite por ejecucion: 600 segundos")

for numero in range(1, 4):
    print(f"\n=== EJECUCION {numero}/3 ===")

    inicio_total = perf_counter()

    for nombre in NOTEBOOKS:
        ruta_notebook = CARPETA_NOTEBOOKS / nombre

        notebook = nbformat.read(
            str(ruta_notebook),
            as_version=4
        )

        inicio = perf_counter()

        cliente = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={
                "metadata": {
                    "path": str(CARPETA_NOTEBOOKS)
                }
            },
            allow_errors=False
        )

        cliente.execute()

        segundos = perf_counter() - inicio
        print(f"{nombre}: {segundos:.3f} s")

    tiempo_total = perf_counter() - inicio_total

    huella_actual = hashlib.sha256(MATRIZ.read_bytes()).hexdigest()
    assert huella_actual == huella_inicial, "La matriz de entrada cambio"

    cumple = tiempo_total <= LIMITE_SEGUNDOS

    resultados.append({
        "ejecucion": numero,
        "tiempo_segundos": round(tiempo_total, 3),
        "cumple": cumple,
    })

    print(f"TOTAL EJECUCION {numero}: {tiempo_total:.3f} s")
    print("CUMPLE RNF-02:", cumple)

SALIDA = RAIZ / "results" / "reproducibilidad"
SALIDA.mkdir(parents=True, exist_ok=True)

archivo = SALIDA / "tiempos_rnf02.csv"

with archivo.open("w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(
        f,
        fieldnames=["ejecucion", "tiempo_segundos", "cumple"]
    )
    escritor.writeheader()
    escritor.writerows(resultados)

print("\n=== RESULTADO FINAL ===")
print("Tiempos:", [r["tiempo_segundos"] for r in resultados])

if all(r["cumple"] for r in resultados):
    print("RNF-02 SUPERADO: las tres ejecuciones cumplen")
else:
    print("RNF-02 NO SUPERADO: revisar los tiempos registrados")

print("Evidencia:", archivo)

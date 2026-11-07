
from pathlib import Path
import nbformat

RAIZ = Path(__file__).resolve().parent
errores = []
advertencias = []

ETAPAS = {
    "Extraccion preliminar": "01_extraccion_de_5_pdf.ipynb",
    "Extraccion completa": "02_extraccion_dataset_muestra_completa.ipynb",
    "Construccion de trayectorias": "03_construccion_trayectorias.ipynb",
    "Analisis exploratorio": "04_analisis_exploratorio.ipynb",
    "K-Means": "05_kmeans.ipynb",
    "K-Medoids": "06_kmedoids.ipynb",
    "Clustering jerarquico": "07_clustering_jerarquico.ipynb",
    "HDBSCAN": "08_hdbscan.ipynb",
    "Comparacion de modelos": "09_comparacion_modelos.ipynb",
}

print("=== VERIFICACION RNF-04 ===\n")

for etapa, nombre in ETAPAS.items():
    ruta = RAIZ / "notebooks" / nombre

    if not ruta.is_file():
        errores.append(f"Falta notebook: {nombre}")
        print(f"[ERROR] {etapa}: archivo no encontrado")
        continue

    try:
        notebook = nbformat.read(str(ruta), as_version=4)
        nbformat.validate(notebook)

        codigo = sum(
            celda.cell_type == "code"
            for celda in notebook.cells
        )

        documentacion = sum(
            celda.cell_type == "markdown"
            for celda in notebook.cells
        )

        if codigo == 0:
            errores.append(f"Sin codigo: {nombre}")
            print(f"[ERROR] {etapa}: sin celdas de codigo")
        else:
            print(
                f"[OK] {etapa}: "
                f"{codigo} celdas de codigo, "
                f"{documentacion} de documentacion"
            )

        if documentacion == 0:
            advertencias.append(
                f"Agregar documentacion a {nombre}"
            )

    except Exception as error:
        errores.append(f"Notebook invalido: {nombre}")
        print(f"[ERROR] {etapa}: {type(error).__name__}")

SOPORTE = [
    "README.md",
    "requirements.txt",
    "tests/test_flujo_hdbscan.py",
    "verificar_rnf01.py",
    "verificar_rnf02.py",
    "verificar_rnf03.py",
    "verificar_rnf06.py",
]

print("\n--- Documentacion y pruebas ---")

for nombre in SOPORTE:
    ruta = RAIZ / nombre

    if ruta.is_file() and ruta.stat().st_size > 0:
        print(f"[OK] {nombre}")
    else:
        print(f"[ERROR] {nombre}")
        errores.append(f"Archivo ausente o vacio: {nombre}")

readme = RAIZ / "README.md"

if readme.is_file():
    texto = readme.read_text(encoding="utf-8")

    faltantes = [
        nombre for nombre in ETAPAS.values()
        if nombre not in texto
    ]

    if faltantes:
        print("[ERROR] README no documenta todos los notebooks")
        errores.extend(faltantes)
    else:
        print("[OK] README documenta los nueve notebooks")

print("\n=== RESULTADO FINAL ===")

if errores:
    print("RNF-04: REQUIERE REVISION")
    for problema in errores:
        print("Pendiente:", problema)
else:
    print("RNF-04: VERIFICACIONES ESTRUCTURALES SUPERADAS")

if advertencias:
    print("\nRecomendaciones:")
    for aviso in advertencias:
        print("-", aviso)

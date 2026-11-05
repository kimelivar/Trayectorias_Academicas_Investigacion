"""Análisis de sensibilidad/repetibilidad sobre los datos autorizados del proyecto.

Ejecución desde la raíz del repositorio:
  python experimento_estabilidad_20_remuestreos.py

Dependencias: pandas numpy scikit-learn pyclustering hdbscan
Entrada: data/processed/trayectorias_academicas.csv
Salida: results/reproducibilidad/ (SOLO ESTADÍSTICAS AGREGADAS)

La ejecución NO se ha realizado con los datos del usuario por este asistente.
"""

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.kmedoids import kmedoids
import hdbscan

SEEDS = list(range(101, 121))  # 20 remuestreos reproducibles
FRACCION = 0.80             # muestras sin reemplazo
VARIABLES = [
    "periodos_registrados", "promedio_global", "creditos_totales",
    "cursos_aprobados", "cursos_desaprobados", "cursos_previo_inicio",
    "promedio_inicial", "promedio_final", "tendencia",
    "tasa_aprobacion", "variabilidad_nota",
]
ENTRADA = Path("data/processed/trayectorias_academicas.csv")
SALIDA = Path("results/reproducibilidad")


def calcular_etiquetas(z: np.ndarray, nombre: str, seed: int) -> np.ndarray:
    n = len(z)
    if nombre == "K-Means":
        return KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(z)
    if nombre == "K-Medoids":
        # Inicialización con semilla; examina también sensibilidad de la inicialización.
        medoides = np.random.default_rng(seed).choice(n, 2, replace=False).tolist()
        modelo = kmedoids(z.tolist(), medoides, ccore=False)
        modelo.process()
        etiquetas = np.full(n, -100, dtype=int)
        for k, grupo in enumerate(modelo.get_clusters()):
            etiquetas[np.asarray(grupo, dtype=int)] = k
        if np.any(etiquetas == -100):
            raise RuntimeError("K-Medoids devolvió asignaciones incompletas")
        return etiquetas
    if nombre == "Jerárquico (Ward)":
        return AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(z)
    if nombre == "HDBSCAN":
        return hdbscan.HDBSCAN(
            min_cluster_size=30, min_samples=10, metric="euclidean"
        ).fit_predict(z)
    raise ValueError(f"Modelo desconocido: {nombre}")


def medir(z: np.ndarray, labels: np.ndarray) -> dict:
    mascara = labels != -1
    y = labels[mascara]
    distintos = np.unique(y)
    salida = {
        "cobertura": round(float(mascara.mean()), 9),
        "n_asignados": int(mascara.sum()),
        "n_ruido": int((~mascara).sum()),
        "n_clusters": len(distintos),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "metrica_valida": False,
    }
    if 2 <= len(distintos) < len(y):
        salida["silhouette"] = float(silhouette_score(z[mascara], y))
        salida["davies_bouldin"] = float(davies_bouldin_score(z[mascara], y))
        salida["metrica_valida"] = True
    return salida


def resumen(series: pd.Series) -> dict:
    a = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(a):
        return dict(n_valido=0, media=np.nan, sd=np.nan, p025=np.nan, p975=np.nan)
    return {
        "n_valido": int(len(a)), "media": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a)>1 else 0.0,
        "p025": float(np.quantile(a,.025)),
        "p975": float(np.quantile(a,.975)),
    }


def ejecutar():
    if not ENTRADA.exists():
        raise FileNotFoundError(f"No se encontró {ENTRADA}. Ejecutar desde la raíz del repositorio.")
    SALIDA.mkdir(parents=True, exist_ok=True)
    datos = pd.read_csv(ENTRADA)
    faltantes = sorted(set(VARIABLES)-set(datos.columns))
    if faltantes:
        raise ValueError(f"Faltan características: {faltantes}")
    x = datos[VARIABLES].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("La matriz contiene valores faltantes o infinitos")
    n, d = x.shape
    if n != 244 or d != 11:
        raise ValueError(f"Se esperaban 244×11 y se recibió {n}×{d}; revisar procedencia")
    tam = int(round(n*FRACCION))
    modelos = ["K-Means", "K-Medoids", "Jerárquico (Ward)", "HDBSCAN"]
    resultados, particiones = [], {m: [] for m in modelos}
    for seed in SEEDS:
        indices = np.sort(np.random.default_rng(seed).choice(n, size=tam, replace=False))
        # Escalado nuevamente para cada muestra: no usar media/SD de filas excluidas.
        z = StandardScaler().fit_transform(x[indices])
        for nombre in modelos:
            labels = calcular_etiquetas(z, nombre, seed)
            fila = {"modelo": nombre, "semilla": seed, "n_muestra": tam}
            fila.update(medir(z, labels))
            resultados.append(fila)
            particiones[nombre].append((indices, labels))
        print(f"Remuestreo completado: semilla {seed}", flush=True)

    detalle = pd.DataFrame(resultados)
    # Estabilidad: ARI entre dos repeticiones en los estudiantes comunes.
    # Para HDBSCAN, ARI solo en puntos asignados EN AMBAS repeticiones;
    # la estabilidad del ruido se informa por separado (Jaccard de máscaras).
    pares = []
    for nombre, ejecuciones in particiones.items():
        for (idx1, y1), (idx2, y2) in combinations(ejecuciones, 2):
            comun, p1, p2 = np.intersect1d(idx1, idx2, return_indices=True)
            yy1, yy2 = y1[p1], y2[p2]
            comun_asignado = (yy1 != -1) & (yy2 != -1)
            ari = np.nan
            if comun_asignado.sum() >= 2 and len(np.unique(yy1[comun_asignado])) >= 2 and len(np.unique(yy2[comun_asignado])) >= 2:
                ari = float(adjusted_rand_score(yy1[comun_asignado], yy2[comun_asignado]))
            jaccard_ruido = np.nan
            if nombre == "HDBSCAN":
                ruido1, ruido2 = yy1==-1, yy2==-1
                union = int(np.sum(ruido1 | ruido2))
                jaccard_ruido = float(np.sum(ruido1 & ruido2)/union) if union else 1.0
            pares.append({"modelo": nombre, "n_comun": len(comun),
                "n_comun_ambos_asignados": int(comun_asignado.sum()),
                "ari_condicional": ari, "jaccard_ruido": jaccard_ruido})
    parejas = pd.DataFrame(pares)
    resumenes = []
    for nombre in modelos:
        dmod = detalle[detalle["modelo"]==nombre]
        pmod = parejas[parejas["modelo"]==nombre]
        for campo in ("silhouette", "davies_bouldin", "cobertura", "n_clusters"):
            resumenes.append({"modelo": nombre, "indicador":campo, **resumen(dmod[campo])})
        for campo in ("ari_condicional", "jaccard_ruido"):
            if campo == "jaccard_ruido" and nombre != "HDBSCAN":
                continue
            resumenes.append({"modelo": nombre, "indicador":campo, **resumen(pmod[campo])})
        resumenes.append({"modelo":nombre,"indicador":"tasa_metricas_validas", **resumen(dmod["metrica_valida"].astype(float))})
    resumen_df = pd.DataFrame(resumenes)
    detalle.to_csv(SALIDA/"estabilidad_detalle_20.csv", index=False, encoding="utf-8-sig")
    parejas.to_csv(SALIDA/"concordancia_pares.csv", index=False, encoding="utf-8-sig")
    resumen_df.to_csv(SALIDA/"resumen_estabilidad_20.csv", index=False, encoding="utf-8-sig")
    metadata = {"semillas": SEEDS, "fraccion": FRACCION,
        "n_completo": n, "n_submuestra": tam, "n_caracteristicas": d,
        "caracteristicas": VARIABLES,
        "hash_sha256_archivo_entrada": hashlib.sha256(ENTRADA.read_bytes()).hexdigest(),
        "nota": "Resultados propios de remuestreo. No incluyen identificadores estudiantiles. La métrica ARI de HDBSCAN condiciona a observaciones asignadas en ambas repeticiones."}
    (SALIDA/"protocolo.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nRESUMEN FINAL:\n",resumen_df.to_string(index=False))
    print("\nArchivos agregados en",SALIDA)

if __name__ == "__main__":
    ejecutar()


from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import hdbscan

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score
)

raiz = Path(__file__).resolve().parent
ruta = raiz / "data" / "processed" / "X_clustering.csv"
X = pd.read_csv(ruta)

assert X.shape == (244, 11), "Dimensiones inesperadas"

resultados = []
tolerancia = 1e-6

print("VERIFICACION RNF-06")
print("Matriz:", X.shape)

for numero in range(1, 4):
    modelo = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    etiquetas = modelo.fit_predict(X)
    mascara = etiquetas != -1

    silueta = silhouette_score(
        X[mascara], etiquetas[mascara]
    )
    davies = davies_bouldin_score(
        X[mascara], etiquetas[mascara]
    )

    tamanos = sorted(
        np.unique(etiquetas[mascara], return_counts=True)[1]
    )

    ruido = int((~mascara).sum())

    assert tamanos == [35, 166]
    assert ruido == 43
    assert abs(silueta - 0.593727) <= tolerancia
    assert abs(davies - 0.598189) <= tolerancia

    resultados.append({
        "etiquetas": etiquetas,
        "ruido": ~mascara,
        "silueta": silueta,
        "davies": davies
    })

    print(
        f"Ejecucion {numero}: "
        f"clusters={tamanos}, ruido={ruido}, "
        f"silueta={silueta:.6f}, "
        f"DB={davies:.6f}"
    )

for i, j in combinations(range(3), 2):
    a, b = resultados[i], resultados[j]

    ari = adjusted_rand_score(
        a["etiquetas"], b["etiquetas"]
    )

    mismo_ruido = np.array_equal(
        a["ruido"], b["ruido"]
    )

    mismas_metricas = (
        abs(a["silueta"] - b["silueta"]) <= tolerancia
        and abs(a["davies"] - b["davies"]) <= tolerancia
    )

    print(
        f"Comparacion {i + 1}-{j + 1}: "
        f"ARI={ari:.6f}, "
        f"mismo_ruido={mismo_ruido}, "
        f"mismas_metricas={mismas_metricas}"
    )

    assert np.isclose(ari, 1.0, atol=1e-12, rtol=0)
    assert mismo_ruido
    assert mismas_metricas

print("\nRNF-06 SUPERADO: tres ejecuciones reproducibles")

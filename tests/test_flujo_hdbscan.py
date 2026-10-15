import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


COLUMNAS = [
    "periodos_registrados", "promedio_global", "creditos_totales",
    "cursos_aprobados", "cursos_desaprobados", "cursos_previo_inicio",
    "promedio_inicial", "promedio_final", "tendencia",
    "tasa_aprobacion", "variabilidad_nota",
]


class PruebaIntegracionHDBSCAN(unittest.TestCase):
    def test_entrada_guardado_y_respuesta(self):
        raiz_proyecto = Path(__file__).resolve().parents[1]
        archivo_notebook = raiz_proyecto / "notebooks" / "08_hdbscan.ipynb"
        self.assertTrue(archivo_notebook.exists(), f"No existe {archivo_notebook}")

        with tempfile.TemporaryDirectory(prefix="test_trayectorias_") as carpeta:
            base = Path(carpeta)
            trabajo = base / "notebooks"
            datos = base / "data" / "processed"
            salida = base / "results" / "hdbscan"
            for directorio in (trabajo, datos, salida):
                directorio.mkdir(parents=True, exist_ok=True)

            rng = np.random.default_rng(42)
            bloques = [
                rng.normal(loc=centro, scale=0.22, size=(80, len(COLUMNAS)))
                for centro in (-4.5, 0.0, 4.5)
            ]
            atipicos = rng.normal(loc=11.0, scale=1.5, size=(4, len(COLUMNAS)))
            matriz = np.vstack([*bloques, atipicos])
            matriz = StandardScaler().fit_transform(matriz)
            self.assertEqual(matriz.shape, (244, 11))

            X = pd.DataFrame(matriz, columns=COLUMNAS)
            X.to_csv(datos / "X_clustering.csv", index=False)

            trayectorias = X.copy()
            self.assertEqual(trayectorias.shape, (244, 11))

            trayectorias.insert(0, "id_estudiante", np.arange(900001, 900245))
            trayectorias.to_csv(datos / "trayectorias_academicas.csv", index=False)

            nb = nbformat.read(archivo_notebook, as_version=4)
            cliente = NotebookClient(
                nb,
                timeout=240,
                kernel_name="python3",
                resources={"metadata": {"path": str(trabajo)}},
                allow_errors=False,
            )
            ejecutado = cliente.execute()

            ruta_eval = salida / "evaluacion_hdbscan.csv"
            ruta_asig = salida / "asignaciones_hdbscan.csv"
            ruta_caract = salida / "caracterizacion_clusters.csv"
            for ruta in (ruta_eval, ruta_asig, ruta_caract):
                self.assertTrue(ruta.is_file(), f"No se guardó {ruta.name}")
                self.assertGreater(ruta.stat().st_size, 0)

            evaluacion = pd.read_csv(ruta_eval)
            asignaciones = pd.read_csv(ruta_asig)
            caracterizacion = pd.read_csv(ruta_caract)
            self.assertEqual(len(evaluacion), 1)
            self.assertEqual(len(asignaciones), 244)
            self.assertEqual(asignaciones["fila"].tolist(), list(range(244)))
            self.assertIn("cluster", asignaciones.columns)
            self.assertGreater(len(caracterizacion), 0)

            labels = asignaciones["cluster"].to_numpy()
            n_ruido = int((labels == -1).sum())
            n_agrupados = int((labels != -1).sum())
            self.assertEqual(n_ruido + n_agrupados, 244)
            self.assertEqual(int(evaluacion.loc[0, "estudiantes_total"]), 244)
            self.assertEqual(int(evaluacion.loc[0, "estudiantes_agrupados"]), n_agrupados)
            self.assertEqual(int(evaluacion.loc[0, "ruido"]), n_ruido)

            mask = labels != -1
            grupos = np.unique(labels[mask])
            self.assertGreaterEqual(len(grupos), 2, "La muestra sintética debe producir >=2 clústeres")
            s = silhouette_score(X.loc[mask], labels[mask])
            db = davies_bouldin_score(X.loc[mask], labels[mask])
            self.assertAlmostEqual(float(evaluacion.loc[0, "silhouette"]), s, places=6)
            self.assertAlmostEqual(float(evaluacion.loc[0, "davies_bouldin"]), db, places=6)

            salidas = "\n".join(
                "".join(out.get("text", []))
                for celda in ejecutado.cells if celda.cell_type == "code"
                for out in celda.get("outputs", []) if out.get("output_type") == "stream"
            )
            self.assertIn("ESTUDIANTES POR GRUPO", salidas)
            self.assertIn("Asignaciones guardadas correctamente", salidas)

            print("PRUEBA SUPERADA: 244 entradas sintéticas, 3 CSV y respuesta validada")


if __name__ == "__main__":
    unittest.main()
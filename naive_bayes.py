from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np

class NaiveBayes():
    def __init__(self, ds_e, ds_p):
        self.ds_entrenamiento = ds_e
        self.ds_prueba = ds_p
        self.tablas_frecuencia = {}

    def fit(self):
        self.crear_tablas_frecuencia()
        self.evaluar_pruebas()
        self.matriz_confusion()
        

    def crear_tablas_frecuencia(self):
        
        estadisticas = {
            "Iris-setosa": {},
            "Iris-virginica": {},
            "Iris-versicolor": {}
        }

        # {
        #     'Iris-setosa':    {'media':{'col1':#, 'col2':#},
        #                        'des_est':{'nombres_colomnas':#}},
        #     'Iris-virginica': {'media':{'nombres_colomnas':#},
        #                        'des_est':{'nombres_colomnas':#}},
        #     'Iris-versicolor':{'media':{'nombres_colomnas':#},
        #                        'des_est':{'nombres_colomnas':#}},
        # }
        clases = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
        for clase in clases:
            ds_clase = self.ds_entrenamiento[self.ds_entrenamiento["iris"] == clase].iloc[:, :-1]
            media = ds_clase.mean()
            desviacion_estandar = ds_clase.std()
            
            estadisticas[clase]["media"] = media.to_dict()
            estadisticas[clase]["desviacion_estandar"] = desviacion_estandar.to_dict()

        self.tablas_frecuencia = estadisticas

    def evaluar_pruebas(self):
        clase_esperada = self.ds_prueba['iris'].values
        clase_estimada = []

        for _, instancia in self.ds_prueba.iterrows():
            probabilidades = []

            for clase in self.tablas_frecuencia:
                probabilidad_clase = 1.0

                for c in instancia.index[:-1]:
                    mu = self.tablas_frecuencia[clase]["media"][c]
                    sigma = self.tablas_frecuencia[clase]["desviacion_estandar"][c]
                    probabilidad_clase *= self.densidad_probabilidad(instancia[c], mu, sigma)

                probabilidades.append(probabilidad_clase)
            
            clase_predicha = list(self.tablas_frecuencia.keys())[np.argmax(probabilidades)]
            clase_estimada.append(clase_predicha)

        matriz_confusion = confusion_matrix(clase_esperada, clase_estimada, labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
        exactitud = accuracy_score(clase_esperada, clase_estimada)
        precision_por_clase = precision_score(clase_esperada, clase_estimada, average=None, labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
        recall_por_clase = recall_score(clase_esperada, clase_estimada, average=None, labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

        print("Matriz de Confusión:")
        print(matriz_confusion)
        print("Exactitud del modelo:", exactitud)
        print("Precision por clase:", precision_por_clase)
        print("Recall por clase:", recall_por_clase)


    def densidad_probabilidad(self, x, mu, sigma):
        return 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma)**2)

    def matriz_confusion(self):
        print("matriz confision")
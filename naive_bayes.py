import pandas as pd

class NaiveBayes():
    def __init__(self, ds_e, ds_p):
        self.ds_entrenamiento = ds_e
        self.ds_prueba = ds_p
        self.tablas_frecuencia = {}
        self.tablas_verosimilitud = {}

    def fit(self):
        self.crear_tablas_frecuencia()
        self.crear_tablas_verosimilitud()
        self.evaluar_pruebas()
        self.matriz_confusion()
        

    def crear_tablas_frecuencia(self):
        
        ds_entrenamiento_setosa = self.ds_entrenamiento[self.ds_entrenamiento["iris"] == "Iris-setosa"]
        ds_entrenamiento_virginica = self.ds_entrenamiento[self.ds_entrenamiento["iris"] == "Iris-virginica"]
        ds_entrenamiento_versicolor = self.ds_entrenamiento[self.ds_entrenamiento["iris"] == "Iris-versicolor"]

        tablas_frecuencia = {"setosa":{"sepal_l":[],"sepal_w":[], "petal_l":[], "petal_w": []},"versicolor":[],"virginica":[]}

        for columna in self.ds_entrenamiento.columns:
            frecuencia = ds_entrenamiento_setosa[columna].value_counts()
            self.tablas_frecuencia[f"{columna}_setosa"] = frecuencia

        for columna in self.ds_entrenamiento.columns:
            frecuencia = ds_entrenamiento_virginica[columna].value_counts()
            self.tablas_frecuencia[f"{columna}_virginica"] = frecuencia

        for columna in self.ds_entrenamiento.columns:
            frecuencia = ds_entrenamiento_versicolor[columna].value_counts()
            self.tablas_frecuencia[f"{columna}_versicolor"] = frecuencia


    def crear_tablas_verosimilitud(self):

        for columna in self.ds_entrenamiento.columns:
            if columna != 'Play':
                valores_unicos = self.ds_entrenamiento[columna].unique()
                for valor in valores_unicos:
                    for categoria in ['Yes', 'No']:
                        frecuencia = self.tablas_frecuencia[f"{columna}_{categoria}"].get(valor, 0)
                        total_categoria = self.tablas_frecuencia[f"{columna}_{categoria}"].sum()
                        verosimilitud = frecuencia / total_categoria
                        self.tablas_verosimilitud[f"{columna}_{categoria}_{valor}"] = verosimilitud

        total_play = self.tablas_frecuencia['Play_Yes'] + self.tablas_frecuencia['Play_No']
        self.tablas_verosimilitud['Play_Yes'] = self.tablas_frecuencia['Play_Yes'] / total_play
        self.tablas_verosimilitud['Play_No'] = self.tablas_frecuencia['Play_No'] / total_play

    def evaluar_pruebas(self):
        aciertos = 0
        total_instancias = len(self.ds_prueba)

        for _, instancia in self.ds_prueba.iterrows():
            clase_real = instancia['Play']
            instancia = instancia.drop('Play')

            # Calcular la probabilidad para cada clase ("Yes" y "No")
            probabilidad_yes = self.calcular_probabilidad('Yes', instancia)
            probabilidad_no = self.calcular_probabilidad('No', instancia)

            # Asignar la clase más probable
            clase_predicha = 'Yes' if probabilidad_yes > probabilidad_no else 'No'

            # Verificar si la predicción es correcta
            if clase_predicha == clase_real:
                aciertos += 1

        # Calcular el porcentaje de acierto
        porcentaje_acierto = aciertos / total_instancias
        return porcentaje_acierto

    def calcular_probabilidad(self, clase, instancia):
        probabilidad = 1.0
        for columna, valor in instancia.items():
            clave = f"{columna}_{clase}_{valor}"
            probabilidad *= self.tablas_v.get(clave, 0)

        return probabilidad

    def matriz_confusion(self):
        print("matriz confision")
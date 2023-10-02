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
        

    def crear_tablas_frecuencia(self):
        
        ds_entrenamiento_yes = self.ds_entrenamiento[self.ds_entrenamiento["Play"] == "Yes"]
        ds_entrenamiento_no = self.ds_entrenamiento[self.ds_entrenamiento["Play"] == "No"]

        for columna in self.ds_entrenamiento.columns:
            frecuencia = ds_entrenamiento_yes[columna].value_counts()
            self.tablas_frecuencia[f"{columna}_Yes"] = frecuencia

        for columna in self.ds_entrenamiento.columns:
            frecuencia = ds_entrenamiento_no[columna].value_counts()
            self.tablas_frecuencia[f"{columna}_No"] = frecuencia


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
        print("Evalua cada instancia y cueta aciertos totales")
        print("muestra desempeño (#%)")

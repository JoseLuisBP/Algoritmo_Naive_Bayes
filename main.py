import pandas as pd

from naive_bayes import NaiveBayes

def main():
    archivo = "datasets/iris.data"
    tam_ds_entrenamiento = 0.7
    column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'iris']

    df = pd.read_csv(archivo, names = column_names)

    ds_entrenamiento = df.sample(frac = tam_ds_entrenamiento)
    ds_prueba = df.drop(ds_entrenamiento.index)

    print(ds_prueba)
    
    modelo = NaiveBayes(ds_entrenamiento, ds_prueba)

    modelo.fit()

if __name__ == '__main__':
    main()

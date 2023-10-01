import pandas as pd

from naive_bayes import NaiveBayes

def main():
    archivo = "datasets/golf-dataset-categorical.csv"
    tam_ds_entrenamiento = 0.7
    columna_clase = 'Play'

    ds = pd.read_csv(archivo)

    print(ds)

    ds_entrenamiento = ds.sample(frac = tam_ds_entrenamiento)
    ds_prueba = ds.drop(ds_entrenamiento.index)

    # modelo = NaiveBayes()

if __name__ == '__main__':
    main()
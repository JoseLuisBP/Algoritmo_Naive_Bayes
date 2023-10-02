import pandas as pd

from naive_bayes import NaiveBayes

def main():
    archivo = "datasets/golf-dataset-categorical.csv"
    tam_ds_entrenamiento = 0.7

    df = pd.read_csv(archivo)

    ds_entrenamiento = df.sample(frac = tam_ds_entrenamiento)
    print(ds_entrenamiento)
    ds_prueba = df.drop(ds_entrenamiento.index)
    
    modelo = NaiveBayes(ds_entrenamiento, ds_prueba)

    modelo.fit()

if __name__ == '__main__':
    main()

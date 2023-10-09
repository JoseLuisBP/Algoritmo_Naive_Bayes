import pandas as pd

from naive_bayes import NaiveBayes

def main():
    archivo = "datasets/iris.data"
    tam_ds_entrenamiento = 0.7
    column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'iris']

    df = pd.read_csv(archivo, names = column_names)

    ds_entrenamiento = df.sample(frac = tam_ds_entrenamiento)
    ds_prueba = df.drop(ds_entrenamiento.index)
    
    modelo = NaiveBayes(ds_entrenamiento, ds_prueba)

    modelo.fit()

if __name__ == '__main__':
    main()

    


# ##########################Ejemplo output#########################
#
# Matriz de Confusión:
#     s  vc v    -  M
# s [[16  0  0]
#vc [ 0 14  0]
# v [ 0  2 13]]

# Exactitud del modelo:  0.9555555555555556

# Precision por clase:
#  [1.    0.875 1.   ]

# Recall por clase:
#  [1.         1.         0.86666667]
# Documentación Técnica en Markdown para el Proyecto R
### Descripción del Código
Este documento detalla el código R proporcionado en el archivo proyecto fase 2.R, el cual utiliza métodos de minería de datos como árboles de decisión y bosques aleatorios (Random Forest). A continuación, se describen las distintas secciones del código y las instrucciones necesarias para implementarlo en otro ambiente.

### 1. Cargar Librerías Necesarias
library(readr) 

library(arules)

library(rpart)

library(rpart.plot)

library(randomForest)

library(ggplot2)

### 2. Lectura y Transformación de Datos
Leer el archivo CSV

data <- read_csv('..\\edad y causas.csv')

Limpieza de Datos

data <- data %>%
  mutate(Hombres = as.numeric(gsub(",", "", Hombres)),
         Mujeres = as.numeric(gsub(",", "", Mujeres)))

### 3. Filtrado de Datos

data_filtrado <- subset(data, !(Enfermedad %in% c("Todas las causas", "Ignorado")))

### 4. Categorización de Datos
data$Grupo_Hombres <- ifelse(data$Hombres > 50000, "Alta incidencia", "Baja incidencia")

data$Grupo_Mujeres <- ifelse(data$Mujeres > 50000, "Alta incidencia", "Baja incidencia")

### 7. Implementación de Árboles y Random Forest

data_arbol <- read_csv('C:\\Users\\NOrellana\\Documents\\U\\Maestria\\data mining\\proyecto No1\\departamento y causas.csv')

# Limpieza de columnas numéricas
data_arbol$`Primera consulta` <- as.numeric(gsub(",", "", data_arbol$`Primera consulta`))

data_arbol$Reconsulta <- as.numeric(gsub(",", "", data_arbol$Reconsulta))

data_arbol$Emergencia <- as.numeric(gsub(",", "", data_arbol$Emergencia))

## Instrucciones para Implementar en Otro Ambiente

1. Instalar R y RStudio:

    a. Descargar R desde [CRAN](https://cran.r-project.org/).
  
    b. Descargar RStudio desde [Rstudio](https://posit.co/).

2. Instalar las Librerías Necesarias:

    Ejecutar el siguiente código para instalar todas las librerías:

    install.packages(c("readr", "arules", "rpart", "rpart.plot", "randomForest", "ggplot2"))

3. Actualizar Rutas de los Archivos:

    Reemplazar las rutas de archivos CSV por las rutas locales en tu sistema.

   data <- read_csv("ruta/a/tu/archivo/edad_y_causas.csv")

   data_arbol <- read_csv("ruta/a/tu/archivo/departamento_y_causas.csv")

4. Ejecutar el Código:

    Cargar el script completo en RStudio y ejecutar secciones del código utilizando Ctrl + Enter.

### Notas Adicionales

Parámetros del Algoritmo Apriori:
Ajusta los valores de support y confidence según el tamaño y características de tu conjunto de datos.
Recursos de Hardware:
Procesar grandes conjuntos de datos con Random Forest puede requerir mayor capacidad de memoria RAM.



#📄 Documentación Técnica para el Código de Redes Neuronales
### 📝 Descripción del Código
Este script utiliza una red neuronal con TensorFlow y Keras para predecir una variable a partir de un conjunto de datos CSV. Se incluyen pasos de limpieza, preprocesamiento, definición del modelo, entrenamiento y evaluación.

### 📌 Bibliotecas Utilizadas
pandas: Para la manipulación y limpieza de datos.
numpy: Para operaciones numéricas.
scikit-learn: Para dividir el conjunto de datos en entrenamiento y prueba.
TensorFlow y Keras: Para construir y entrenar una red neuronal.
### 🚀 Explicación Detallada del Código

1. Importación de Bibliotecas
    ``` python
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    ```
    Estas bibliotecas se utilizan para manipular datos, dividir el dataset y definir una red neuronal.
2. Carga y Limpieza del Conjunto de Datos
    ```python
    
    data = pd.read_csv("./departamento y causas.csv")
    data.replace(',', '.', regex=True, inplace=True)
    data.replace('-', np.nan, inplace=True)
    ```

    Carga de datos desde un archivo CSV.

    Reemplazo de comas por puntos para estandarizar datos numéricos.

    Reemplazo de guiones (-) con valores nulos (NaN).
4. Conversión de Columnas Numéricas
    ```python
    
    numerical_columns = data.columns[2:]  # Ajustar según tu archivo
    for col in numerical_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    ```
    Convierte columnas numéricas a tipo float.
5. Codificación de Variables Categóricas
    ```python
    categorical_columns = ['Grupos de edad', 'Sexo']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    ```
    Codificación One-Hot para columnas categóricas (Grupos de edad y Sexo).
6. Eliminación de Filas con Datos Faltantes
    ```python
    
    data.dropna(inplace=True)
    ```
    Elimina filas con valores nulos.
7. Separación de Características y Objetivo
    ```python
    
    y = data['Primera consulta']
    X = data.drop('Primera consulta', axis=1)
    ```
    X: Características (variables predictoras).
    y: Variable objetivo (Primera consulta).
8. División del Conjunto de Datos
    ```python
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
    Divide los datos en 80% entrenamiento y 20% prueba.
9. Definición del Modelo de Red Neuronal
    ```python
    
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    ```
    Red neuronal secuencial con capas densas:
    5 capas ocultas con funciones de activación ReLU.
    1 capa de salida con activación linear para regresión.
10. Compilación del Modelo
    ```python
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    ```
    Función de pérdida: mean_squared_error.
    Optimizador: adam.
    Métrica: mean_absolute_error.
11. Entrenamiento del Modelo
    ```python
    Copy code
    model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
    ```
    Épocas: 50.
    Batch size: 8.
12. Evaluación del Modelo
    ```python
    Copy code
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Mean Absolute Error: {mae}")
    ```
    Evalúa el modelo y muestra el error absoluto medio.
13. Predicciones
    ```python
    Copy code
    predictions = model.predict(X_test)
    print("Primeras 5 predicciones:")
    print(predictions[:5])
    ```
    Genera predicciones en el conjunto de prueba.
### 🛠️ Instrucciones para Implementar en Otro Entorno
1. Requisitos Previos

   Instalar las bibliotecas necesarias:

  ```bash
  pip install pandas numpy scikit-learn tensorflow
  ```
2. Estructura de Archivos
   
    Asegúrate de tener un archivo CSV llamado departamento y causas.csv en el mismo directorio que el script.

3. Ejecución del Código

    Guarda el script como red_neuronal.py y ejecútalo con:

    ```bash
    python red_neuronal.py
    ```


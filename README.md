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
  

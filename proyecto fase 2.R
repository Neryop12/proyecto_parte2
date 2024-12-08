# Cargar librerías necesarias
library(readr)
library(arules)
library(rpart)
library(rpart.plot)
library(readr)

library(randomForest)
library(ggplot2)

# Leer el archivo CSV
data <- read_csv('C:\\Users\\NOrellana\\Documents\\U\\Maestria\\data mining\\proyecto No1\\edad y causas.csv')

# Limpieza y transformación: eliminar comas en los números y convertir a valores numéricos
data <- data %>%
  mutate(Hombres = as.numeric(gsub(",", "", Hombres)),
         Mujeres = as.numeric(gsub(",", "", Mujeres)))


# Remover filas donde Sexo es "Total sexo" o "Ignorado"
data_filtrado <- subset(data, !(Enfermedad %in% c("Todas las causas", "Ignorado")))
# Convertir la información a transacciones categorizadas
# Aquí, asignamos categorías según el número de casos en "Hombres" y "Mujeres" para cada enfermedad
data$Grupo_Hombres <- ifelse(data$Hombres > 50000, "Alta incidencia", "Baja incidencia")
data$Grupo_Mujeres <- ifelse(data$Mujeres > 50000, "Alta incidencia", "Baja incidencia")


# Crear un dataframe de transacciones
transacciones <- data %>%
  select(Enfermedad, Grupo_Hombres, Grupo_Mujeres) %>%
  tidyr::pivot_longer(cols = c("Grupo_Hombres", "Grupo_Mujeres"),
                      names_to = "Genero", values_to = "Incidencia") %>%
  mutate(Item = paste(Enfermedad, Incidencia, sep = "=")) %>%
  select(Item)

# Crear un objeto de transacciones para arules
trans_list <- split(transacciones$Item, 1:nrow(transacciones))
trans <- as(trans_list, "transactions")

# Aplicar el algoritmo Apriori para encontrar reglas frecuentes
rules <- apriori(data_filtrado, parameter = list(support = 0.05, confidence = 0.3))
# Visualizar las reglas
inspect(rules)



data_arbol <- read_csv('C:\\Users\\NOrellana\\Documents\\U\\Maestria\\data mining\\proyecto No1\\departamento y causas.csv')


# Limpiar los datos
data_arbol$`Primera consulta` <- as.numeric(gsub(",", "", data_arbol$`Primera consulta`))
data_arbol$Reconsulta <- as.numeric(gsub(",", "", data_arbol$Reconsulta))
data_arbol$Emergencia <- as.numeric(gsub(",", "", data_arbol$Emergencia))
data_arbol$`Primera consulta y emergencia` <- as.numeric(gsub("-", NA, data_arbol$`Primera consulta y emergencia`))
data_arbol$`Reconsulta y emergencia` <- as.numeric(gsub("-", NA, data_arbol$`Reconsulta y emergencia`))


# Crear un modelo de árbol de decisión
modelo <- rpart(Reconsulta ~ Sexo + `Primera consulta` + Emergencia + 
                  `Primera consulta y emergencia` + `Reconsulta y emergencia`,
                data = data_arbol, method = "anova")

# Visualizar el árbol de decisión
rpart.plot(modelo, type = 3, extra = 101, under = TRUE, fallen.leaves = TRUE)

# Realizar predicciones
predicciones <- predict(modelo, data_arbol)

# Añadir las predicciones al dataframe original
data_arbol$Predicciones <- predicciones


head(data_arbol)

# Filtrar los datos para Mujeres
data_mujeres <- subset(data_arbol, Sexo == "Mujeres")
# Remover filas donde Sexo es "Total sexo" o "Ignorado"
data_filtrado <- subset(data_arbol, !(Sexo %in% c("Total sexo", "Ignorado")))

# Crear un modelo de árbol de decisión utilizando solo la variable Sexo
modelo_mujeres <- rpart(Reconsulta ~ Sexo, data = data_filtrado, method = "anova")

# Visualizar el árbol
rpart.plot(modelo_mujeres, type = 3, extra = 101, under = TRUE, fallen.leaves = TRUE)

# Realizar predicciones para el sexo de Mujeres
predicciones_mujeres <- predict(modelo_mujeres, data_mujeres)

# Añadir las predicciones al dataframe original
data_mujeres$Predicciones <- predicciones_mujeres

# Ver las predicciones
head(data_mujeres)

# Crear un modelo de árbol de decisión usando "Primera consulta y emergencia" para predecir "Reconsulta"
modelo_emergencia <- rpart(Reconsulta ~ `Primera consulta y emergencia`, data = data_filtrado, method = "anova")

# Visualizar el árbol de decisión
rpart.plot(modelo_emergencia, type = 3, extra = 101, under = TRUE, fallen.leaves = TRUE)
  

# Crear un modelo de árbol de decisión usando "Sexo" y "Reconsulta y emergencia" para predecir "Reconsulta"
modelo_fpgrowth <- rpart(Reconsulta ~ Sexo + `Reconsulta y emergencia`, data = data_filtrado, method = "anova")

# Visualizar el árbol de decisión
rpart.plot(modelo_fpgrowth, type = 3, extra = 101, under = TRUE, fallen.leaves = TRUE)

# Modelo segunda regla FPGrowth

# Crear un modelo de árbol de decisión usando "Primera consulta" para predecir "Reconsulta"
modelo_regla <- rpart(Reconsulta ~ `Primera consulta`, data = data_filtrado, method = "anova")

# Visualizar el árbol de decisión
rpart.plot(modelo_regla, type = 3, extra = 101, under = TRUE, fallen.leaves = TRUE)


# Limpiar los datos
data_filtrado$`Primera consulta` <- as.numeric(gsub(",", "", data_filtrado$`Primera consulta`))
data_filtrado$Reconsulta <- as.numeric(gsub(",", "", data_filtrado$Reconsulta))
data_filtrado$Emergencia <- as.numeric(gsub(",", "", data_filtrado$Emergencia))
data_filtrado$`Primera consulta y emergencia` <- as.numeric(gsub("-", 0, data_filtrado$`Primera consulta y emergencia`))
data_filtrado$`Reconsulta y emergencia` <- as.numeric(gsub("-", 0, data_filtrado$`Reconsulta y emergencia`))

data_arbol <- na.omit(data_arbol)

names(data_arbol)
# Entrenar un modelo de Random Forest para predecir "Reconsulta"
arbol <- rpart(Reconsulta ~ `Primera consulta` + Emergencia + `Reconsulta y emergencia`, data = data_filtrado, method = "anova")

# Mostrar el árbol
print(arbol)


rpart.plot(arbol, main = "Árbol de Decisión para Predecir Reconsulta", type = 3, extra = 101)

data_filtrado$Emergencia <- as.factor(ifelse(data_filtrado$Emergencia > median(data_filtrado$Emergencia), "Alta", "Baja"))
# Dividir los datos en entrenamiento y prueba
set.seed(100)
train_indices <- sample(1:nrow(data_filtrado), size = 0.7 * nrow(data_filtrado))
train_data <- data_filtrado[train_indices, ]
test_data <- data_filtrado[-train_indices, ]

colnames(data_filtrado) <- make.names(colnames(data_filtrado))

modelo_rf <- randomForest(
  Emergencia ~ Primera.consulta + Reconsulta + Primera.consulta.y.emergencia + Reconsulta.y.emergencia, 
  data = train_data, 
  ntree = 200, 
  importance = TRUE, 
  proximity = TRUE
)
# Extraer el error para graficar
error_data <- data.frame(
  Trees = 1:100,
  OOB = modelo_rf$err.rate[, "OOB"],
  Alta = modelo_rf$err.rate[, "Alta"],  # Cambia estos nombres si tus clases son diferentes
  Baja = modelo_rf$err.rate[, "Baja"]  # Cambia estos nombres si tus clases son diferentes
)

library(reshape2)
error_data_long <- melt(error_data, id.vars = "Trees", variable.name = "Type", value.name = "Error")
# Graficar con ggplot2
ggplot(error_data_long, aes(x = Trees, y = Error, color = Type)) +
  geom_line() +
  labs(
    title = "Evolución del error por número de árboles",
    x = "Árboles",
    y = "Error"
  ) +
  theme_minimal()


data_filtrado$Emergencia <- as.factor(ifelse(data_filtrado$Emergencia > median(data_filtrado$Emergencia), "Alta", "Baja"))
# Dividir los datos en entrenamiento y prueba
set.seed(123)
train_indices <- sample(1:nrow(data_filtrado), size = 0.7 * nrow(data_filtrado))
train_data <- data_filtrado[train_indices, ]
test_data <- data_filtrado[-train_indices, ]

set.seed(123)
train_indices <- sample(1:nrow(data_filtrado), size = 0.7 * nrow(data_filtrado))
train_data <- data_filtrado[train_indices, ]
test_data <- data_filtrado[-train_indices, ]

# Entrenar el modelo Random Forest con 'Primera consulta' como objetiv
modelo_rf <- randomForest(
  Primera.consulta ~ Sexo  + Primera.consulta.y.emergencia + Reconsulta.y.emergencia,
  data = train_data,
  ntree = 200,
  mtry = 10,
  importance = TRUE,
  proximity = TRUE
)
# Transformar a formato largo para ggplot
# Extraer el error del modelo Random Forest

error_data <- data.frame(
  Trees = 1:length(modelo_rf$mse),  # Número de árboles
  OOB = modelo_rf$mse  # Mean Squared Error (MSE) para regresión
)

# Si quieres añadir más líneas (opcional), como la raíz del MSE (RMSE):
error_data$RMSE <- sqrt(error_data$OOB)

error_data_long <- melt(error_data, id.vars = "Trees", variable.name = "Type", value.name = "Error")


ggplot(error_data_long, aes(x = Trees, y = Error, color = Type)) +
  geom_line() +
  labs(
    title = "Evolución del error por número de árboles (Regresión)",
    x = "Árboles",
    y = "Error"
  ) +
  theme_minimal()

head(error_data$RMSE)
predicciones <- predict(modelo_rf, test_data)
mae <- mean(abs(predicciones - test_data$`Primera.Consulta`))
cat("Error Medio Absoluto (MAE):", mae, "\n")


# Convertir las columnas necesarias a formato adecuado
data_arbol$`Reconsulta y emergencia` <- as.factor(data_arbol$`Reconsulta y emergencia`)
data_arbol$Sexo <- as.factor(data_arbol$Sexo)

# Filtrar los datos según la regla {Reconsulta y emergencia = "-"}
data_filtrado <- subset(data_arbol, Sexo == "Hombres")
data_filtrado$Sexo <- as.factor(ifelse(data_arbol$Sexo == "Hombres", "Hombres", "Otros"))

# Dividir los datos en entrenamiento y prueba
set.seed(123)
train_indices <- sample(1:nrow(data_filtrado), size = 0.7 * nrow(data_filtrado))
train_data <- data_filtrado[train_indices, ]
test_data <- data_filtrado[-train_indices, ]

table(train_data$Sexo)


colnames(data_filtrado) <- make.names(colnames(data_filtrado))


train_data$Sexo <- as.factor(train_data$Sexo)
test_data$Sexo <- as.factor(test_data$Sexo)

# Asegurarse de que los predictores son numéricos
train_data$`Primera.consulta` <- as.numeric(gsub(",", "", train_data$`Primera.consulta`))
train_data$`Primera.consulta.y.emergencia` <- as.numeric(gsub(",", "", train_data$`Primera.consulta.y.emergencia`))
train_data$`Emergencia` <- as.numeric(gsub(",", "", train_data$`Emergencia`))
train_data <- na.omit(train_data)

# Entrenar el modelo Random Forest
modelo_rf <- randomForest(
  Sexo ~ Primera.consulta + Primera.consulta.y.emergencia + Emergencia,
  data = train_data,
  ntree = 500,
  importance = TRUE,
  proximity = TRUE
)

error_data_long <- melt(error_data, id.vars = "Trees", variable.name = "Type", value.name = "Error")

ggplot(error_data_long, aes(x = Trees, y = Error, color = Type)) +
  geom_line() +
  labs(
    title = "Evolución del error por número de árboles (Clasificación)",
    x = "Árboles",
    y = "Error"
  ) +
  theme_minimal()

data_filtrado <- subset(data_arbol, !(Sexo %in% c("Total sexo", "Ignorado")))
data_filtrado$`Primera consulta` <- as.numeric(gsub(",", "", data_filtrado$`Primera consulta`))
data_filtrado$`Emergencia` <- as.numeric(gsub(",", "", data_filtrado$`Emergencia`))
data_filtrado$`Primera consulta y emergencia` <- as.numeric(gsub(",", "", data_filtrado$`Primera consulta y emergencia`))
data_filtrado$Reconsulta <- as.numeric(gsub(",", "", data_filtrado$Reconsulta))
data_filtrado <- na.omit(data_filtrado)
# Dividir los datos en conjuntos de entrenamiento y prueba
set.seed(123)
train_indices <- sample(1:nrow(data_filtrado), size = 0.7 * nrow(data_filtrado))
train_data <- data_filtrado[train_indices, ]
test_data <- data_filtrado[-train_indices, ]
# Entrenar el modelo Random Forest para predecir "Reconsulta"
modelo_rf_3 <- randomForest(
  Reconsulta ~ Primera.consulta + `Emergencia` + `Primera.consulta.y.emergencia`,
  data = train_data,
  ntree = 300, 
  importance = TRUE, 
  proximity = TRUE
)
# Realizar predicciones en los datos de prueba
predicciones <- predict(modelo_rf_3, test_data)
# Calcular métricas de evaluación
mae <- mean(abs(predicciones - test_data$Reconsulta))  
mse <- mean((predicciones - test_data$Reconsulta)^2)  
cat("Error Medio Absoluto (MAE):", mae, "\n")
cat("Error Cuadrático Medio (MSE):", mse, "\n")
# Visualizar el error del modelo por número de árboles
error_data <- data.frame(
  Trees = 1:length(modelo_rf_3$mse),  # Número de árboles
  MSE = modelo_rf_3$mse              # Error cuadrático medio
)
ggplot(error_data_long, aes(x = Trees, y = Value, color = Metric)) +
  geom_line() +
  labs(
    title = "Evolución del Error por Número de Árboles (Random Forest)",
    x = "Árboles",
    y = "Error",
    color = "Métrica"
  ) +
  theme_minimal()



# Ruta del CSV con el Dataset de Predicción de Lluvia al dia siguiente
ruta_archivo_clima <- "datasets/weatherAUS.csv"
datos_clima <- read.csv(file.path(getwd(), ruta_archivo_clima))


# Verifico de la estructura del conjunto de datos
str(datos_clima)


# Agrego librerias
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)


# Convertir variables categóricas a factores
datos_clima$Location <- as.factor(datos_clima$Location)
datos_clima$WindGustDir <- as.factor(datos_clima$WindGustDir)
datos_clima$WindDir9am <- as.factor(datos_clima$WindDir9am)
datos_clima$WindDir3pm <- as.factor(datos_clima$WindDir3pm)
datos_clima$RainToday <- as.factor(datos_clima$RainToday)
datos_clima$RainTomorrow <- as.factor(datos_clima$RainTomorrow)

# Configurar semilla para reproducibilidad
set.seed(123)

# Dividir el conjunto de datos en 70% para entrenamiento y 30% para prueba y validación
indices_entrenamiento_clima <- createDataPartition(datos_clima$RainTomorrow, p = 0.7, list = FALSE)
datos_entrenamiento_clima <- datos_clima[indices_entrenamiento_clima, ]
datos_prueba_validacion_clima <- datos_clima[-indices_entrenamiento_clima, ]

# Eliminar la variable Date antes de entrenar el modelo
datos_entrenamiento_clima <- datos_entrenamiento_clima[, -which(names(datos_entrenamiento_clima) %in% "Date")]

# Dividir el conjunto de prueba y validación en 50% para prueba y 50% para validación
indices_prueba_validacion_clima <- createDataPartition(datos_prueba_validacion_clima$RainTomorrow, p = 0.5, list = FALSE)
datos_prueba_clima <- datos_prueba_validacion_clima[indices_prueba_validacion_clima, ]
datos_validacion_clima <- datos_prueba_validacion_clima[-indices_prueba_validacion_clima, ]

# Crear el modelo de árbol de decisión usando el conjunto de entrenamiento
modelo_arbol_clima <- rpart(RainTomorrow ~ ., data = datos_entrenamiento_clima, method = "class")

# Realizar predicciones en el conjunto de prueba (sin la variable Date)
predicciones_clima <- predict(modelo_arbol_clima, datos_prueba_clima[, -which(names(datos_prueba_clima) %in% "Date")], type = "class")

# Ajustar los niveles de la variable objetivo en las predicciones
predicciones_clima <- factor(predicciones_clima, levels = levels(datos_prueba_clima$RainTomorrow))

# Convertir la variable RainTomorrow a factor en ambos conjuntos de datos
datos_prueba_clima$RainTomorrow <- factor(datos_prueba_clima$RainTomorrow, levels = levels(predicciones_clima))

# Evaluar el rendimiento del modelo en el conjunto de prueba
matriz_confusion_clima <- confusionMatrix(predicciones_clima, datos_prueba_clima$RainTomorrow)
print(matriz_confusion_clima)



# Visualización de la Matriz de Confusión - Lluvia
plot(matriz_confusion_clima$table, col = c("#FFA07A", "#90EE90"), 
     main = "Matriz de Confusión - Lluvia",
     sub = paste("Accuracy =", round(matriz_confusion_clima$overall["Accuracy"], 4)))


# Curva ROC - Lluvia
predicciones_probabilidades_clima <- predict(modelo_arbol_clima, datos_prueba_clima[, -which(names(datos_prueba_clima) %in% "Date")], type = "prob")

# Obtener la dimensión de la matriz de probabilidades
dim_probabilidades <- dim(predicciones_probabilidades_clima)

# Acceder a la columna de probabilidades de la clase positiva
probabilidad_lluvia <- predicciones_probabilidades_clima[, dim_probabilidades[2]]

# Generar la curva ROC
curva_roc_clima <- roc(as.factor(datos_prueba_clima$RainTomorrow), probabilidad_lluvia)

plot(curva_roc_clima, col = "blue", main = "Curva ROC - Lluvia")


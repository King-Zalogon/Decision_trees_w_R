# Ruta del CSV con el Dataset de Predicción de Diabetes
ruta_archivo_diabetes <- "datasets/diabetes_prediction_dataset.csv"
datos_diabetes <- read.csv(file.path(getwd(), ruta_archivo_diabetes))


# Verifico de la estructura del conjunto de datos
str(datos_diabetes)


# Agrego librerias
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)


# Convierto las variables categóricas "gender" y "smoking_history" a factores
datos_diabetes$gender <- as.factor(datos_diabetes$gender)
datos_diabetes$smoking_history <- as.factor(datos_diabetes$smoking_history)


# Creo el modelo de árbol de decisión
modelo_arbol_diabetes <- rpart(diabetes ~ ., data = datos_diabetes, method = "class")

# Ploteo el árbol de decisión
rpart.plot(modelo_arbol_diabetes)


# Ahora busco entrenarlo usando una division de los datos entre entrenamiento, prueba y validación

# Configuro semilla para reproducibilidad
set.seed(123)


# Divido el conjunto de datos en 70% para entrenamiento y 30% para prueba y validación (si esto no funciona voy a probar 75%/25%)
indices_entrenamiento <- createDataPartition(datos_diabetes$diabetes, p = 0.7, list = FALSE)
datos_entrenamiento <- datos_diabetes[indices_entrenamiento, ]
datos_prueba_validacion <- datos_diabetes[-indices_entrenamiento, ]


# Luego divido el conjunto de prueba y validación en 50% para prueba y 50% para validación
indices_prueba_validacion <- createDataPartition(datos_prueba_validacion$diabetes, p = 0.5, list = FALSE)
datos_prueba <- datos_prueba_validacion[indices_prueba_validacion, ]
datos_validacion <- datos_prueba_validacion[-indices_prueba_validacion, ]


# Creo el modelo de árbol de decisión usando el conjunto de entrenamiento
modelo_arbol_diabetes <- rpart(diabetes ~ ., data = datos_entrenamiento, method = "class")

# Realizo predicciones en el conjunto de prueba
predicciones <- predict(modelo_arbol_diabetes, datos_prueba, type = "class")

# Convierto la variable diabetes a factor en ambos conjuntos de datos
datos_prueba$diabetes <- factor(datos_prueba$diabetes, levels = levels(predicciones))

# Evalúo el rendimiento del modelo en el conjunto de prueba
matriz_confusion <- confusionMatrix(predicciones, datos_prueba$diabetes)
print(matriz_confusion)


# Como el valor de Especificidad (cantidad de casos negativos que identifica correctamente) me da bajo (0.68) en comparación a los otros, puedo jugar con el umbral de decisión, normalmente 0.5 si quiero que sea más conservador al predecir casos positivos y ver si eso cambia la especificidad.

# Ajusto el umbral de decisión
umbral_decision <- 0.7

# Realizo predicciones con el nuevo umbral de decisión
predicciones_ajustadas <- ifelse(predict(modelo_arbol_diabetes, datos_prueba, type = "prob")[, "1"] > umbral_decision, 1, 0)

# Ajusto los niveles de la variable objetivo en las predicciones
predicciones_ajustadas <- factor(predicciones_ajustadas, levels = levels(datos_prueba$diabetes))

# Luego onvierto la variable diabetes a factor en ambos conjuntos de datos
datos_prueba$diabetes <- factor(datos_prueba$diabetes, levels = levels(predicciones_ajustadas))

# Y por último evalúo el rendimiento del modelo ajustado en el conjunto de prueba
matriz_confusion_ajustada <- confusionMatrix(predicciones_ajustadas, datos_prueba$diabetes)
print(matriz_confusion_ajustada)



# Visualización de la Matriz de Confusión
plot(matriz_confusion$table, col = c("#FFA07A", "#90EE90"), 
     main = "Matriz de Confusión - Diabetes",
     sub = paste("Accuracy =", round(matriz_confusion$overall["Accuracy"], 4)))


# Curva ROC - Diabetes
predicciones_probabilidades <- predict(modelo_arbol_diabetes, datos_prueba, type = "prob")
curva_roc_diabetes <- roc(as.factor(datos_prueba$diabetes), predicciones_probabilidades[, "1"])
plot(curva_roc_diabetes, col = "blue", main = "Curva ROC - Diabetes")



using Statistics
using Flux
using Flux.Losses
using DelimitedFiles

include("PAA.jl")   


dataset = rand(Float32,5,3)
sal = rand(Bool,3,3)
dataset2 = rand(Float32,5,3)
dataset3 = rand(Float32,5,3)
sal2 = rand(Bool,3,3)
salidas =  
Bool[0 0 1 ;1 0 0 ;0 0 1 ;0 0 1 ;0 1 0 ] 
print(salidas)
exits =  
Bool[0 0 1 
1 0 0 
0 0 1 
0 1 0 
0 1 0]
ecsits =  
Bool[0 1 0 
1 0 0 
0 0 1 
0 1 0 
1 0 0]
trainClassANN([3],(dataset,exits);validationDataset=(dataset2,salidas),testDataset=(dataset3,ecsits))

params = calculateZeroMeanNormalizationParameters(dataset)
ss= normalizeZeroMean!(dataset)
s = classifyOutputs(salidas,threshold=0.99)


printConfusionMatrix(salidas,exits)



ddd









# si = crossvalidation(dataset[1:100, 5],10)
# Cargar datos y preparar entradas y salidas
# dataset = readdlm("Iris.data", ',');
# inputs = convert(Array{Float32, 2}, dataset[1:100, 1:4]);
# classes = ["Iris-setosa"]
# targets = dataset[1:100, 5]
# numIns = size(inputs,1);

# trainingDataset = (inputs, targets)
# reshaped_training_targets = reshape(trainingDataset[2], :, 1)

# ANNCrossValidation([3],inputs, targets, si)
# outputs = rand(Float32,300)
# outputs_example = classifyOutputs(outputs)
# print(inputs) # Vector aleatorio de Booleanos (puedes cambiar el tamaño según tus necesidades)
# targets_example = rand(Bool, 100) 
# print(targets)
# printConfusionMatrix(outputs_example, outputs_example)


# # Imprimir algunas muestras para verificar el preprocesamiento
# println("Muestras con el oneHotEncoding:")
# for i in 1:100
#     println("Entrada: ", inputs[i, :], " - Salida: ", targets[i, :])
# end


# minmax_params = calculateMinMaxNormalizationParameters(inputs)
# zeromean_params = calculateZeroMeanNormalizationParameters(inputs)

# min_values, max_values = minmax_params
# mean_values, std_values = zeromean_params

# println("Mínimos de cada columna: ", min_values)
# println("Máximos de cada columna: ", max_values)

# println("Medias de cada columna: ", mean_values)
# println("Desviaciones típicas de cada columna: ", std_values)

# normalizado = normalizeMinMax!(inputs)

# for i in 1:numIns
#     println("Valor ",i," normalizado: ", normalizado[i, :])
# end

# Nos dan los parámetros de normalización y se quiere modificar el array de entradas
# normalizeZeroMean!(inputs, calculateZeroMeanNormalizationParameters(inputs))
# for i in 1:150
#     println("Entrada ",i," normalizada a 0: ", inputs[i, :])
# end
# println("Datos normalizados con media 0 (modificando la matriz original):\n")


# No nos dan los parámetros de normalización y se quiere modificar el array de entradas
# normalization_params = calculateZeroMeanNormalizationParameters(inputs)
# inputs_copy = normalizeZeroMean(inputs,normalization_params)

# for i in 1:numIns
#     println("Entrada ",i," normalizada a 0: ", inputs_copy[i, :])
# end
# println("Datos normalizados con media 0 (modificando la matriz original sin parámetros):\n")

# # Nos dan los parámetros de normalización y no se quiere modificar el array de entradas (se crea uno nuevo)
# inputs_normalized = normalizeZeroMean(inputs, calculateZeroMeanNormalizationParameters(inputs))
# for i in 1:150
#     println("Entrada ",i," normalizada a 0: ", inputs_normalized[i, :])
# end
# println("Nueva matriz normalizada con media 0:\n")

# # No nos dan los parámetros de normalización y no se quiere modificar el array de entradas (se crea uno nuevo)
# inputs_copy = copy(inputs)
# inputs_normalized_copy = normalizeZeroMean(inputs_copy)
# for i in 1:numIns
#     println("Entrada ",i," normalizada a 0: ", inputs_normalized_copy[i, :])
# end
# println("Nueva matriz normalizada con media 0 (creando una copia):\n")



# filas = 5
# columnas = 3
# # Supongamos que tienes una matriz de salidas (ejemplo)
# outputs = rand(Float32, filas, columnas)
# outputs2 = rand(Float32, filas, columnas)



# Llamada a classifyOutputs
# classifications1 = classifyOutputs(outputs)
# classifications2 = classifyOutputs(outputs2)
# print(classifications1)
# print(classifications2)




# println("Matriz de salidas:")
# println("Valor salida: ", outputs)
# println("Valor clasificado: ", classifications1)

# Llamada a accuracy

# diferencia = accuracy(outputs,classifications2)
# println(classifications1, classifications2)
# println("Precisión del ",diferencia * 100 ,"%")


# Definir la función σ
# Definir funciones de transferencia personalizadas
# mi_funcion_σ(x) = 1 / (1 + exp(-x))
# mi_funcion_tanh(x) = tanh(x)

# Definir la topología, el número de entradas y salidas
# topology = [5, 3]  # Por ejemplo, dos capas ocultas con 5 y 3 neuronas respectivamente
# numInputs = 3   # Número de neuronas de entrada
# numOutputs = 5    # Número de neuronas de salida (clasificación binaria)

# Crear la RNA con funciones de transferencia por defecto (σ)
# ann_default = buildClassANN(numInputs, topology, numOutputs)

# # Crear la RNA con funciones de transferencia personalizadas
# transferFunctions_custom = [mi_funcion_σ, mi_funcion_tanh]
# ann_custom = buildClassANN(numInputs, topology, numOutputs, transferFunctions=transferFunctions_custom)
# println(ann_custom)
# # # Crear conjuntos de datos de prueba
# inputs = rand(3, numInputs)  # 100 ejemplos de entrenamiento
# targets = rand(Bool, 3, numOutputs)  # Etiquetas de clasificación binaria
# print(ann_custom)
# # Entrenar las redes con conjuntos de datos de prueba
# trained_ann_default, losses_default = trainClassANN(topology, (inputs, targets);)
# # trained_ann_custom, losses_custom = trainClassANN(topology, (inputs, targets), transferFunctions=transferFunctions_custom)

# # Verificar los resultados
# println("Red Neuronal Entrenada con funciones de transferencia por defecto:")
# println(trained_ann_default)

# println("\nRed Neuronal Entrenada con funciones de transferencia personalizadas:")
# # println(trained_ann_custom)

# println("\nPérdidas durante el entrenamiento con funciones de transferencia por defecto:")
# println(losses_default)

# println("\nPérdidas durante el entrenamiento con funciones de transferencia personalizadas:")
# # println(losses_custom)
# pval=  0.2
# ptest = 0.3
# x = holdOut(10,0.3,0.2)
# println(x)



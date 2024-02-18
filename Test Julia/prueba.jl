using Flux
using DelimitedFiles
include("PAA.jl")   



# Cargar datos y preparar entradas y salidas
dataset = readdlm("Iris.data", ',');
inputs = convert(Array{Float32, 2}, dataset[:, 1:4]);
targets = oneHotEncoding(dataset[:, 5]);
numIns = length(inputs);

# Imprimir algunas muestras para verificar el preprocesamiento
println("Muestras con el oneHotEncoding:")
for i in 1:150
    println("Entrada: ", inputs[i, :], " - Salida: ", targets[i, :])
end


minmax_params = calculateMinMaxNormalizationParameters(inputs)
zeromean_params = calculateZeroMeanNormalizationParameters(inputs)

min_values, max_values = minmax_params
mean_values, std_values = zeromean_params

println("Mínimos de cada columna: ", min_values)
println("Máximos de cada columna: ", max_values)

println("Medias de cada columna: ", mean_values)
println("Desviaciones típicas de cada columna: ", std_values)

normalizado = normalizeMinMax!(inputs)

for i in 1:150
    println("Valor ",i," normalizado: ", normalizado[i, :])
end

# Nos dan los parámetros de normalización y se quiere modificar el array de entradas
# normalizeZeroMean!(inputs, calculateZeroMeanNormalizationParameters(inputs))
# for i in 1:150
#     println("Entrada ",i," normalizada a 0: ", inputs[i, :])
# end
# println("Datos normalizados con media 0 (modificando la matriz original):\n")


# No nos dan los parámetros de normalización y se quiere modificar el array de entradas
# inputs_copy = copy(inputs)
# normalizeZeroMean!(inputs_copy)
# for i in 1:150
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
inputs_copy = copy(inputs)
inputs_normalized_copy = normalizeZeroMean(inputs_copy)
for i in 1:150
    println("Entrada ",i," normalizada a 0: ", inputs_normalized_copy[i, :])
end
println("Nueva matriz normalizada con media 0 (creando una copia):\n")


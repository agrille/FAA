using Statistics
using Flux
using Flux.Losses
using DelimitedFiles

include("PAA.jl")

x = 5
y = 5
matriz = rand(Float16, x, y)


outputs = rand(Float32, 5,5)

print(classifyOutputs(outputs))
np = calculateZeroMeanNormalizationParameters(outputs)
normalizeZeroMean!(outputs)
print(outputs)

accuracy(outputs, outputs)

# Definir los vectores de salida y objetivos
outputs = [1, 0, 1, 0, 1]
targets = [1, 1, 0, 0, 1]

# Llamar a la función y recibir los resultados
precision, error_rate, sensitivity, specificity, precision_pos, precision_neg, f1_score, confusion_matrix = confusionMatrix(outputs, targets)

# Imprimir los resultados
println("Precision: ", precision)
println("Error Rate: ", error_rate)
println("Sensitivity: ", sensitivity)
println("Specificity: ", specificity)
println("Precision Positive: ", precision_pos)
println("Precision Negative: ", precision_neg)
println("F1 Score: ", f1_score)
println("Confusion Matrix:\n", confusion_matrix)

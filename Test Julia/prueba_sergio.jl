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
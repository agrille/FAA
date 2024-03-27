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
ERROR: MethodError: no method matching accuracy(::Matrix{Float32}, ::Matrix{Float32})

Closest candidates are:
  accuracy(::AbstractMatrix{<:Real}, ::AbstractMatrix{Bool}; threshold)
   @ Main c:\Users\sergi\Documents\GitHub\FAA\Test Julia\PAA.jl:198

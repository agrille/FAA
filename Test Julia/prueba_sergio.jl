using Statistics
using Flux
using Flux.Losses
using DelimitedFiles

include("PAA.jl")

x = 5
y = 4
matriz = rand(Float16, x, y)


outputs = rand(Float32, 10)

print(classifyOutputs(outputs))

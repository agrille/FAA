
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 2 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses


# -------------------------------------------------------------------------
# Funciones para codificar entradas y salidas categóricas

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(unique(classes))
    
    if numClasses == 2
        # Si hay dos clases, convierte el vector en una matriz de 1 columna y de valores booleanos
        # Si coinciden los valores característica con clase, devuelve true, sino devuelve false
        encoded_matrix = reshape(feature .== classes[1], :, 1)
    else
        # Si hay más de dos clases, se crea una matriz booleana con una columna por clase
        encoded_matrix = falses(length(feature), numClasses)
        
        # Itera sobre cada columna/categoría
        for class_index in 1:numClasses
            # Asigna los valores de esa columna como el resultado de comparar el vector feature con la categoría correspondiente
            encoded_matrix[:, class_index] .= (feature .== classes[class_index])
        end
    end
    
    return encoded_matrix
end
# Esta funcion es similar a la anterior, pero si no es especifican las clases, se toman de la propia variable
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature)) 

# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
# En este caso, el propio vector ya está codificado, simplemente lo convertimos a una matriz columna
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)
# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
# la llamada a la función correspondiente


# -------------------------------------------------------------------------
# Funciones para calcular los parametros de normalizacion y normalizar

# Para calcular los parametros de normalizacion, segun la forma de normalizar que se desee:
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real, 2})
    min_values = minimum(dataset, dims=1)
    max_values = maximum(dataset, dims=1)
    return (min_values, max_values)
end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real, 2})
    mean_values = mean(dataset, dims=1)
    std_values = std(dataset, dims=1)
    return (mean_values, std_values)
end

# 4 versiones de la funcion para normalizar entre 0 y 1:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalization_params::NTuple{2, AbstractArray{<:Real,2}})
    min_values, max_values = normalization_params
    dataset .-= min_values
    dataset ./= (max_values .- min_values)
    dataset[:, vec(min_values .== max_values)] .= 0
    return dataset
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalization_params = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, normalization_params)
end
function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalization_params::NTuple{2, AbstractArray{<:Real,2}})
    return normalizeMinMax!(copy(dataset), normalization_params)
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    return normalizeMinMax!(copy(dataset))
end

# 4 versiones similares de la funcion para normalizar de media 0:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mean_values, _ = normalizationParameters
    dataset .-= mean_values
    return dataset
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalization_params = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, normalization_params)
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    data_normalized = copy(dataset)
    return normalizeZeroMean!(data_normalized, normalizationParameters)
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    data_normalized = copy(dataset)
    return normalizeZeroMean!(data_normalized)
end


# -------------------------------------------------------
# Funcion que permite transformar una matriz de valores reales con las salidas del clasificador o clasificadores en una matriz de valores booleanos con la clase en la que sera clasificada

function classifyOutputs(outputs::AbstractArray{<:Real, 1}; threshold::Real=0.5) 
    
    vec = outputs .>= threshold
    return reshape(vec, :, 1)
end
    


function classifyOutputs(outputs::AbstractArray{<:Real, 2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        # Si tiene una columna, convertir a vector y devolver el resultado directamente
        outputs_vec = vec(outputs)
        return classifyOutputs(outputs_vec; threshold = threshold)
        

    else
        # Si tiene más de una columna, obtener los índices de los máximos en cada fila
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        
        # Crear una matriz booleana de la misma dimensionalidad que la matriz de salidas
        output_matrix = falses(size(outputs))
        
        # Asignar a true los valores de los índices que contienen los máximos de cada fila
        output_matrix[indicesMaxEachInstance] .= true
        
        return output_matrix
    end
end

# -------------------------------------------------------
# Funciones para calcular la precision

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "Las matrices de salidas y objetivos deben tener la misma longitud"
    
    correct_predictions = sum(outputs .== targets)
    total_predictions = length(outputs)
    
    return correct_predictions / total_predictions
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert length(outputs) == length(targets) "Las matrices de salidas y objetivos deben tener la misma longitud"
    if size(outputs, 2) == 1
        # Si solo tienen una columna, llamamos a la función anterior
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        # Si el número de columnas es mayor que 2, comparamos ambas matrices
        return mean(outputs .== targets)
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    predictions = classifyOutputs(outputs, threshold = threshold)
    return accuracy(predictions[:, 1], targets[:, 1])
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert size(outputs) == size(targets) "Las matrices de salidas y objetivos deben tener la misma dimensión"
    
    if size(outputs, 2) == 1
        # Si solo tienen una columna, llamamos a la función anterior
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        # Si el número de columnas es mayor que 2, convertimos outputs a booleanos
        predictions = classifyOutputs(outputs, threshold=threshold)
        return accuracy(predictions, targets)
    end
end

# -------------------------------------------------------
# Funciones para crear y entrenar una RNA
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    @assert length(topology) >= 0 "El tamaño de la topología debe ser mayor o igual a cero"

    # Crear una RNA vacía
    ann = Chain()

    # Crear una variable numInputsLayer
    numInputsLayer = numInputs

    # Añadir capas ocultas
    for (numOutputsLayer, transferFunction) in zip(topology, transferFunctions)
        transferFunction = isempty(transferFunctions) ? σ : transferFunction
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunction))
        numInputsLayer = numOutputsLayer
    end

    # Añadir capa final
    if numOutputs == 2
        # Problema de clasificación binaria
        ann = Chain(ann, Dense(numInputsLayer, 1, σ))
    else
        # Problema de clasificación multiclase
        ann = Chain(ann, Dense(numInputsLayer, numOutputs), softmax)
    end

    return ann
end


function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    # Obtener el número de neuronas de entrada y salida
    numInputs, numOutputs = size(dataset[1], 1), size(dataset[2], 2)

    # Convertir dataset a Float32 si no lo es
    dataset_float32 = (convert(Array{Float32, 2}, dataset[1]), convert(Array{Bool, 2}, dataset[2]))

    # Crear la RNA
    ann = buildClassANN(numInputs, topology, numOutputs, transferFunctions=transferFunctions)

    # Inicializar el vector de loss
    losses = Float32[]

    # Crear la función de pérdida
    loss(ann, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Crear el optimizador
    opt = Flux.setup(Adam(learningRate), ann)

    # Entrenar la RNA
    for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(dataset_float32[1]', dataset_float32[2]')], opt)

        # Calcular la pérdida
        current_loss = crossentropy(ann(dataset_float32[1]'), dataset_float32[2]')

        # Agregar la pérdida al vector
        push!(losses, current_loss)

        # Verificar el criterio de parada
        if current_loss ≤ minLoss
            break
        end
    end

    return ann, losses
end

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    # Convertir las salidas deseadas a una matriz de una sola columna
    targets_matrix = reshape(targets, :, 1)

    # Llamar a la función principal
    return trainClassANN(topology, (inputs, targets_matrix), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)
end




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 3 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    #
    # Codigo a desarrollar
    #
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    #
    # Codigo a desarrollar
    #
end;


# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test. Estos dos ultimos son opcionales
# Es la funcion anterior, modificada para calcular errores en los conjuntos de validacion y test y realizar parada temprana si es necesario
function trainClassANN2(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;


function trainClassANN2(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 4 ---------------------------------------------
# ----------------------------------------------------------------------------------------------



function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 5 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;







function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20, showText::Bool=false)
    #
    # Codigo a desarrollar
    #
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 6 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using ScikitLearn: @sk_import, fit!, predict

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    #
    # Codigo a desarrollar
    #
end;
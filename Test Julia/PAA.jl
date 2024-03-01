
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

# Nota según aplicación: 0.02
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

# Nota según aplicación: 0.02
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

# Nota según aplicación: 0.02
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

# Nota según aplicación: 0.02
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



# normalizeZeroMean!:
# Matrices incorrectas al normalizar de media 0 con parametros AbstractArray{<:Real,2}
# Matrices incorrectas al normalizar de media 0 con parametros (AbstractArray{<:Real,2}, NTuple{2, AbstractArray{<:Real,2}})
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mean_values, _ = normalizationParameters

    if size(dataset, 2) == size(mean_values, 2)
        dataset .-= mean_values
        return dataset
    else
        throw(ArgumentError("Error"))
    end
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalization_params = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, normalization_params)
end


# Nota según aplicación: 0.02
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

# classifyOutputs:
# Salidas incorrectas al clasificar los patrones en un vector: no devuelve un vector de valores booleanos
function classifyOutputs(outputs::AbstractArray{<:Real, 1}; threshold::Real=0.5) 
    
    vec = outputs .>= threshold
    return vec
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

# accuracy:
# Salidas incorrectas al hacer el calculo con parametros (AbstractArray{Bool,2},AbstractArray{Bool,2} al usar valores booleanos de mas de una columna
# Salidas incorrectas al hacer el calculo con parametros (AbstractArray{<:Real,2}, AbstractArray{Bool,2}; threshold::Real=0.5) al usar una matriz de valores reales como salidas y una matriz de valores booleanos como salidas deseadas, ambas de una columna, con un umbral distinto
# Salidas incorrectas con parametros (AbstractArray{<:Real,2}, AbstractArray{Bool,2}; threshold::Real=0.5) al hacer el calculo con una matriz de valores reales como salidas y una matriz de valores booleanos como salidas deseadas, ambas de mas de una columna

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
    predictions = classifyOutputs(outputs, threshold=threshold)
    return accuracy(predictions[:, 1], targets[:, 1])
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert size(outputs) == size(targets) "Las matrices de salidas y objetivos deben tener la misma dimensión"
    
    if size(outputs, 2) == 1
        # Si solo tienen una columna, llamamos a la función anterior
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        # Si el número de columnas es mayor que 1, convertimos outputs a booleanos
        predictions = classifyOutputs(outputs, threshold=threshold)
        return accuracy(predictions, targets)
    end
end

# -------------------------------------------------------
# Funciones para crear y entrenar una RNA


# buildClassANN:
# RNA incorrecta con 2 clases: nÃºmero de capas incorrecto
# RNA incorrecta con 2 clases: se aplica la funciÃ³n softmax y no deberÃ­a
# Error al ejecutar la funciÃ³n con 2 clases: type Chain has no field Ïƒ
# RNA incorrecta con mÃ¡s de 2 clases: numero de capas incorrecto
# Error al ejecutar la funciÃ³n con mÃ¡s de 2 clases: type Chain has no field Ïƒ
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    @assert length(topology) > 0 "El tamaño de la topología debe ser mayor a cero"

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
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        # Problema de clasificación multiclase
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs), softmax)
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

# holdOut:
# Salida incorrecta el ejecutar con argumentos de tipo (Int, Real, Real): el vector de Ã­ndices de validacion tiene una longitud incorrecta
# Salida incorrecta el ejecutar con argumentos de tipo (Int, Real, Real): el vector de Ã­ndices de test tiene una longitud incorrecta


function holdOut(N::Int, P::Real)
    @assert 0 <= P <= 1 "El valor de P debe estar entre 0 y 1"

    # Obtener el número de patrones para el conjunto de test
    numTest = round(Int, N * P)

    # Obtener los índices de todos los patrones
    all_indices = randperm(N)

    # Tomar los primeros numTest índices para el conjunto de test
    test_indices = all_indices[1:numTest]

    # Tomar el resto de los índices para el conjunto de entrenamiento
    train_indices = all_indices[numTest+1:end]

    return train_indices, test_indices
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert 0 <= Pval +  Ptest <= 1 "No puedes usar más patrónes de los que tienes, Pval y Ptest deben estar entre 0 y 1"


    # Obtener los índices para el conjunto de test
    train_indices, remaining_indices = holdOut(N, Pval + Ptest)

    # Obtener los índices para el conjunto de validación
    val_indices, test_indices = holdOut(length(remaining_indices), Pval / (Pval + Ptest))

    # Ajustar los índices a los originales
    val_indices = remaining_indices[val_indices]
    test_indices = remaining_indices[test_indices]

    return train_indices, val_indices, test_indices
end;


# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test. Estos dos ultimos son opcionales
# Es la funcion anterior, modificada para calcular errores en los conjuntos de validacion y test y realizar parada temprana si es necesario
# function trainClassANN(topology::AbstractArray{<:Int,1},
#     trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
#     validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
#     testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
#     transferFunctions::AbstractArray{<:Function,1}=fill(mi_funcion_σ, length(topology)),
#     maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

#    #
# end



# function trainClassANN(topology::AbstractArray{<:Int,1},
#     trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
#     validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
#     testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
#     transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
#     maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

#     #
# end




function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    # Extraer las matrices de entradas y salidas deseadas
    inputs, targets = dataset

    # Obtener el número de neuronas de entrada y salida
    numInputs, numOutputs = size(inputs, 2), size(targets, 2)

    # Crear la red neuronal con la topología proporcionada
    ann = buildClassANN(numInputs, topology, numOutputs, transferFunctions=transferFunctions)

    # Vector para almacenar los valores de loss en cada ciclo de entrenamiento
    lossValues = Vector{Float32}()
    opt = Flux.setup(Adam(learningRate), ann)
    # Entrenamiento de la red neuronal
    for epoch in 1:maxEpochs
        # Entrenar un ciclo
        Flux.train!(loss, ann, [(dataset_float32[1]', dataset_float32[2]')], opt)


        # Calcular el valor de loss en cada ciclo de entrenamiento
        loss = Flux.crossentropy(ann(inputs'), targets')
        push!(lossValues, loss)

        # Criterio de parada: si la pérdida es menor que minLoss, detener el entrenamiento
        if loss < minLoss
            break
        end
    end

    return ann, lossValues
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    # Si el conjunto de validación no está vacío, se establece la parada temprana
    early_stopping = !isempty(validationDataset)

    # Inicializar los vectores de loss
    training_loss = Float32[]
    validation_loss = Float32[]
    test_loss = Float32[]

    # Inicializar la mejor RNA encontrada hasta el momento
    best_model = Chain(Dense(size(trainingDataset[1], 2), topology[1], transferFunctions[1]),
        [Dense(topology[i], topology[i+1], transferFunctions[i+1]) for i = 1:length(topology)-1]...,
        Dense(topology[end], size(trainingDataset[2], 2)))

    # Inicializar el mejor loss de validación encontrado hasta el momento
    best_validation_loss = Inf

    # Inicializar el contador de épocas sin mejorar el loss de validación
    epochs_without_improvement = 0

    # Ciclo de entrenamiento
    for epoch in 0:maxEpochs-1
        # Realizar un ciclo de entrenamiento
        Flux.train!(loss, params(best_model), trainingDataset, ADAM(learningRate))

        # Calcular el loss en el conjunto de entrenamiento
        push!(training_loss, loss(best_model, trainingDataset...))

        # Si se proporcionó un conjunto de validación, calcular el loss en el conjunto de validación
        if early_stopping
            validation_loss_current = loss(best_model, validationDataset...)
            push!(validation_loss, validation_loss_current)

            # Verificar si se ha encontrado un mejor loss de validación
            if validation_loss_current < best_validation_loss
                best_model = deepcopy(best_model) # Guardar una copia de la mejor RNA
                best_validation_loss = validation_loss_current
                epochs_without_improvement = 0
            else
                epochs_without_improvement += 1
            end

            # Verificar si se ha alcanzado el criterio de parada temprana
            if epochs_without_improvement >= maxEpochsVal
                break
            end
        end
    end

    # Calcular el loss en el conjunto de test si está disponible
    if !isempty(testDataset)
        test_loss = loss(best_model, testDataset...)
    end

    return (best_model, training_loss, validation_loss, test_loss)
end


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 4 ---------------------------------------------
# ----------------------------------------------------------------------------------------------



function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Calcular la matriz de confusión
    TP = sum(outputs .& targets)
    TN = sum(!outputs .& !targets)
    FP = sum(outputs .& !targets)
    FN = sum(!outputs .& targets)

    # Calcular las métricas de evaluación
    accuracy = (TP + TN) / length(outputs)
    error_rate = 1 - accuracy

    sensitivity = TP == 0 ? 1.0 : TP / (TP + FN)
    specificity = TN == 0 ? 1.0 : TN / (TN + FP)
    precision = TP == 0 ? 1.0 : TP / (TP + FP)
    negative_predictive_value = TN == 0 ? 1.0 : TN / (TN + FN)

    f1_score = sensitivity + precision == 0 ? 0 : 2 * sensitivity * precision / (sensitivity + precision)

    # Crear la matriz de confusión
    confusion_matrix = [TN FP; FN TP]

    return (accuracy, error_rate, sensitivity, specificity, precision, negative_predictive_value, f1_score, confusion_matrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Convertir las salidas a valores booleanos basados en el umbral
    outputs_bool = outputs .>= threshold

    # Calcular las métricas de evaluación
    confusionMatrix(outputs_bool, targets)
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Verificar que el número de columnas en ambas matrices es igual y distinto de 2
    if size(outputs, 2) != size(targets, 2) || size(outputs, 2) == 2
        error("Las matrices deben tener el mismo número de columnas distinto de 2.")
    end

    num_classes = size(outputs, 2)

    # Vectores para almacenar las métricas por clase
    sensitivity = zeros(Float32, num_classes)
    specificity = zeros(Float32, num_classes)
    precision_pos = zeros(Float32, num_classes)
    precision_neg = zeros(Float32, num_classes)
    f1_score = zeros(Float32, num_classes)

    # Matriz de confusión
    confusion = zeros(Int, num_classes, num_classes)

    # Calcular métricas para cada clase
    for c in 1:num_classes
        # Extraer las columnas correspondientes a la clase actual
        output_class = outputs[:, c]
        target_class = targets[:, c]

        # Calcular métricas usando confusionMatrix de la práctica anterior
        accuracy, errorRate, sensitivity[c], specificity[c], precision_pos[c], precision_neg[c], f1_score[c], confusion_class = confusionMatrix(output_class, target_class)

        # Actualizar la matriz de confusión
        confusion[c, :] = confusion_class[1, :]
    end

    # Calcular las métricas macro o weighted
    if weighted
        # Ponderar las métricas por la cantidad de ejemplos de cada clase
        total_samples_per_class = sum(targets, dims=1)
        total_samples = sum(total_samples_per_class)

        sensitivity = sum(sensitivity .* total_samples_per_class) / total_samples
        specificity = sum(specificity .* total_samples_per_class) / total_samples
        precision_pos = sum(precision_pos .* total_samples_per_class) / total_samples
        precision_neg = sum(precision_neg .* total_samples_per_class) / total_samples
        f1_score = sum(f1_score .* total_samples_per_class) / total_samples
    else
        # Calcular las métricas promedio macro
        sensitivity = mean(sensitivity)
        specificity = mean(specificity)
        precision_pos = mean(precision_pos)
        precision_neg = mean(precision_neg)
        f1_score = mean(f1_score)
    end

    # Calcular precisión y tasa de error
    accuracy = mean(diag(confusion)) / sum(confusion)
    error_rate = 1 - accuracy

    # Devolver las métricas calculadas
    return (accuracy, error_rate, sensitivity, specificity, precision_pos, precision_neg, f1_score, confusion)
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Convertir las salidas del modelo a valores booleanos
    outputs_bool = classifyOutputs(outputs)

    # Llamar a la función confusionMatrix con los nuevos parámetros
    return confusionMatrix(outputs_bool, targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Asegurar que todas las clases de outputs estén incluidas en las clases de targets
    if !all(output -> output in targets, outputs)
        error("No todas las clases de outputs están incluidas en las clases de targets.")
    end

    # Obtener las posibles clases
    classes = unique([outputs; targets])

    # Codificar las salidas del modelo y las salidas deseadas
    outputs_encoded = oneHotEncoding(outputs, classes)
    targets_encoded = oneHotEncoding(targets, classes)

    # Llamar a la función confusionMatrix con los nuevos parámetros
    return confusionMatrix(outputs_encoded, targets_encoded; weighted=weighted)
end


function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Calcular la matriz de confusión
    metrics = confusionMatrix(outputs, targets)

    # Extraer los valores de las métricas
    accuracy, errorRate, sensitivity, specificity, precisionPos, precisionNeg, f1Score, confusion = metrics

    # Imprimir las métricas
    println("Accuracy: $accuracy")
    println("Error Rate: $errorRate")
    println("Sensitivity: $sensitivity")
    println("Specificity: $specificity")
    println("Precision Pos: $precisionPos")
    println("Precision Neg: $precisionNeg")
    println("F1 Score: $f1Score")

    # Imprimir la matriz de confusión
    println("Confusion Matrix:")
    println(confusion)
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Calcular la matriz de confusión
    metrics = confusionMatrix(outputs, targets; threshold=threshold)

    # Extraer los valores de las métricas
    accuracy, errorRate, sensitivity, specificity, precisionPos, precisionNeg, f1Score, confusion = metrics

    # Imprimir las métricas
    println("Accuracy: $accuracy")
    println("Error Rate: $errorRate")
    println("Sensitivity: $sensitivity")
    println("Specificity: $specificity")
    println("Precision Pos: $precisionPos")
    println("Precision Neg: $precisionNeg")
    println("F1 Score: $f1Score")

    # Imprimir la matriz de confusión
    println("Confusion Matrix:")
    println(confusion)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Calcular la matriz de confusión
    confusion = confusionMatrix(outputs, targets; weighted=weighted)

    # Mostrar resultados
    println("Confusion Matrix:")
    println("Precision: $(confusion[1])")
    println("Error Rate: $(confusion[2])")
    println("Sensitivity (Recall): $(confusion[3])")
    println("Specificity: $(confusion[4])")
    println("Positive Predictive Value: $(confusion[5])")
    println("Negative Predictive Value: $(confusion[6])")
    println("F1 Score: $(confusion[7])")
    println("Matrix:")
    display(confusion[8])
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Calcular la matriz de confusión
    confusion = confusionMatrix(outputs, targets; weighted=weighted)

    # Mostrar resultados
    println("Confusion Matrix:")
    println("Precision: $(confusion[1])")
    println("Error Rate: $(confusion[2])")
    println("Sensitivity (Recall): $(confusion[3])")
    println("Specificity: $(confusion[4])")
    println("Positive Predictive Value: $(confusion[5])")
    println("Negative Predictive Value: $(confusion[6])")
    println("F1 Score: $(confusion[7])")
    println("Matrix:")
    display(confusion[8])
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Calcular la matriz de confusión
    confusion = confusionMatrix(outputs, targets; weighted=weighted)

    # Mostrar resultados
    println("Confusion Matrix:")
    println("Precision: $(confusion[1])")
    println("Error Rate: $(confusion[2])")
    println("Sensitivity (Recall): $(confusion[3])")
    println("Specificity: $(confusion[4])")
    println("Positive Predictive Value: $(confusion[5])")
    println("Negative Predictive Value: $(confusion[6])")
    println("F1 Score: $(confusion[7])")
    println("Matrix:")
    display(confusion[8])
end
# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 5 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    # Paso 1: Crear un vector con k elementos ordenados
    subsets = collect(1:k)

    # Paso 2: Repetir el vector hasta que la longitud sea mayor o igual a N
    subsets = repeat(subsets, ceil(Int, N / k))

    # Paso 3: Tomar los N primeros valores
    subsets = subsets[1:N]

    # Paso 4: Desordenar el vector
    shuffle!(subsets)

    return subsets
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    # Paso 1: Obtener el número de instancias positivas y negativas
    num_positive = sum(targets)
    num_negative = length(targets) - num_positive

    # Paso 2: Partición estratificada para instancias positivas
    indices_positive = crossvalidation(num_positive, k)

    # Paso 3: Partición estratificada para instancias negativas
    indices_negative = crossvalidation(num_negative, k)

    # Paso 4: Crear vector de índices estratificado
    indices = fill(0, length(targets))
    indices[findall(targets)] .= indices_positive
    indices[.!targets] .= indices_negative

    return indices
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    # Paso 1: Obtener el número de filas en la matriz targets
    num_samples = size(targets, 1)
    size = size(targets,2)
    # Paso 2 y 3: Partición estratificada para cada clase
    indices = fill(0, num_samples)
    for class_idx in 1:size
        # Obtener el número de elementos que pertenecen a esta clase
        num_elements = sum(targets[:, class_idx])

        # Partición estratificada para esta clase
        class_indices = crossvalidation(num_elements, k)

        # Asignar los índices a las posiciones correspondientes en el vector de índices
        indices[findall(targets[:, class_idx]), class_idx] .= class_indices
    end

    # Paso 4: Devolver el vector de índices
    return indices
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    # Paso 1: Obtener el número total de muestras
    num_samples = length(targets)

    # Paso 2: Obtener las clases únicas
    unique_classes = unique(targets)

    # Paso 3: Crear un vector de índices
    indices = fill(0, num_samples)

    # Paso 4: Particionar estratificadamente cada clase
    for class in unique_classes
        # Obtener los índices de muestras pertenecientes a esta clase
        class_indices = findall(x -> x == class, targets)

        # Si el número de muestras en esta clase es menor que k, advertir y saltar a la siguiente clase
        if length(class_indices) < k
            println("Advertencia: Clase '$class' tiene menos de $k muestras.")
            continue
        end

        # Particionar estratificadamente las muestras de esta clase
        partitioned_indices = crossvalidation(length(class_indices), k)

        # Asignar los índices obtenidos a las muestras correspondientes en el vector de índices
        indices[class_indices] .= partitioned_indices
    end

    # Paso 5: Devolver el vector de índices
    return indices
end







# function ANNCrossValidation(topology::AbstractArray{<:Int,1},
#     inputs::AbstractArray{<:Real,2},
#     targets::AbstractArray{<:Any,1},
#     crossValidationIndices::Array{Int64,1};
#     numExecutions::Int=50,
#     transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
#     maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
#     validationRatio::Real=0, maxEpochsVal::Int=20)

#     # Convertir las salidas deseadas a one-hot-encoding
#     encoded_targets = oneHotEncoding(targets)

#     # Calcular el número de folds
#     num_folds = maximum(crossValidationIndices)

#     # Inicializar vectores para almacenar las métricas
#     precision = zeros(num_folds)
#     error_rate = zeros(num_folds)
#     sensitivity = zeros(num_folds)
#     specificity = zeros(num_folds)
#     VPP = zeros(num_folds)
#     VPN = zeros(num_folds)
#     F1 = zeros(num_folds)

#     # Iterar sobre cada fold
#     for fold in 1:num_folds
#         # Indices para el conjunto de entrenamiento y test
#         train_indices = findall(x -> x != fold, crossValidationIndices)
#         test_indices = findall(x -> x == fold, crossValidationIndices)

#         # Datos de entrenamiento y test
#         train_inputs = inputs[:, train_indices]
#         train_targets = encoded_targets[:, train_indices]
#         test_inputs = inputs[:, test_indices]
#         test_targets = encoded_targets[:, test_indices]

#         # Variables para almacenar los resultados de cada ejecución
#         metrics = zeros(numExecutions, 7)  # 7 métricas: precision, error_rate, sensitivity, specificity, VPP, VPN, F1

#         # Iterar sobre cada ejecución dentro del fold
#         for i in 1:numExecutions
#             # Dividir el conjunto de entrenamiento en entrenamiento y validación si es necesario
#             if validationRatio > 0
#                 train_inputs, train_targets, val_inputs, val_targets = holdOut(train_inputs, train_targets, validationRatio)
#             else
#                 val_inputs, val_targets = [], []
#             end

#             Entrenar la RNA y obtener las métricas
#             model = trainClassANN(topology=topology, (inputs=inputs, targets=targets),
#                 transferFunctions=transferFunctions,maxEpochs=maxEpochs, 
#                 minLoss=minLoss, learningRate=learningRate)

#             outputs = predictANN(model, test_inputs)
#             confusion_matrix = confusionMatrix(outputs, test_targets)
#             metrics[i, :] = [accuracy(confusion_matrix,targets), errorRate(confusion_matrix),
#                 sensitivity(confusion_matrix), specificity(confusion_matrix),
#                 positivePredictiveValue(confusion_matrix), negativePredictiveValue(confusion_matrix),
#                 F1Score(confusion_matrix)]
#         end

#         # Calcular la media y desviación estándar de las métricas para este fold
#         precision[fold], std_precision = mean_and_std(metrics[:, 1])
#         error_rate[fold], std_error_rate = mean_and_std(metrics[:, 2])
#         sensitivity[fold], std_sensitivity = mean_and_std(metrics[:, 3])
#         specificity[fold], std_specificity = mean_and_std(metrics[:, 4])
#         VPP[fold], std_VPP = mean_and_std(metrics[:, 5])
#         VPN[fold], std_VPN = mean_and_std(metrics[:, 6])
#         F1[fold], std_F1 = mean_and_std(metrics[:, 7])
#     end

#     # Devolver las métricas
#     return (precision=(precision, std_precision), error_rate=(error_rate, std_error_rate),
#         sensitivity=(sensitivity, std_sensitivity), specificity=(specificity, std_specificity),
#         VPP=(VPP, std_VPP), VPN=(VPN, std_VPN), F1=(F1, std_F1))
# end

# Función auxiliar para calcular la media y desviación estándar
function mean_and_std(values)
    return mean(values), std(values)
end


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 6 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

# using ScikitLearn: @sk_import, fit!, predict

# @sk_import svm: SVC
# @sk_import tree: DecisionTreeClassifier
# @sk_import neighbors: KNeighborsClassifier


# function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
#     #
#     # Codigo a desarrollar
#     #
# end;


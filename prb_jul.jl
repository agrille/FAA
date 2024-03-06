# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 1 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using FileIO;
using DelimitedFiles;
using Statistics;
using Flux
using Flux.Losses
using Random

#Función para codificar las salidas puesto que son categóricas.
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    @assert(numClasses > 1)
    if numClasses == 2
        # Si solo hay dos clases, se devuelve una matriz con una columna
        return reshape(feature .== classes[1], :, 1)
    else
        # Si hay más de dos clases, se crea una matriz de valores booleanos
        oneHot = falses(length(feature), numClasses)
        for (i, cls) in enumerate(classes)
            oneHot[:, i] .= (feature .== cls)
        end
        return oneHot
    end
    println("Tamaño de la matriz de salidas deseadas despues de codificar: ", size(targets, 1), "x", size(targets, 2), " de tipo ", typeof(targets))
end


oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)



# Funciones para calcular parámetros de normalización

# Función que calcula los mínimos y máximos de cada columna de la matriz dataset
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mins = minimum(dataset, dims=1)
    maxs = maximum(dataset, dims=1)
    return (mins, maxs)
end

# Función que calcula las medias y desviaciones estándar de cada columna de la matriz dataset
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    means = mean(dataset, dims=1)
    stds = std(dataset, dims=1)
    return (means, stds)
end

# Funciones para normalizar entre máximo y mínimo

# Función que normaliza la matriz dataset entre mínimo y máximo utilizando los parámetros de normalización dados
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    mins, maxs = normalizationParameters
    dataset .-= mins
    dataset ./= (maxs .- mins)
    return dataset
end

# Función que calcula los parámetros de normalización entre mínimo y máximo y normaliza la matriz dataset
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, normalizationParameters)
end

# Función que crea una copia de la matriz dataset, calcula los parámetros de normalización entre mínimo y máximo y normaliza la copia
function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    dataset_copy = copy(dataset)
    return normalizeMinMax!(dataset_copy, normalizationParameters)
end

# Función que crea una copia de la matriz dataset, calcula los parámetros de normalización entre mínimo y máximo y normaliza la copia
function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    dataset_copy = copy(dataset)
    return normalizeMinMax!(dataset_copy)
end

# Funciones para normalizar a media 0

# Función que normaliza la matriz dataset a media 0 utilizando los parámetros de normalización dados
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    means, stds = normalizationParameters
    dataset .-= means
    dataset ./= stds
    return dataset
end

# Función que calcula los parámetros de normalización a media 0 y normaliza la matriz dataset
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, normalizationParameters)
end

# Función que crea una copia de la matriz dataset, calcula los parámetros de normalización a media 0 y normaliza la copia
function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    dataset_copy = copy(dataset)
    return normalizeZeroMean!(dataset_copy, normalizationParameters)
end

# Función que crea una copia de la matriz dataset, calcula los parámetros de normalización a media 0 y normaliza la copia
function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    dataset_copy = copy(dataset)
    return normalizeZeroMean!(dataset_copy)
end

# Función classifyOutputs para un vector de salidas
function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    # Aplica un broadcast del operador >= para comparar cada valor con el umbral y genera un vector de valores booleanos
    return outputs .>= threshold
end

# Función classifyOutputs para una matriz de salidas
function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        # Si hay una sola columna, convierte la matriz en un vector y llama a la función classifyOutputs para vectores
        vector_outputs = reshape(outputs[:], :, 1)
        return classifyOutputs(vector_outputs, threshold=threshold)
    else
        # Si hay más de una columna, encuentra el índice del máximo en cada fila y crea una matriz booleana donde cada fila tiene un único valor verdadero en la posición del máximo
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        classified_outputs = falses(size(outputs))
        classified_outputs[indicesMaxEachInstance] .= true
        return classified_outputs
    end
end

# Función accuracy para vectores de salidas y objetivos booleanos
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # La precisión es simplemente el valor promedio de la comparación de ambos vectores
    return mean(outputs .== targets)
end

# Función accuracy para matrices de salidas y objetivos booleanos
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(outputs, 2) == 1
        # Si solo tienen una columna, se comparan las primeras columnas de outputs y targets
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        # Si tienen más de una columna, se compara si todas las clases coinciden para cada patrón
        correctClassifications = all(outputs .== targets, dims=2)
        # La precisión es el promedio de las clasificaciones correctas
        return mean(correctClassifications)
    end
end

# Función accuracy para vector de salidas reales y objetivos booleanos
function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Clasifica las salidas reales como booleanos utilizando el umbral dado
    outputs_bool = outputs .>= threshold
    # Llama a la función anterior para calcular la precisión
    return accuracy(outputs_bool, targets)
end

# Función accuracy para matriz de salidas reales y objetivos booleanos
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        # Si solo tienen una columna, se compara la primera columna de outputs con targets
        outputs_bool = classifyOutputs(outputs, threshold=threshold)
        return accuracy(outputs_bool, targets)
    else
        # Si tienen más de una columna, se clasifican las salidas reales como booleanos
        outputs_bool = classifyOutputs(outputs, threshold=threshold)
        # Luego se compara si todas las clases coinciden para cada patrón
        correctClassifications = all(outputs_bool .== targets, dims=2)
        # La precisión es el promedio de las clasificaciones correctas
        return mean(correctClassifications)
    end
end


function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    # Crea una RNA vacía
    ann = Chain()
    # Variable para llevar el número de entradas de la capa actual
    numInputsLayer = numInputs

    # Si hay capas ocultas
    if !isempty(topology)
        # Itera sobre cada capa oculta
        for numOutputsLayer in topology
            # Crea una capa oculta con el número de entradas y salidas correspondiente
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[numOutputsLayer]))
            # Actualiza el número de entradas para la siguiente capa
            numInputsLayer = numOutputsLayer
        end
    end

    # Añade la capa de salida con el número de salidas y la función de transferencia adecuada
    if numOutputs == 1
        # Si hay una sola clase de salida, usa una función sigmoide
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    else
        # Si hay más de una clase de salida, usa una función de activación softmax
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs), softmax)
    end

    return ann
end


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


function holdOut(N::Int, P::Real)
    # Obtener el número de patrones para el conjunto de test
    test_size = round(Int, N * P)

    # Obtener los índices de los patrones aleatoriamente
    indices = randperm(N)

    # Dividir los índices en dos conjuntos: entrenamiento y test
    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]

    return (train_indices, test_indices)
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    # Calcular el tamaño de los conjuntos de validación y test
    val_size = round(Int, N * Pval)
    test_size = round(Int, N * Ptest)

    # Calcular el tamaño del conjunto de entrenamiento
    train_size = N - val_size - test_size

    # Obtener los índices de los patrones utilizando holdOut para el conjunto de test
    train_indices, rest_indices = holdOut(N, Pval + Ptest)

    # Obtener los índices de los patrones para el conjunto de validación y test
    val_indices = rest_indices[1:val_size]
    test_indices = rest_indices[val_size+1:end]

    return (train_indices, val_indices, test_indices)
end


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

    # Paso 2 y 3: Partición estratificada para cada clase
    indices = fill(0, num_samples)
    for class_idx in 1:size(targets, 2):
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

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20)

    # Convertir las salidas deseadas a one-hot-encoding
    encoded_targets = oneHotEncoding(targets)

    # Calcular el número de folds
    num_folds = maximum(crossValidationIndices)

    # Inicializar vectores para almacenar las métricas
    precision = zeros(num_folds)
    error_rate = zeros(num_folds)
    sensitivity = zeros(num_folds)
    specificity = zeros(num_folds)
    VPP = zeros(num_folds)
    VPN = zeros(num_folds)
    F1 = zeros(num_folds)

    # Iterar sobre cada fold
    for fold in 1:num_folds
        # Indices para el conjunto de entrenamiento y test
        train_indices = findall(x -> x != fold, crossValidationIndices)
        test_indices = findall(x -> x == fold, crossValidationIndices)

        # Datos de entrenamiento y test
        train_inputs = inputs[:, train_indices]
        train_targets = encoded_targets[:, train_indices]
        test_inputs = inputs[:, test_indices]
        test_targets = encoded_targets[:, test_indices]

        # Variables para almacenar los resultados de cada ejecución
        metrics = zeros(numExecutions, 7)  # 7 métricas: precision, error_rate, sensitivity, specificity, VPP, VPN, F1

        # Iterar sobre cada ejecución dentro del fold
        for i in 1:numExecutions
            # Dividir el conjunto de entrenamiento en entrenamiento y validación si es necesario
            if validationRatio > 0
                train_inputs, train_targets, val_inputs, val_targets = holdOut(train_inputs, train_targets, validationRatio)
            else
                val_inputs, val_targets = [], []
            end

            # Entrenar la RNA y obtener las métricas
            model = trainClassANN(topology, train_inputs, train_targets,
                transferFunctions=transferFunctions,
                maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
                validationInputs=val_inputs, validationTargets=val_targets,
                maxEpochsVal=maxEpochsVal)

            outputs = predictANN(model, test_inputs)
            confusion_matrix = confusionMatrix(outputs, test_targets)
            metrics[i, :] = [accuracy(confusion_matrix), errorRate(confusion_matrix),
                sensitivity(confusion_matrix), specificity(confusion_matrix),
                positivePredictiveValue(confusion_matrix), negativePredictiveValue(confusion_matrix),
                F1Score(confusion_matrix)]
        end

        # Calcular la media y desviación estándar de las métricas para este fold
        precision[fold], std_precision = mean_and_std(metrics[:, 1])
        error_rate[fold], std_error_rate = mean_and_std(metrics[:, 2])
        sensitivity[fold], std_sensitivity = mean_and_std(metrics[:, 3])
        specificity[fold], std_specificity = mean_and_std(metrics[:, 4])
        VPP[fold], std_VPP = mean_and_std(metrics[:, 5])
        VPN[fold], std_VPN = mean_and_std(metrics[:, 6])
        F1[fold], std_F1 = mean_and_std(metrics[:, 7])
    end

    # Devolver las métricas
    return (precision=(precision, std_precision), error_rate=(error_rate, std_error_rate),
        sensitivity=(sensitivity, std_sensitivity), specificity=(specificity, std_specificity),
        VPP=(VPP, std_VPP), VPN=(VPN, std_VPN), F1=(F1, std_F1))
end

# Función auxiliar para calcular la media y desviación estándar
function mean_and_std(values)
    return mean(values), std(values)
end

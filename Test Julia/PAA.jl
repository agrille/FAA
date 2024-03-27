
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

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mean_values, std = normalizationParameters

    if size(dataset, 2) == size(mean_values, 2) == size(std, 2)
        dataset .-= mean_values
        dataset ./= std
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
function classifyOutputs(outputs::AbstractArray{<:Real, 1}; threshold::Real=0.5) 
    
    vec = outputs .>= threshold
    return vec
    
end
    



function classifyOutputs(outputs::AbstractArray{<:Real, 2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        # Si tiene una columna, convertir a vector y devolver el resultado directamente
        outputs_vec = vec(outputs)
        outputs = classifyOutputs(outputs_vec; threshold = threshold)
        return reshape(outputs, :, 1)
        

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
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        return mean(all(outputs .== targets, dims=2))
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
        predictions = classifyOutputs(outputs, threshold=threshold)
        return accuracy(predictions[:, 1], targets[:, 1])
    else
        # Si el número de columnas es mayor que 1, convertimos outputs a booleanos
        predictions = classifyOutputs(outputs)
        return accuracy(predictions, targets)
    end
end

# -------------------------------------------------------
# Funciones para crear y entrenar una RNA


# buildClassANN:

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
    if numOutputs == 1  
        # Problema de clasificación binaria
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        # Problema de clasificación multiclase
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs), softmax)
    end

    return ann
end


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20)
        # Extract input and target matrices from the training dataset
        inputs, targets = trainingDataset

        # Get the number of input and output neurons
        numInputs, numOutputs = size(inputs, 2), size(targets, 2)
    
        # Create the neural network with the provided topology
        ann = buildClassANN(numInputs, topology, numOutputs, transferFunctions=transferFunctions)
    
        # Vector to store loss values during training
        trainingLossValues = Float32[]
        validationLossValues = Float32[]
        testLossValues = Float32[]
    
        # Variables to store the best ANN and its best validation loss
        bestANN = deepcopy(ann)
        bestValidationLoss = Inf
    
        # Configure the optimizer
        opt = Flux.setup(Adam(learningRate), ann)
        
        # Define loss function based on the number of classes
        loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);        

        # Training the neural network
        for epoch in 1:maxEpochs
            # Train one epoch
            Flux.train!(loss, ann, [(inputs', targets')], opt)
    
            # Calculate loss value for training set
            trainingLoss = loss(ann, inputs', targets')
            push!(trainingLossValues, trainingLoss)
    
            # Calculate loss value for validation set if provided
            if !isempty(validationDataset[1])
                validationInputs, validationTargets = validationDataset
                validationLoss = loss(ann, validationInputs', validationTargets')
                push!(validationLossValues, validationLoss)
    
                # Update the best ANN if a new validation loss minimum is found
                if validationLoss < bestValidationLoss
                    bestValidationLoss = validationLoss
                    bestANN = deepcopy(ann)
                end
    
                # Early stopping criterion: if maxEpochsVal epochs pass without improving the best validation loss, stop training
                if epoch - argmin(validationLossValues) >= maxEpochsVal
                    break
                end
            end
    
            # Calculate loss value for test set if provided
            if !isempty(testDataset[1])
                testInputs, testTargets = testDataset
                testLoss = loss(ann, testInputs', testTargets')
                push!(testLossValues, testLoss)
            end
    
            # Stopping criterion: if loss is less than minLoss, stop training
            if trainingLoss < minLoss
                break
            end
        end
    
        # Select the final ANN to return (the best ANN if there is validation, the last trained ANN otherwise)
        finalANN = isempty(validationDataset[1]) ? ann : bestANN
    
        return finalANN, trainingLossValues, validationLossValues, testLossValues
    end


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    # Reshape de las salidas deseadas del conjunto de entrenamiento
    reshaped_training_targets = reshape(trainingDataset[2], :, 1)

    # Reshape de las salidas deseadas del conjunto de validación
    reshaped_validation_targets = reshape(validationDataset[2], :, 1)

    # Reshape de las salidas deseadas del conjunto de test
    reshaped_test_targets = reshape(testDataset[2], :, 1)

    # Llamar a la función original con las salidas deseadas reestructuradas
    return trainClassANN(topology, 
        (trainingDataset[1], reshaped_training_targets); 
        validationDataset=(validationDataset[1],reshaped_validation_targets),
        testDataset=(testDataset[1],reshaped_test_targets),
        transferFunctions = transferFunctions,
        maxEpochs = maxEpochs,
        minLoss = minLoss,
        learningRate = learningRate,
        maxEpochsVal = maxEpochsVal)
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
    test_indices,val_indices = holdOut(length(remaining_indices), Pval / (Pval + Ptest))

    # Ajustar los índices a los originales
    val_indices = remaining_indices[val_indices]
    test_indices = remaining_indices[test_indices]

    return train_indices, val_indices, test_indices
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 4 ---------------------------------------------
# ----------------------------------------------------------------------------------------------



function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "Los vectores deben tener la misma longitud"

    # Calcular los valores de la matriz de confusión
    VP = sum(outputs .& targets)
    VN = sum((.!outputs) .& (.!targets))
    FP = sum(outputs .& (.!targets))
    FN = sum((.!outputs) .& targets)

    # Calcular métricas
    accuracy = (VN + VP + FN + FP == 0) ? 0.0 : (VN + VP)/(VN + VP + FN + FP)#(VP + VN == 0) ? 1.0 : (VP / (VP + FP))
    error_rate = (VP + VN == 0) ? 0.0 : ((FN + FP) / (VP + VN + FN + FP))
    sensitivity = (VP == FN == 0) ? 1.0 : (VP / (FN + VP))
    specificity = (VN == FP == 0) ? 1.0 : (VN / (FP + VN))
    precision_pos = (VP == FP == 0) ? 1.0 : (VP / (VP + FP))
    precision_neg = (VN==FN==0) ? 1.0 : (VN / (VN + FN))
    f1_score = (sensitivity == precision_pos == 0) ? 0.0 : (2 * sensitivity * precision_pos / (sensitivity + precision_pos))

    # Crear la matriz de confusión
    confusion_matrix = [VN FP; FN VP]

    return accuracy, error_rate, sensitivity, specificity, precision_pos, precision_neg, f1_score, confusion_matrix
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Convertir las salidas a valores booleanos basados en el umbral
    outputs_bool = classifyOutputs(outputs,threshold=threshold)

    # Calcular las métricas de evaluación
    confusionMatrix(outputs_bool, targets)
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Verificar que ambas matrices tengan el mismo número de columnas y no sean matrices de dos columnas
    @assert size(outputs, 2) == size(targets, 2) != 2 "Las matrices deben tener el mismo número de columnas y no pueden tener solo dos columnas"

    num_classes = size(outputs, 2)

    # Inicializar vectores de métricas por clase
    sensitivity = zeros(Float64, num_classes)
    specificity = zeros(Float64, num_classes)
    precision_pos = zeros(Float64, num_classes)
    precision_neg = zeros(Float64, num_classes)
    f1_score = zeros(Float64, num_classes)

    # Inicializar matriz de confusión
    confusion_matrix = [confusionMatrix(outputs[:, i], targets[:, i]) for i in 1:num_classes]

    # Calcular métricas macro o weighted según se especifique
    for i in 1:num_classes
        sensitivity[i], _, _, specificity[i], precision_pos[i], precision_neg[i], f1_score[i], _ = confusion_matrix[i]
    end

    if weighted
        weights = sum(targets, dims=1) / size(targets, 1)
        sensitivity = sum(sensitivity .* weights)
        specificity = sum(specificity .* weights)
        precision_pos = sum(precision_pos .* weights)
        precision_neg = sum(precision_neg .* weights)
        f1_score = sum(f1_score .* weights)
    else
        sensitivity = mean(sensitivity)
        specificity = mean(specificity)
        precision_pos = mean(precision_pos)
        precision_neg = mean(precision_neg)
        f1_score = mean(f1_score)
    end

    # Calcular la precisión y la tasa de error
    accuracy_value = accuracy(outputs, targets, threshold=0.5)
    error_rate = 1.0 - accuracy_value

    return (accuracy_value, error_rate, sensitivity, specificity, precision_pos, precision_neg, f1_score, confusion_matrix)
end


function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Convertir las salidas del modelo a valores booleanos
    outputs_bool = classifyOutputs(outputs)

    # Llamar a la función confusionMatrix con los nuevos parámetros
    return confusionMatrix(outputs_bool, targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Asegurar que todas las clases de salida estén incluidas en las clases deseadas
    @assert all(in.(unique(outputs), unique(targets))) "Todas las clases de salida deben estar incluidas en las clases deseadas"

    # Codificar las matrices outputs y targets
    encoded_outputs = oneHotEncoding(outputs)
    encoded_targets = oneHotEncoding(targets)

    # Llamar a la función confusionMatrix con las matrices codificadas
    return confusionMatrix(encoded_outputs, encoded_targets; weighted=weighted)
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
    print("VN: ",confusion[1]," ")
    println("FP: ",confusion[2]," ")
    print("FN: ",confusion[3]," ")
    println("VP: ",confusion[4]," ")
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
    print("VN: ",confusion[1]," ")
    println("FP: ",confusion[2]," ")
    print("FN: ",confusion[3]," ")
    println("VP: ",confusion[4]," ")
end

function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Calcular la matriz de confusión
    confusion = confusionMatrix(outputs, targets; weighted=weighted)

    # Mostrar resultados
    println("Confusion Matrix:")
    println("Accuracy: $(confusion[1])")
    println("Error Rate: $(confusion[2])")
    println("Sensitivity (Recall): $(confusion[3])")
    println("Specificity: $(confusion[4])")
    println("Positive Predictive Value: $(confusion[5])")
    println("Negative Predictive Value: $(confusion[6])")
    println("F1 Score: $(confusion[7])")
    println("Matrix:")
    println("Confusion Matrix:")
    display(confusion[8])
    

    
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Calcular la matriz de confusión
    confusion = confusionMatrix(outputs, targets; weighted=weighted)

    # Mostrar resultados
    println("Confusion Matrix:")
    println("Accuracy: $(confusion[1])")
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
    println("Accuracy: $(confusion[1])")
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
    # Obtener el número total de patrones
    num_samples = size(targets, 1)

    # Crear un vector de índices con tantos valores como filas en la matriz targets
    indices = collect(1:num_samples)

    # Hacer un bucle sobre las clases y asignar valores al vector de índices
    for class_column in eachcol(targets)
        # Obtener el número de elementos que pertenecen a esa clase
        num_elements = sum(class_column)

        # Llamar a la función crossvalidation para obtener los índices de partición
        partition_indices = crossvalidation(num_elements, k)

        # Asignar los valores del vector resultado al vector de índices
        indices[class_column] .= partition_indices
    end

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

    # Función auxiliar para calcular la media y desviación estándar
    function mean_and_std(values)
        return mean(values), std(values)
    end

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
            model = trainClassANN(topology, (inputs, targets);
                transferFunctions=transferFunctions,maxEpochs=maxEpochs, 
                minLoss=minLoss, learningRate=learningRate)

            outputs = predictANN(model, test_inputs)
            confusion_matrix = confusionMatrix(outputs, test_targets)
            metrics[i, :] = [accuracy(confusion_matrix,targets), errorRate(confusion_matrix),
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


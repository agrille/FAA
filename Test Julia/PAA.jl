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
        # Si solo tienen una columna, llamamos a la función anterior
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        
        total_rows = size(outputs, 1)
        correct_count = sum(all(outputs .== targets, dims=2))
        
        return correct_count / total_rows
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

    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);        

    numInputs, numOutputs = size(inputs, 2), size(targets, 2)

    

    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)

    # Var de parada temprana
   unimprovedCicles= Int
   unimprovedCicles= 0

    # Configure the optimizer
    opt = Flux.setup(Adam(learningRate), ann)
    loss_0 = loss(ann, inputs', targets')
    trainingLossValues = Float32[] 
    push!(trainingLossValues,loss_0)

    if !isempty(validationDataset[1])
        validationInputs, validationTargets = validationDataset
        vLoss_0 = loss(ann, validationInputs', validationTargets')
        validationLossValues = Float32[]
        push!(validationLossValues,vLoss_0)

    else 
        validationLossValues = Float32[]
    end

    if !isempty(testDataset[1])
        testInputs, testTargets = testDataset
        tLoss_0 = loss(ann, testInputs', testTargets')
        testLossValues = Float32[]
        push!(validationLossValues,tLoss_0)
    else
        testLossValues = Float32[]
    end

    # Variables to store the best ANN and its best validation loss
    bestANN = deepcopy(ann)
    bestValidationLoss = Inf

    
    

    # Training the neural network
    for epoch in 1:maxEpochs
        # Train one epoch
        Flux.train!(loss, ann, [(inputs', targets')], opt)

        # Calculate loss value for training set
        trainingLoss = loss(ann, inputs', targets')
        push!(trainingLossValues, trainingLoss)

        # Calculate loss value for validation set if provided
        if !isempty(validationDataset[1])
            # Flux.train!(loss, ann, [(validationInputs', validationTargets')], opt)
            validationLoss = loss(ann, validationInputs', validationTargets')
            push!(validationLossValues, validationLoss)

            # Update the best ANN if a new validation loss minimum is found
            if validationLoss < bestValidationLoss
                bestValidationLoss = validationLoss
                bestANN = deepcopy(ann)
                unimprovedCicles=0
                
            else 
               unimprovedCicles+= 1
            end
            # print(unimprovedCicles)
            # Evitar Sobreentrenamiento
            if unimprovedCicles >= maxEpochsVal
                # print(n)
                break
            end
        end

        # Calculate loss value for test set if provided
        if !isempty(testDataset[1])
            # Flux.train!(loss, ann, [(testInputs', testTargets')], opt)
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
    accuracy = (VN + VP + FN + FP == 0) ? 0.0 : (VN + VP)/(VN + VP + FN + FP)
    error_rate = (VN + VP + FN + FP == 0) ? 1.0 : ((FN + FP) / (VP + VN + FN + FP))
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
    if size(outputs,2) == 1 && size(targets,2 ) == 1
        return confusionMatrix(vec(outputs),vec(targets))
    # Inicializar vectores de métricas por clase
    else 
        accuracy_value = accuracy(outputs, targets, threshold=0.5)
        error_rate = 1.0 - accuracy_value
        sensitivity = zeros(Float32, num_classes)
        specificity = zeros(Float32, num_classes)
        precision_pos = zeros(Float32, num_classes)
        precision_neg = zeros(Float32, num_classes)
        f1_score = zeros(Float32, num_classes)
        matrix = zeros(Int,num_classes,num_classes)
        total = size(outputs,1)

        for i in 1:total
            true_label = findfirst(targets[i, :])
            predicted_label = findfirst(outputs[i, :])
            matrix[true_label, predicted_label] += 1
        end

        # Calcular métricas macro o weighted según se especifique
        for i in 1:num_classes
            _, _,sensitivity[i], specificity[i], precision_pos[i], precision_neg[i], f1_score[i], _ = confusionMatrix(outputs[:,i], targets[:,i])
        end

        if weighted
            weights = vec(sum(targets, dims=1))/ size(targets, 1)
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

    end

    return (accuracy_value, error_rate, sensitivity, specificity, precision_pos, precision_neg, f1_score, matrix)
end




function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Convertir las salidas del modelo a valores booleanos
    outputs_bool = classifyOutputs(outputs)

    # Llamar a la función confusionMatrix con los nuevos parámetros
    return confusionMatrix(outputs_bool, targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Asegurar que todas las clases de salida estén incluidas en las clases deseadas
    @assert(all([in(output, unique(targets)) for output in outputs])) 

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
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20)

    # Crear vectores para almacenar los resultados de cada métrica
    precision_results = Float32[]
    error_rate_results = Float32[]
    sensitivity_results = Float32[]
    specificity_results = Float32[]
    vpp_results = Float32[]
    vpn_results = Float32[]
    f1_results = Float32[]

    # Calcular el número de folds
    num_folds = maximum(crossValidationIndices)

    # Iterar sobre cada fold de validación cruzada
    for fold in 1:num_folds
        # Crear vectores para almacenar los resultados de cada repetición
        precision_fold = Float32[]
        error_rate_fold = Float32[]
        sensitivity_fold = Float32[]
        specificity_fold = Float32[]
        vpp_fold = Float32[]
        vpn_fold = Float32[]
        f1_fold = Float32[]

        # Obtener los índices de entrenamiento y test para este fold
        test_indices = findall(crossValidationIndices .== fold)
        train_indices = setdiff(1:length(crossValidationIndices), test_indices)

        # Convertir targets a matriz de valores booleanos mediante one-hot-encoding
        encoded_targets = oneHotEncoding(targets)
        # print(encoded_targets)
        # Dividir el conjunto de entrenamiento en entrenamiento y validación si validationRatio > 0
        if validationRatio > 0
            train_indices, val_indices = holdOut(length(train_indices), validationRatio)
            validation_inputs = inputs[val_indices, :]
            validation_targets = encoded_targets[val_indices, :]
            # print(validation_targets)
        else
            validation_inputs = Array{eltype(inputs)}(undef, 0, size(inputs, 2))
            validation_targets = Array{eltype(encoded_targets)}(undef, 0, size(encoded_targets, 2))
        end

        # Iterar sobre cada ejecución dentro del fold
        for _ in 1:numExecutions
            # Entrenar la RNA
            trained_ann, _, _, _ = trainClassANN(topology, (inputs[train_indices, :], encoded_targets[train_indices, :]);
                validationDataset=(validation_inputs, validation_targets),
                maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
                maxEpochsVal=maxEpochsVal, transferFunctions=transferFunctions)

            # Evaluar el rendimiento en el conjunto de test
            predictions = reshape(classifyOutputs(trained_ann(inputs')),:,size(encoded_targets,2))
            # print(predictions)
            # print(encoded_targets)
            confusion_matrix = confusionMatrix(predictions[test_indices, :], encoded_targets[test_indices, :])
            precision, error_rate, sensitivity, specificity, vpp, vpn, f1, _ = confusion_matrix

            # Almacenar los resultados de esta repetición
            push!(precision_fold, precision)
            push!(error_rate_fold, error_rate)
            push!(sensitivity_fold, sensitivity)
            push!(specificity_fold, specificity)
            push!(vpp_fold, vpp)
            push!(vpn_fold, vpn)
            push!(f1_fold, f1)
        end

        # Calcular la media y desviación estándar de las métricas para este fold
        mean_precision = mean(precision_fold)
        std_precision = std(precision_fold)
        mean_error_rate = mean(error_rate_fold)
        std_error_rate = std(error_rate_fold)
        mean_sensitivity = mean(sensitivity_fold)
        std_sensitivity = std(sensitivity_fold)
        mean_specificity = mean(specificity_fold)
        std_specificity = std(specificity_fold)
        mean_vpp = mean(vpp_fold)
        std_vpp = std(vpp_fold)
        mean_vpn = mean(vpn_fold)
        std_vpn = std(vpn_fold)
        mean_f1 = mean(f1_fold)
        std_f1 = std(f1_fold)

        # Almacenar los resultados de este fold
        push!(precision_results, mean_precision)
        push!(precision_results, std_precision)
        push!(error_rate_results, mean_error_rate)
        push!(error_rate_results, std_error_rate)
        push!(sensitivity_results, mean_sensitivity)
        push!(sensitivity_results, std_sensitivity)
        push!(specificity_results, mean_specificity)
        push!(specificity_results, std_specificity)
        push!(vpp_results, mean_vpp)
        push!(vpp_results, std_vpp)
        push!(vpn_results, mean_vpn)
        push!(vpn_results, std_vpn)
        push!(f1_results, mean_f1)
        push!(f1_results, std_f1)

    end

    # Devolver los resultados
    return (precision_results, error_rate_results, sensitivity_results,
            specificity_results, vpp_results, vpn_results, f1_results)
end


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 6 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using ScikitLearn: @sk_import, fit!, predict
using Random: seed!

# Importar los modelos necesarios
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    # Preprocesar los datos de destino si es necesario (convertir a cadena)
    targets = string.(targets)
    
    # Variables para almacenar resultados de métricas
    num_folds = length(crossValidationIndices)
    precision = zeros(num_folds)
    error_rate = zeros(num_folds)
    sensitivity = zeros(num_folds)
    specificity = zeros(num_folds)
    vpp = zeros(num_folds)
    vpn = zeros(num_folds)
    f1 = zeros(num_folds)
    
    # Verificar el tipo de modelo deseado
    if modelType == :ANN
        # Implementar lógica para entrenar redes neuronales (no incluido aquí)
        # Aquí se podría llamar a una función existente para entrenar redes neuronales
        println("Entrenamiento de redes neuronales aún no implementado")
        return
    elseif modelType == :SVC
        # Crear modelo SVM
        model = SVC(C=modelHyperparameters["C"],
                    kernel=modelHyperparameters["kernel"],
                    degree=modelHyperparameters["degree"],
                    gamma=modelHyperparameters["gamma"],
                    coef0=modelHyperparameters["coef0"])
    elseif modelType == :DecisionTreeClassifier
        # Crear modelo de Árbol de Decisión
        model = DecisionTreeClassifier(max_depth=modelHyperparameters["max_depth"])
    elseif modelType == :KNeighborsClassifier
        # Crear modelo kNN
        model = KNeighborsClassifier(n_neighbors=modelHyperparameters["n_neighbors"])
    else
        println("Tipo de modelo no reconocido")
        return
    end
    
    # Iterar sobre las particiones de validación cruzada
    for i in 1:num_folds
        # Obtener índices de entrenamiento y prueba
        train_indices = setdiff(collect(1:size(inputs, 1)), crossValidationIndices[i])
        test_indices = crossValidationIndices[i]
        
        # Obtener datos de entrenamiento y prueba
        train_inputs = inputs[train_indices, :]
        train_targets = targets[train_indices]
        test_inputs = inputs[test_indices, :]
        test_targets = targets[test_indices]
        
        # Entrenar el modelo
        fit!(model, train_inputs, train_targets)
        
        # Realizar predicciones
        predictions = predict(model, test_inputs)
        
        # Calcular matriz de confusión
        cm = confusionMatrix(predictions, test_targets)
        
        # Calcular métricas de desempeño
        precision[i] = cm[1][1]
        error_rate[i] = cm[2][1]
        sensitivity[i] = cm[3][1]
        specificity[i] = cm[4][1]
        vpp[i] = cm[5][1]
        vpn[i] = cm[6][1]
        f1[i] = cm[7][1]
    end
    
    # Devolver resultados de métricas
    return (mean(precision), std(precision)), 
           (mean(error_rate), std(error_rate)), 
           (mean(sensitivity), std(sensitivity)), 
           (mean(specificity), std(specificity)), 
           (mean(vpp), std(vpp)), 
           (mean(vpn), std(vpn)), 
           (mean(f1), std(f1))
end


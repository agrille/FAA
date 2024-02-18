# Package Install
begin
    import Pkg;
    Pkg.add("XLSX");
    Pkg.add("FileIO");
    Pkg.add("JLD2");
    Pkg.add("Flux");
    Pkg.add("ScikitLearn");
    Pkg.add("Plots");
    Pkg.add("MAT");
    Pkg.add("Images");
    Pkg.add("DelimitedFiles");
    Pkg.add("CSV");
    Pkg.update()
end


using FileIO;
using DelimitedFiles;
using Statistics;
using Flux
using Flux.Losses
using Random
using Random:seed!
using ScikitLearn
@sk_import svm: SVC     
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

# Read Iris Dataset
dataset = readdlm("Iris.data",',');
inputs = dataset[:,1:4];
targets = dataset[:,5];

print(typeof(dataset))
print(typeof(inputs))
print(typeof(targets))

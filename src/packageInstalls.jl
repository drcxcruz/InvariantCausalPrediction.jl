using Pkg;

##Install main packages
Pkg.add([
    "MLJ",
    "Query",
    "VegaLite",
    "CategoricalArrays",
    "DataFrames",
    "Combinatorics",
    "HypothesisTests",
    "CSV",
    "PrettyPrinting",
    "Missings",
    "StatsBase",
    "Tables",
    "Hwloc",
    "ShapML",
    "PkgTemplates",
    "FreqTables",
    "Distributions",
    "IJulia"
])
Pkg.status()

##Install all MLJ relevant packages 
Pkg.add([
    "Clustering",
    "DecisionTree",
    "EvoTrees",
    "GLM",
    "LightGBM",
    "LIBSVM",
    "MLJModels",
    "MLJLinearModels",
    "XGBoost",
    "ParallelKMeans"
])
Pkg.status()

#  add MLJ@0.12
#  using Pkg; Pkg.add(PackageSpec(url = "https://github.com/nredell/ShapML.jl"))
#  C:\Users\BCP\.julia\environments\v1.4

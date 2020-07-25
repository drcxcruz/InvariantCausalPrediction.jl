# Invariant Causal Prediction

Here is a Julia 1.4.2 translation of the core functionality of R lanague packages
- https://cran.r-project.org/web/packages/InvariantCausalPrediction/InvariantCausalPrediction.pdf
- https://cran.r-project.org/web/packages/nonlinearICP/nonlinearICP.pdf


Two Jupyter labs showcase the R language version of ICP. 
1.  [Linear Invariant Causal Prediction Using Employment Data From The Work Bank](https://notes.quantecon.org/submission/5e851bfecc00b7001acde469)
2.  [Nonlinear Invariant Causal Prediction Using Unemployment Data and Inflation Adjusted Prices from the USA Bureau of Labor](https://notes.quantecon.org/submission/5e8e2a6cd079ab001915ca09)

The core and minimal functionality of the original Invariant Causal Prediction (ICP) R packages is implemented in [Julia](https://julialang.org) 1.4.2. There are improvements over the R programming, however. The Julia version makes it easier to define **Y** and **E** input arguments with an integer mapper. There are refinements for code readability and support ability. There are new [VegaLite](https://www.queryverse.org/VegaLite.jl/stable/)  plots of the Invariant Causal Prediction results.  The output of the ICP Julia functions is more informative.

Furthermore, there are enhancements to program speed such parallelism of random forest computations, and linear p-value computations. There are two versions of the ICP main functions. One version is sequential and the default is parallel. The Julia [pmap](https://docs.julialang.org/en/v1/manual/parallel-computing/) function and Julia [@spawnat](https://docs.julialang.org/en/v1/manual/parallel-computing/) macro implement the parallelism.  

The [MLJ](https://alan-turing-institute.github.io/MLJTutorials) framework is heavily utilized to implement the machine learning in the ICP Julia version. MLJ models are in Julia and are faster than the old R language machine learning algorithms.  The [Alan Turing Institute](https://www.turing.ac.uk) sponsors the package and MLJ supports most of the machine learning models in Julia.  MLJ "offers a consistent way to use, compose and tune machine learning models in Julia.  ... MLJ unlocks performance gains by exploiting Julia's support for parallelism, automatic differentiation, GPU, optimisation etc."  The Julia ICP functions execute the MLJ models many times in parallel.   This allows to process larger files than in the R language version of ICP.  Next are the MLJ models that are run in parallel.  
'''
    @everywhere rfc = MLJ.@load RandomForestClassifier  pkg = DecisionTree
    @everywhere rfr = MLJ.@load RandomForestRegressor   pkg = DecisionTree

    @everywhere @load LinearRegressor pkg = GLM
    @everywhere @load LinearBinaryClassifier pkg = GLM 
'''
A better booster is used to reduce the number of predictors before running the parallel MLJ models.  [XGBoostRegressor](https://github.com/dmlc/XGBoost.jl) is a scalable, portable and Distributed gradient boosting framework.  The XGBooster is highly recommended when running linear ICP to reduce speed and memory usage.  A sophisticated (and slow) stochastic feature-level Shapley algorithm named [ShapML](https://github.com/nredell/ShapML.jl) is used to reduce the number of predictors before running the nonlinear models in parallel. The nonlinear models are random forests.  Incredible, ShapML does not directly read a dataset but it takes as input a MLJ model which encompasses the dataset.  ShapML is implemented in Julia.  
'''
    xgc = @load XGBoostClassifier;
    xgr = @load XGBoostRegressor;

    using ShapML
'''

## Install
'''
using Pkg
Pkg.add("InvariantCausalPrediction")
'''

## Documentation 

"In practice, first apply ICP with linear models.  Apply a nonlinear version if all linear models are rejected by linear IPC."

Jupyter lab [Invariant Causal Prediction in Julia:  How Some Are Able Earn More Than 50 Thousands a Year?](https://notes.quantecon.org/submission/)  showcases the Julia 1.4.2 version of IPC.  

## Example
'''
using InvariantCausalPrediction, Queryverse, DataFrames, CSV

# OpenML file:  [salary.csv](https://www.openml.org/d/179)
dfSalary = CSV.File("Salary.csv") |> DataFrame

qfLatinSalary = dfSalary |>     
    @filter( occursin(r"Peru|Mexico|Dominican-Republic|Haiti|El-Salvador|Puerto-Rico|Columbia|Cuba|Nicaragua|Honduras|Ecuador|Jamaica", _.native_country)) |> 
    @orderby(_.native_country)

dfLatinSalary = qfLatinSalary |> DataFrame
select!(dfLatinSalary, Not([:education, :fnlwgt]))  # using education_num
'''
'''
X = select(dfLatinSalary, Not([:class, :native_country]))       
Y = select(dfLatinSalary, [:class]) 
E = select(dfLatinSalary, [:native_country]) 

rLatin = LinearInvariantCausalPrediction!(X, Y, E, α = 0.10)
'''



module GetBlankets

using MLJ, DataFrames, ShapML
using StatsBase: sample
using CategoricalArrays: CategoricalArray
using Combinatorics: powerset
using Tables
xgc = @load XGBoostClassifier;
xgr = @load XGBoostRegressor;
rfc = MLJ.@load RandomForestClassifier  pkg = DecisionTree
rfr = MLJ.@load RandomForestRegressor   pkg = DecisionTree
include("ICPlibrary.jl")

export getBlanketAll, getBlanketBoosting, getBlanketRandomForest

function getBlanketAll(X::DataFrame)
    if ncol(X) >= 2
        s = collect(powerset(1:ncol(X)))
        return (s[2:end, 1])
    elseif ncol(X) == 1
        return ([[1]])
    else
        return ([[]])
    end
end # getBlanketAll


function getBlanketBoosting(
    X::DataFrame,
    target::DataFrame,
    selection::String="booster",
    maxNoVariables::Int64=10, 
)
    if selection == "all" || ncol(X) <= maxNoVariables
        return ((blankets = getBlanketAll(X), featureImportances = missing))
    else
        indexFromPredictor = getIndexFromPredictor(X)
        (usingVariables, featureImportances) = selectXGBooster(X, target, maxNoVariables)

        if length(usingVariables) >= 1
            usingVariablesIndex = [indexFromPredictor[n] for n in usingVariables]
            s = sort(usingVariablesIndex)
            s = collect(powerset(s))
            return ((blankets = s[2:end, 1], featureImportances = featureImportances))
        else
            return ((blankets = [[]], featureImportances = missing))
        end
    end
end # getBlanketBoosting


function selectXGBooster(
    X::DataFrame,
    target::DataFrame,
    maxNoVariables::Int64,
)
    selectionVec = []
    featureImportances = trainXGBooster(X, target)
    println(featureImportances)

    gainSum = 0
    variableCount = 0
    for f in featureImportances
        gainSum += f.gain
        variableCount += 1
        push!(selectionVec, f.fname)
        if variableCount >= maxNoVariables || variableCount >= length(featureImportances) || gainSum >= 0.90 
            break
        end
    end

    println("XGBooster selection:  ", selectionVec)
    GC.gc()
    return ((selectionVec = selectionVec, featureImportances = featureImportances))
end # selectXGBooster


function trainXGBooster(X::DataFrame, y::DataFrame)
    GC.gc()
    if length(unique(y[:, 1])) > 2
        pipeXGBoostRegressor = @pipeline XGBoostRegressor 
        r1 = range(pipeXGBoostRegressor, :(xg_boost_regressor.max_depth), lower=3, upper=10)
        r2 = range(pipeXGBoostRegressor, :(xg_boost_regressor.num_round), lower=1, upper=25)

        tmXGBoostRegressor = TunedModel(
            model=pipeXGBoostRegressor,
            tuning=Grid(resolution=7),
            resampling=CV(rng=11),
            ranges=[r1, r2],
            measure=rms,
        )
        mtm = machine(
            tmXGBoostRegressor,
            setScientificTypes!(X),
            Float64.(y[:, 1]),
        )
        fit!(mtm)
        k = collect(keys(report(mtm).best_report.report_given_machine))[1]
        return (report(mtm).best_report.report_given_machine[k][1])
    else
        pipeXGBoostClassifier = @pipeline XGBoostClassifier 
        r1 = range(pipeXGBoostClassifier, :(xg_boost_classifier.max_depth), lower=3, upper=10)
        r2 = range(pipeXGBoostClassifier, :(xg_boost_classifier.num_round), lower=1, upper=25)

        tmXGBoostClassifier = TunedModel(
            model=pipeXGBoostClassifier,
            tuning=Grid(resolution=7),
            resampling=CV(rng=11),
            ranges=[r1, r2],
            measure=cross_entropy, # don't use rms for probabilistic responses
        )
        mtm = machine(
            tmXGBoostClassifier,
            setScientificTypes!(X),
            categorical(y[:, 1]),
        )
        fit!(mtm)
        k = collect(keys(report(mtm).best_report.report_given_machine))[1]
        return (report(mtm).best_report.report_given_machine[k][1])
    end
end # trainXGBooster


#####################################################################################################################################################
function getBlanketRandomForest(
    X::DataFrame,
    target::DataFrame,
    selection::String="forest",
    maxNoVariables::Int64=10,
)
    if selection == "all" 
        return ((blankets = getBlanketAll(X), featureImportances = missing))
    elseif selection == "booster"
        return (getBlanketBoosting(X, target, selection, maxNoVariables))
    else
        indexFromPredictor = getIndexFromPredictor(X)
        (usingVariables, featureImportances) = selectRandomForest(X, target, maxNoVariables)
        if length(usingVariables) >= 1
            usingVariablesIndex = [indexFromPredictor[String(n)] for n in usingVariables]
            s = sort(usingVariablesIndex)
            s = collect(powerset(s))
            return ((blankets = s[2:end, 1], featureImportances = featureImportances))
        else
            return ((blankets = [[]], featureImportances = missing))
        end
    end
end # getBlanketRandomForest


function selectRandomForest(
    X::DataFrame,
    target::DataFrame,
    maxNoVariables::Int64=10,
)
    selectionVec = []
    featureImportances = trainRandomForest(X, target)

    variableCount = 0
    meanEffectPercentSum = 0
    for f in featureImportances
        meanEffectPercentSum += f.meanEffectPercent
        variableCount += 1
        push!(selectionVec, f.feature_name)
        if variableCount >= maxNoVariables || variableCount >= length(featureImportances) || f.meanEffectPercent < 0.05 || meanEffectPercentSum >= 0.90 
            break
        end
    end

    println("Random Forest with Shapley selection:  ", selectionVec)
    GC.gc()
    return ((selectionVec = selectionVec, featureImportances = featureImportances))
end # selectRandomForest


function predict_function(model, data)
    data_pred = DataFrame(y_pred=predict(model, data))
    return data_pred
end # predict_function


function predict_function_mode(model, data)
    ŷ = MLJ.predict(model, data)
    ŷMode = [convert(Int64, mode(ŷ[i])) for i in 1:length(ŷ)]
    data_pred = DataFrame(y_pred=ŷMode)
    return data_pred
end # predict_function_mode


function trainRandomForest(
    X::DataFrame, 
    y::DataFrame
)
    GC.gc()
    if length(unique(y[:, 1])) > 2
        pipeRandomForestRegressor = @pipeline FeatureSelector RandomForestRegressor
        cases = [[Symbol(names(X)[j]) for j in 1:i] for i in 1:ncol(X)]     
        r1 = range(pipeRandomForestRegressor, :(feature_selector.features), values=cases)
        tmRandomForestRegressor = TunedModel(
            model=pipeRandomForestRegressor,
            range=r1,
            measures=rms,
            resampling=CV(nfolds=5)
        )
        mtm = machine(tmRandomForestRegressor, setScientificTypes!(X), Float64.(y[:, 1]))
        Base.invokelatest(MLJ.fit!, mtm)
    
        predictor = predict_function
    else
        pipeRandomForestClassifier = @pipeline FeatureSelector RandomForestClassifier #prediction_type = :probabilistic
        cases = [[Symbol(names(X)[j]) for j in 1:i] for i in 1:ncol(X)]   
        r1 = range(pipeRandomForestClassifier, :(feature_selector.features), values=cases)
        tmRandomForestClassifier = TunedModel(
            model=pipeRandomForestClassifier,
            range=r1,
            measures=[cross_entropy, BrierScore()],
            resampling=CV(nfolds=5)
        )
        mtm = machine(tmRandomForestClassifier, setScientificTypes!(X), categorical(y[:, 1]))
        Base.invokelatest(MLJ.fit!, mtm)

        predictor = predict_function_mode
    end

    dfShapMeanEffect = Shapley(X, mtm, predictor)
    println("Shapley Effect of Random Forest\n", dfShapMeanEffect, "\n")
    return (Tables.rowtable(dfShapMeanEffect))
end # trainRandomForest


function Shapley(
    X::DataFrame,
    mtm,
    predictor
)
    dataShap = Missing
    sample_size = 60            # Number of Monte Carlo samples for Shapley, must be > 30 to apply the Central Limit Theorem
    if ncol(X) > 12
    # ShapML is slow and runs out of memory on larger datasets
        if nrow(X) > 5_000
            lDivisors = [3,4,6,8,10,12]
        else    
            lDivisors = [2,4,6,8,10,12]
        end
    else 
        lDivisors = [1,2,3,4,6,8,10]
    end 
    for d in lDivisors
        GC.gc()
        r = sample(1:nrow(X), Int(round(nrow(X) / d)), replace=false)
        explain = copy(X[r, :])     # Compute Shapley feature-level predictions 
        reference = copy(X)         # An optional reference population to compute the baseline prediction.
        try
            println("Computing Shapley Effect of Random Forest using ", length(r), " random rows.  It might take a few minutes")
            dataShap = ShapML.shap( explain=explain,
                            reference=reference,
                            model=mtm,
                            predict_function=predictor,
                            sample_size=sample_size,
                            seed=20200629
            )
            GC.gc()
            break
        catch
            continue
        end
    end
    dfShapMean = DataFrames.by(dataShap, [:feature_name], mean_effect=[:shap_effect] => x -> mean(abs.(x.shap_effect)))
    dfShapMeanEffect = sort(dfShapMean, order(:mean_effect, rev=true))
    totalEffect = sum(dfShapMeanEffect.mean_effect)
    dfShapMeanEffect.meanEffectPercent = dfShapMeanEffect.mean_effect / totalEffect
    return dfShapMeanEffect
end # Shapley

end # model 

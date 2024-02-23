# ----- Mean Fields -----
"""
    Mean Field
"""
struct MeanFields{alg<:AbstractMeanIsingAlgorithm} <: AbstractMeanField
    fields::Vector{<:AbstractField}
    upd_alg::AbstractUpdateIsingModel
    avg_alg::alg
end

function MeanFields(fields::Vector{AbstractField}, upd_alg::AbstractUpdateIsingModel,
    avg_alg::alg) where {alg<:AbstractMeanIsingAlgorithm}
    MeanFields{alg}(fields, upd_alg, avg_alg)
end



# ----- Mean Fields Algorithms -----

"""
    For taking temporal Averages
"""
struct TemporalAverageAlgorithm <: AbstractTemporalAverageAlgorithm 
    # total time
    tt::Int
    # transient time
    tr::Int
    # sampling frequency
    smp::Int
end

function TemporalAverageAlgorithm(tt::Int)
    tr=0
    smp=1
    TemporalAverageAlgorithm(tt, tr, smp)
end

"""
    calculate fields average

    Inverse temperature
        Beta::T
    Ising Model
        ising::AbstractIsingModel
    Mean Fields to find
        meanFields:: MeanFields
"""
function calculateFieldsAverage!(ising::AbstractIsingModel, Beta::T, 
    meanFields::MeanFields{TemporalAverageAlgorithm}, rng::AbstractRNG) where {T}
    avg_alg = meanFields.avg_alg
    upd_alg = meanFields.upd_alg
    fields = meanFields.fields

    cont = 0
    for i in 1:avg_alg.tt
        updateIsingModel!(ising, Beta, upd_alg, rng)
        if i>avg_alg.tr & mod(i,avg_alg.smp)==0
            for field in fields
                field.field .+= calcField(ising, Beta, field, upd_alg)
                cont += 1
            end
        end
    end
    if cont > 0
        for field in fields
            field.field ./= cont
        end
    end
end


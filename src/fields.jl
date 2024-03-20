# ----- Instantaneous Fields -----
"""
    For the magnetization field
"""
struct MagnetizationField{T} <: AbstractMagnetization
    field::AbstractArray{T,1}
end

function MagnetizationField(::Type{T}, sze::Int) where {T}
    MagnetizationField{T}(zeros(T, sze))
end
    

"""
    magnetization
"""
function calcField(ising::AbstractIsingModel, Beta::T, field::MagnetizationField,  
    upd_alg::AbstractUpdateIsingModel) where{T}
    magnetization(ising, Beta, field, upd_alg)
end



"""
    for the total local energy field
"""
struct LocalEnergyField{T} <: AbstractEnergy
    field::AbstractArray{T,1}
end

function LocalEnergyField(::Type{T}, sze::Int) where {T}
    LocalEnergyField{T}(zeros(T, sze))
end

"""
    for the interaction energy field
"""
struct InteractionEnergyField{T} <: AbstractEnergy
    field::AbstractArray{T,2}
end

function InteractionEnergyField(::Type{T}, sze::Int) where {T}
    InteractionEnergyField{T}(spzeros(T, sze, sze))
end

"""
    energy
"""
function calcField(ising::AbstractIsingModel, Beta::T, field::LocalEnergyField,  
    upd_alg::AbstractUpdateIsingModel) where{T}
    energy(ising, Beta, field, upd_alg)
end

function calcField(ising::AbstractIsingModel, Beta::T, field::InteractionEnergyField,  
    upd_alg::AbstractUpdateIsingModel) where{T}
    energy(ising, Beta, field, upd_alg)
end





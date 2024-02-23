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
    for the energy field
"""
struct EnergyField{T} <: AbstractEnergy
    field::AbstractArray{T,1}
end

function EnergyField(::Type{T}, sze::Int=1) where {T}
    EnergyField{T}(zeros(T, sze))
end

"""
    energy
"""
function calcField(ising::AbstractIsingModel, Beta::T, field::EnergyField,  
    upd_alg::AbstractUpdateIsingModel) where{T}
    energy(ising, Beta, field, upd_alg)
end





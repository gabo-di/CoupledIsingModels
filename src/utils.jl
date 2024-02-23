"""
    The sigmoid function
    note that 
    σ(-Inf) ~ 0
    σ(Inf) ~ 1
"""
function sigmoid(x)
    1/(1 + exp(-x))
end

"""
    TAP equation for magnetization
        m_i = tanh(β H_i + β J_ij*m_j - β² m_i (J_ij)^2*(1 - m_j^2)  ) )
    see for example https://www.nature.com/articles/s41467-021-20890-5
    equation (20)
"""
function _TAPEquation(m, Beta, J, H)
    tanh.(Beta*(H + J*m - Beta*(J.^2)*(1 .- m.^2 ))) 
end

"""
    Magnetization in Lattice Ising Model in 1D with 
    first neighbours interactions 
    see for example https://www.thphys.uni-heidelberg.de/~wolschin/statsem20_3s.pdf
    equation 3.27
"""
function _magnetization1DLatticeModel(Beta, J, H)
    exp(2*Beta*J)*sinh(Beta*H)/sqrt(exp(4*Beta*J)*sinh(Beta*H)^2+1) 
end

"""
   Spontaneous Magnetization in Lattice Ising Model in 2D with 
   first neighbours interactions 
   see for example https://en.wikipedia.org/wiki/Ising_model#Two_dimensions
   equation in Onsager's formula for spontaneous magnetization 
"""
function _magnetization2DLatticeModel(Beta, J, H)
    (1 - (sinh(2*Beta*J))^(-4))^(1/8)
end
    
    

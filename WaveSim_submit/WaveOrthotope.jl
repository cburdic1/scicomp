export WaveOrthotope, dampingcoef, simtime

struct WaveOrthotope{T<:AbstractFloat, N} <: AbstractArray{Tuple{T, T}, N}
    c::T           # damping coefficient
    t::Ref{T}      # simulation time
    u::Array{T, N} # displacement
    v::Array{T, N} # displacement velocity

    function WaveOrthotope{T}(c::Real, t::Real, u::AbstractArray, v::AbstractArray) where T
        size(u) == size(v) || throw(DimensionMismatch("u and v must be same size"))
        min(size(u)...) > 0 || throw(ArgumentError("Size must be positive"))
        return new{T, ndims(u)}(c, Ref(T(t)), u, v)
    end
end

function WaveOrthotope(c::Real, t::Real, u::AbstractArray, v::AbstractArray)
    T = promote_type(typeof.((c, t))..., eltype.((u, v))...)
    return WaveOrthotope{T}(c, t, u, v)
end

function WaveOrthotope{T}(c::Real, t::Real, m::Integer...) where T
    return WaveOrthotope{T}(c, t, zeros(m...), zeros(m...))
end

function WaveOrthotope(c::Real, t::Real, m::Integer...)
    T = promote_type(typeof.((c, t))...)
    if !(T <: AbstractFloat)
        T = defaultT
    end
    return WaveOrthotope{T}(c, t, zeros(m...), zeros(m...))
end

Base.size(w::WaveOrthotope) = size(w.u)
Base.getindex(w::WaveOrthotope, args...) = getindex(w.u, args...)
Base.setindex!(w::WaveOrthotope, args...) = setindex!(w.u, args...)

simtime(w::WaveOrthotope) = w.t[]
dampingcoef(w::WaveOrthotope) = w.c

function Base.:(==)(w1::WaveOrthotope, w2::WaveOrthotope)
    return all((dampingcoef(w1), simtime(w1), w1.u, w1.v) .==
               (dampingcoef(w2), simtime(w2), w2.u, w2.v))
end

function Base.isapprox(w1::WaveOrthotope, w2::WaveOrthotope; kw...)
    return all(isapprox.((dampingcoef(w1), simtime(w1), w1.u, w1.v),
                         (dampingcoef(w2), simtime(w2), w2.u, w2.v); kw...))
end

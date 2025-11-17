export WaveOrthotope, dampingcoef, simtime



"""
    WaveOrthotope{T<:AbstractFloat, N} <: AbstractArray{Tuple{T, T}, N}

Representation of an `N`-dimensional wave simulation orthotope.

Contains displacement and its velocity, a damping coefficient, and the simulation time.

Indexing returns the displacement at the given index.

The damping coefficient and simulation time can be accessed with `dampingcoef` and
`simtime`, respectively. Iteration and energy calculation are done with `energy`, `step!`,
and `solve!`.
"""
struct WaveOrthotope{T<:AbstractFloat, N} <: AbstractArray{Tuple{T, T}, N}
    c::T           # damping coefficient
    t::Ref{T}      # simulation time (Ref so that it can be modified)
    u::Array{T, N} # displacement
    v::Array{T, N} # displacement velocity

    function WaveOrthotope{T}(c::Real, t::Real, u::AbstractArray, v::AbstractArray) where T
        size(u) == size(v) || throw(DimensionMismatch("u and v must be of the same size"))
        min(size(u)...) > 0 || throw(ArgumentError("Size must be positive"))
        return new{T, ndims(u)}(c, Ref(T(t)), u, v)
    end
end



"""
    WaveOrthotope[{T}](c, t, u::AbstractArray, v::AbstractArray)[ where T]

Return a `WaveOrthotope{T, ndims(u)}` with damping coefficient `c`, simulation time `t`, and
displacement and velocity `u` and `v`.

`u` and `v` must be the same size.

`T` and the dimension of `u` and `v` determine the `WaveOrthotope`'s type parameters. By
default, `T` will be the type that results from promoting the types of `c`, `t`, and the
members of `u` and `v`.
"""
function WaveOrthotope(c::Real, t::Real, u::AbstractArray, v::AbstractArray)
    T = promote_type(typeof.((c, t))..., eltype.((u, v))...)
    return WaveOrthotope{T}(c, t, u, v)
end



"""
    WaveOrthotope[{T}](c, t, m::Integer...)[ where T]

Create a `WaveOrthotope{T, length(m...)}` of size `m...` with damping coefficient `c` and
simulation time `t` with zero displacement and velocity.

By default, `T` will be the type that results from promoting the types of `c` and `t`,
unless this results in an integer type in which case it will be `$defaultT`.
"""
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



# AbstractArray interface
Base.size(w::WaveOrthotope) = size(w.u)

Base.getindex(w::WaveOrthotope, args...) = getindex(w.u, args...)

Base.setindex!(w::WaveOrthotope, args...) = setindex!(w.u, args...)



# Comparison operators

function Base.:(==)(w1::WaveOrthotope, w2::WaveOrthotope)
    return all((dampingcoef(w1), simtime(w1), w1.u, w1.v) .==
               (dampingcoef(w2), simtime(w2), w2.u, w2.v))
end

function Base.isapprox(w1::WaveOrthotope, w2::WaveOrthotope; kw...)
    return all(isapprox.((dampingcoef(w1), simtime(w1), w1.u, w1.v),
                         (dampingcoef(w2), simtime(w2), w2.u, w2.v); kw...))
end



# Getters

"""
    simtime(w::WaveOrthotope)

Return `w`'s simulation time.
"""
simtime(w::WaveOrthotope) = w.t[]

"""
    dampingcoef(w::WaveOrthotope)

Return `w`'s damping coefficient.
"""
dampingcoef(w::WaveOrthotope) = w.c

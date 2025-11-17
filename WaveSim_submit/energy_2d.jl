export energy

"""
    energy(w, component=:total)

Fully optimized 2D energy calculation with SIMD.
"""
function energy(w::WaveOrthotope{<:Real,2}, component::Symbol=:total)
    component in (:total,:dynamic,:potential) ||
        throw(ArgumentError("Energy component must be :total, :dynamic, or :potential"))

    m, n = size(w)
    u = w.u
    v = w.v

    E_dyn = 0.0
    E_pot = 0.0

    # Dynamic energy
    @inbounds @fastmath for j in 2:n-1
        @simd for i in 2:m-1
            E_dyn += 0.5 * v[i,j]^2
        end
    end

    # Potential (x-direction)
    @inbounds @fastmath for j in 2:n-1
        @simd for i in 1:m-1
            d = u[i+1,j] - u[i,j]
            E_pot += 0.25 * d*d
        end
    end

    # Potential (y-direction)
    @inbounds @fastmath for j in 1:n-1
        @simd for i in 2:m-1
            d = u[i,j+1] - u[i,j]
            E_pot += 0.25 * d*d
        end
    end

    component === :dynamic   && return E_dyn
    component === :potential && return E_pot
    return E_dyn + E_pot
end

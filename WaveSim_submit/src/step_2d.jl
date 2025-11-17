export step!

"""
    step!(w::WaveOrthotope, dt=defaultdt)

Optimized 2D leapfrog step with no allocations.
"""
function step!(w::WaveOrthotope{<:Real,2}, dt=defaultdt)
    dt < 0 && throw(ArgumentError("Stepping back is not supported."))

    m, n = size(w)
    u = w.u
    v = w.v
    c = w.c

    half = 0.5
    damp = 1 - dt * c

    @inbounds for j in 2:n-1
        @simd for i in 2:m-1
            L = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]) * half
            v[i,j] = damp * v[i,j] + dt * L
        end
    end

    @inbounds for j in 2:n-1
        @simd for i in 2:m-1
            u[i,j] += dt * v[i,j]
        end
    end

    w.t[] += dt
    return w
end


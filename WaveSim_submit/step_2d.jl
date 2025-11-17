export step!

"""
    step!(w::WaveOrthotope, dt=defaultdt)

Aggressively optimized 2D leapfrog update.
Fully inlined, SIMD-friendly, minimal memory traffic.
"""
function step!(w::WaveOrthotope{<:Real,2}, dt=defaultdt)
    dt < 0 && throw(ArgumentError("Stepping back is not supported."))

    m, n = size(w)
    u = w.u
    v = w.v
    c = w.c

    @inbounds @fastmath for j in 2:n-1
        @simd for i in 2:m-1
            ui = u[i,j]
            L  = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*ui) * 0.5
            vi = v[i,j]
            vi = vi + dt*(L - c*vi)
            v[i,j] = vi
            u[i,j] = ui + dt*vi
        end
    end

    w.t[] += dt
    return w
end

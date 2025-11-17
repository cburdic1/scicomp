import Base.Threads

# Optimized explicit 2D update with 5-point Laplacian
@inline function _step2d!(u::Matrix{Float64}, v::Matrix{Float64},
                          dt::Float64, c::Float64)

    nx, ny = size(u)

    @inbounds begin
        # interior update (threaded, cache-friendly)
        Threads.@threads :static for j in 2:ny-1
            @simd for i in 2:nx-1
                @fastmath begin
                    ui  = u[i,j]
                    vij = v[i,j]

                    lap = u[i-1,j] + u[i+1,j] +
                          u[i,j-1] + u[i,j+1] - 4.0*ui

                    vij += dt*(lap - c*vij)
                    v[i,j] = vij
                    u[i,j] = ui + dt*vij
                end
            end
        end

        # fused reflective boundaries — faster + fewer cache misses
        @simd for j in 1:ny
            u1   = u[2,j]
            uend = u[nx-1,j]
            v1   = v[2,j]
            vend = v[nx-1,j]

            u[1,j]  = u1
            u[nx,j] = uend
            v[1,j]  = v1
            v[nx,j] = vend
        end

        @simd for i in 1:nx
            u2   = u[i,2]
            uend = u[i,ny-1]
            v2   = v[i,2]
            vend = v[i,ny-1]

            u[i,1]  = u2
            u[i,ny] = uend
            v[i,1]  = v2
            v[i,ny] = vend
        end
    end

    return nothing
end

# used by solve! driver
@inline function step!(w::WaveOrthotope, dt::Float64)
    _step2d!(w.u, w.v, dt, w.c)
    return w
end

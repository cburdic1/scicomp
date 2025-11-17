import Base.Threads
export energy

@inline function energy(w::WaveOrthotope, kind::Symbol=:total)
    u = w.u
    v = w.v
    nx, ny = size(u)

    t = Threads.nthreads()
    kpart = zeros(Float64, t)
    ppart = zeros(Float64, t)

    @inbounds begin
        # kinetic energy
        Threads.@threads :static for j in 1:ny
            tid = Threads.threadid()
            acc = 0.0
            @simd for i in 1:nx
                @fastmath acc += 0.5 * v[i,j] * v[i,j]
            end
            kpart[tid] += acc
        end

        # potential energy
        Threads.@threads :static for j in 2:ny-1
            tid = Threads.threadid()
            acc = 0.0
            @simd for i in 2:nx-1
                @fastmath begin
                    gx = u[i+1,j] - u[i-1,j]
                    gy = u[i,j+1] - u[i,j-1]
                    acc += 0.5*(gx*gx + gy*gy)
                end
            end
            ppart[tid] += acc
        end
    end

    kinetic   = sum(kpart)
    potential = sum(ppart)

    kind === :kinetic   && return kinetic
    kind === :potential && return potential
    return kinetic + potential
end

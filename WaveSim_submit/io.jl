export read_wo

function read_wo(path::String)
    open(path) do io
        m  = read(io, Int64)
        n  = read(io, Int64)
        nt = read(io, Int64)
        dt = read(io, Float64)
        c  = read(io, Float64)

        u = Array{Float64}(undef, m, n)
        v = Array{Float64}(undef, m, n)

        read!(io, u)
        read!(io, v)

        return m, n, nt, dt, c, u, v
    end
end

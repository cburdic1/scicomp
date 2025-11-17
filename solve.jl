@inline function solve!(w::WaveOrthotope)
    u  = w.u
    v  = w.v
    dt = (Base.hasproperty(w, :dt) ? getfield(w, :dt) : defaultdt)
    c  = w.c
    nt = w.nt

    @inbounds for _ in 1:nt
        _step2d!(u, v, dt, c)
    end
    return w
end

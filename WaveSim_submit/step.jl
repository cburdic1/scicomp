export step!

"""
    step!(w::WaveOrthotope, dt=$defaultdt)

Update `w` by `dt` seconds in one step using leapfrog integration.
"""
function step!(w::WaveOrthotope{<:Real, N}, dt=defaultdt) where N
    # Helpers
    c = w.c
    u = interior(w.u)
    v = interior(w.v)
    a = adjacents(u)
    # Order of operations and update function for v depend on sign of dt
    if dt < 0
        @. u += dt*v
        L = @. +(a...)/2 - N*u
        @. v = (v+dt*L) / (1+dt*c)
    else
        L = @. +(a...)/2 - N*u
        @. v = (1-dt*c)*v + dt*L
        @. u += dt*v
    end
    w.t[] += dt
    return w
end

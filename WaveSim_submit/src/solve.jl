export solve!



"""
    solve!(w::WaveOrthotope, dt=defaultdt)

Repeatedly update `w` using `step!` with time step `dt` until the energy of `w` drops below
an average of 0.001 per interior cell.
"""
function solve!(w::WaveOrthotope, dt=defaultdt)
    stoppingenergy = prod(size(w).-2) / 1000
    while energy(w) > stoppingenergy
        step!(w, dt)
    end
    return w
end

export energy



"""
    energy(w::WaveOrthotope, component=:total)

Return the specified component of the energy contained in `w`.

`component` can be `:total`, `:dynamic`, or `:potential`.
"""
function energy(w::WaveOrthotope{T, N}, component=:total) where {T, N}
    # Dynamic
    DE = sum(interior(w.v).^2)/2
    # Potential
    PE = 0
    for i in 1:N
        upper = interior(w.u, ntuple(j->j==i ? (0, 1) : (1, 1), N)...) # 2:n-1,...,1:n-1,...
        lower = interior(w.u, ntuple(j->j==i ? (1, 0) : (1, 1), N)...) # 2:n-1,...,2:n  ,...
        PE += sum((upper.-lower).^2)/4
    end
    # Return the appropriate compoment of energy
    component == :total     && return DE+PE
    component == :dynamic   && return DE
    component == :potential && return PE
    throw(ArgumentError("Unrecognized component $component--options are :total, :dynamic," *
                        " and :potential"))
end

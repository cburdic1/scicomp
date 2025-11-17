module WaveSim

using Preferences

export WaveOrthotope, step!, solve!, energy, read_wo, wavefiles

const defaultdt = 0.01
const defaultT  = Float64
const defaultI  = UInt64

include("WaveOrthotope.jl")
include("io.jl")
include("utils.jl")
include("solve.jl")

impl = @load_preference("implementation", "2D")

if impl == "ND"
    include("step.jl")
    include("energy.jl")
else
    include("step_2d.jl")
    include("energy_2d.jl")
end

end

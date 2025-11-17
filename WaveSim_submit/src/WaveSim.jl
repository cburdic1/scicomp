"""
`WaveSim` contains tools to simulate the damped wave equation ``ü=∇²u-cu̇``.

For an overview of the associated implementations, justification of the equation, and
discretization methods, see https://byuhpc.github.io/sci-comp-course/project/overview.html.

The [`WaveOrthotope`](@ref) type is used to store a wave orthotope's simulation time,
damping coefficient, displacement, and displacement velocity. The simulation can be moved
forward in time with [`step!`](@ref) and [`solve!`](@ref), and its energy determined with
[`energy`](@ref). See [`WaveOrthotope`](@ref) for a full list of associated functions.

You can choose between N-dimensional and 2-dimensional implementations of `WaveSim` with
`set_implementation`.

`wavefiles.tar.gz` is provided as an artifact; files therein can be accessed with the
[`wavefiles`](@ref) function.
"""
module WaveSim

using Preferences



"""
    set_implementation(implementation)

Choose "ND" or "2D" versions of `Mountains`; "ND" is the default.

These are the files in `src` that differ by preference:

| "ND"        | "2D"           |
| ----------- | -------------- |
| `energy.jl` | `energy_2d.jl` |
| `step.jl`   | `step_2d.jl`   |

# Examples

Set implementation to "2D" from the shell:

```bash
julia -e 'using WaveSim; WaveSim.set_implementation("2D")'
```
"""
function set_implementation(impl::String)
    impl in ("ND", "2D") || throw(ArgumentError(
            "Invalid implementation '$impl'; options are 'ND' and '2D'."))
    @set_preferences!("implementation" => impl)
    @info("implementation set to $impl; restart Julia for the change to take effect.")
end

# Set default implementation (2D or ND)
const implementation = @load_preference("implementation", "ND")



# Default simulation parameters
const defaultdt = 0.01
const defaultT = Float64
const defaultI = UInt64



# Includes that are identical across implementations
include("WaveOrthotope.jl")
include("io.jl")
include("utils.jl")
include("solve.jl")

# implementations-specific includes
if implementation == "ND"
    include("step.jl")
    include("energy.jl")
else
    include("step_2d.jl")
    include("energy_2d.jl")
end



end

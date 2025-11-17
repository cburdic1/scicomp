# WaveSim.jl

`WaveSim.jl` is a package that's meant to make life easier for students of [BYU's Scientific Computing Course](https://byuhpc.github.io/sci-comp-course/). It serves to make debugging some phases of the [project](https://byuhpc.github.io/sci-comp-course/project/overview.html) easier, and is used as example and base code for others.



## Install

Open the [Julia package manager](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode) and run:

```jldoctest
pkg> add https://github.com/BYUHPC/WaveSim.jl.git
```

`WaveSim.jl` can be removed with:

```jldoctest
pkg> remove WaveSim
```



## Usage Example

The `WaveSim` package can be used to interactively check your assignment output files--for example, given an input file `in.wo`, you can see if your C++ code compiled to `wavesim_serial` is correct, and figure out what's wrong if not:

```jldoctest
shell> ./wavesim_serial in.wo out.wo

julia> using WaveSim

julia> wgood = WaveOrthotope(open("in.wo"));

julia> solve!(wgood)

julia> wtest = WaveOrthotope(open("out.wo"));

julia> if !isapprox(wgood, wtest)
           println("out.wo is incorrect")
           println("Damping coefficients: $(dampingcoef(wgood)), $(dampingcoef(wtest))")
           println("Simulation times:     $(simtime(wgood)), $(simtime(wtest))")
           println("Max u difference:     $(maximum(abs.(wgood.u - wtest.u))...)")
           println("Max v difference:     $(maximum(abs.(wgood.v - wtest.v))...)")
       end
```



## Easy `wavefiles.tar.gz` Access

[`wavefiles.tar.gz`](https://rc.byu.edu/course/wavefiles.tar.gz) is provided as a lazily loaded [artifact](https://docs.julialang.org/en/v1/stdlib/Artifacts/). You can use the provided `wavefiles` function to access files therein:

```julia
small2din = WaveOrthotope(wavefiles(2, :small, :in)) # small-2d-in.wo
tiny3dout = WaveOrthotope(wavefiles("3d-tiny-out.wo"))
wavediffbinary = wavefiles("wavediff")
```

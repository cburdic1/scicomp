export wavefiles

using Artifacts, LazyArtifacts, GZip



"""
    wavefiles()
    wavefiles(filename::AbstractString)
    wavefiles(N::Integer, filesize::Symbol, in_or_out::Symbol)

Get the path to a file from `wavefiles.tar.gz`.

You can access files in the `wavefiles` directory of `wavefiles.tar.gz` (see
https://byuhpc.github.io/sci-comp-course/resources.html#the-project) by `filename` or with
a number of dimensions, `filesize`, and whether you want an input or output file.

`N` can be any integer between 1 and 8, although 5-8 dimensions are only available for small
and medium orthotopes; `filesize` can be `:tiny`, `:small`, `:medium`, or `large`; and
`in_or_out` can be `:in` or `:out`.

If no argument is specified, the path to the directory itself is returned.

# Examples

```jldoctest
julia> size(WaveOrthotope(wavefiles(2, :small, :in)))
(83, 120)

julia> simtime(WaveOrthotope(wavefiles("1D/1d-medium-out.wo")))
69.72999999999877

julia> first(readdir(wavefiles("1D")), 3)
3-element Vector{String}:
 "1d-large-in.wo.gz"
 "1d-medium-in.wo"
 "1d-medium-out.wo"
```
"""
function wavefiles(filename::AbstractString="")
    filepath = joinpath(artifact"wavefiles/wavefiles", filename)
    if ispath(filepath)
        return filepath
    else
        throw(ArgumentError("wave file $filename does not exist in wavefiles.tar.gz"))
    end
end

function wavefiles(N::Integer, filesize::Symbol, in_or_out::Symbol)
    if !(filesize in (:tiny, :small, :medium, :large))
        throw(ArgumentError("filesize must be :tiny, :small, :medium, or :large"))
    end
    in_or_out in (:in, :out) || throw(ArgumentError("in_or_out must be :in or :out"))
    maxN = filesize == :tiny || filesize == :large ? 4 : 8
    1 <= N <= maxN || throw(ArgumentError("N must be in [1, $maxN] for size $filesize"))
    filename = joinpath("$(N)D", "$(N)d-$(String(filesize))-$(String(in_or_out)).wo")
    if filesize == :large
        filename *= ".gz"
    end
    return wavefiles(filename)
end



# Display
function Base.show(io::IO, mime::MIME"text/plain", w::WaveOrthotope{T, N}) where {T, N}
    # Print type information and header
    sizestr = N > 1 ? join(string.(size(w)), "×") : string(length(w))*"-element"
    println(io, "$sizestr WaveOrthotope{$T, $N} with damping coefficient " *
                "$(dampingcoef(w)) and simulation time $(simtime(w))")
    # Print displacement
    println()
    print(io, "Displacement -- ")
    show(io, mime, w.u)
    # Print displacement velocity
    println()
    println()
    print(io, "Displacement velocity -- ")
    show(io, mime, w.v)
end



"""
    WaveOrthotopeReadException(msg::String, w::WaveOrthotope)

Reading a `WaveOrthotope` from an `IO` failed.

Contains members `msg::String` and `w::WaveOrthotope`.

`w` is included for debugging purposes--it likely will not be in a valid state and should be
inspected but not used.

`w` will contain all the data that was successfully read in. Any fields that weren't
successfully read will be set to `typemax(T)`, where `T` is the element type or index type
of `w` depending on the field; the exception is the size, which defaults to `(1,)`. `u` and
`v` will be `Vector`s of length 1 if the size isn't successfully read in.

# Examples

```jldoctest
julia> try
           io = IOBuffer()
           w = WaveOrthotope(io)
       catch e
           println(e.msg)
           println(size(e.w))
       end
stream didn't contain enough data
(1,)
```
"""
struct WaveOrthotopeReadException <: Exception
    msg::String
    w::WaveOrthotope
end



"""
    invertaxes(A::AbstractArray)

Convert an array between C and Julia ordering.

Equivalent to `transpose(A)` for 1- and 2-dimensional arrays.
"""
invertaxes(A::AbstractArray{T, N}) where {T, N} = permutedims(A, ntuple(n->N-n+1, N))



# Construct a WaveOrthotope from an IO

"""
    WaveOrthotope[{T}](io::IO, I=$defaultI; checkstreamlength=io isa IOStream)[ where T]

Create a `WaveOrthotope{T, N}` by reading binary from `io`.

`T` and `I`, the floating point and size types of the binary wave orthotope, default to
`$defaultT` and `$defaultI` respectively. `N` is determined by reading the stream itself.

Throws a `WaveOrthotopeReadException` if the stream doesn't contain enough data, if the data
appears corrupt, or if the stream contains more data than necessary and `checkstreamlength`
is `true` (the default if `io` is an `IOStream`).

See `write` for the binary format of a `WaveOrthotope`.
"""
function WaveOrthotope{T}(io::IO, I=defaultI; checkstreamlength=io isa IOStream) where T
    # Initialize so that a partly correct `WaveOrthotope` can be included in a
    # `WaveOrthotopeReadException` on failure
    N, c, t = typemax.((I, T, T))
    m = (one(I),)
    u = [typemax(T)]
    v = copy(u)
    try
        # Read header
        N = read(io, I)
        m = ntuple(n->read(io, I), N)
        c = read(io, T)
        t = read(io, T)
        # Read body; size is reversed due to opposite array ordering
        u = fill(typemax(T), reverse(m))
        v = copy(u)
        read!(io, u)
        read!(io, v)
        # Construct WaveOrthotope; invertaxes is needed to go from C to Julia array order
        w = WaveOrthotope{T}(c, t, invertaxes(u), invertaxes(v))
        # Throw an exception if there was a problem with the read
        if checkstreamlength && !eof(io)
            throw(WaveOrthotopeReadException("stream contains unread data", w))
        end
        # Otherwise, return the WaveOrthotope
        return w
    catch e
        w = WaveOrthotope{T}(c, t, invertaxes(u), invertaxes(v))
        s = join(string.(size(w)), "×")
        if e isa EOFError
            throw(WaveOrthotopeReadException("stream didn't contain enough data", w))
        elseif (e isa OutOfMemoryError)
            throw(WaveOrthotopeReadException("size requested ($s) won't fit in memory", w))
        elseif (e isa ErrorException && e.msg == "invalid Array size")
            throw(WaveOrthotopeReadException("invalid size ($s) requested", w))
        else
            rethrow(e)
        end
    end
end

WaveOrthotope(io::IO, I=defaultI; kw...) = WaveOrthotope{defaultT}(io; kw...)



# Construct a WaveOrthotope from a file

"""
    WaveOrthotope[{T}](filename::AbstractString, I=$defaultI; kw...)[ where T]

Open `filename` and construct a `WaveOrthotope` from it.

Equivalent to calling `WaveOrthotope[{T}](open(filename), I; kw...)` (or
`WaveOrthotope[{T}](gzopen(filename), I; kw...)` if the file is gzip compressed).
"""
function WaveOrthotope{T}(filename::AbstractString, I=defaultI; kw...) where T
    gzipsignature = 0x8B1F
    f = read(open(filename), UInt16) == gzipsignature ? gzopen(filename) : open(filename)
    return WaveOrthotope{T}(f, I; kw...)
end

function WaveOrthotope(filename::AbstractString, I=defaultI; kw...)
    return WaveOrthotope{defaultT}(filename, I; kw...)
end



# Write a WaveOrthotope to an IO

"""
    write(io::IO, w::WaveOrthotope{T, N}, I=$defaultI) where {T, N}

Write `w` to `io` in binary.

The binary representation of a `WaveOrthotope` is ordered thus:

1. `N`: the number of dimensions of the wave orthotope, an `I`
1. `m...`: the size of the wave orthotope, `N` of `I`
1. `c`: the damping coefficient, a `T`
1. `t`: the simulation time, a `T`
1. `u`: the displacement matrix, `prod(m...)` `T`s in C array order
1. `v`: the displacement velocity matrix, `prod(m...)` `T`s in C array order
"""
function Base.write(io::IO, w::WaveOrthotope{T, N}, I::Type{<:Integer}=defaultI
                    ) where {T, N}
    # Write header
    writes  = write(io, I(N))
    writes += write(io, I.(size(w))...)
    writes += write(io, dampingcoef(w), simtime(w))
    # Write body; permutation is required to go from Julia to C array order
    writes += write(io, invertaxes(w.u), invertaxes(w.v))
    return writes
end

using WaveSim
using Test



"""
    test_on_random_WaveOrthotopes(testname, f::Function)

Run `f(w)` on 1-, 2-, 3-, and 4-dimensional random `WaveOrthotopes` within a `@testset`.

`f` should have at least one `@test` or the resulting `@testset` will be meaningless. The
`WaveOrthotope`s passed to it have random displacement and velocity, although the edges of
each are set to zero. They are of random size, random damping coefficient, and random
simulation time, although sizes are odd along each axis to accomodate the energy test.

If `energy(WaveOrthotope{T, N})` and `step!(WaveOrthotope{T, N})` are not defined for a
given `N`, no test is run. This means that implementations that are only meant to work for
one or two dimensions, for example, can still pass all the tests.
"""
function test_on_random_WaveOrthotopes(testname, f)
    @testset "$testname" begin
        for w in (begin
                      lowest = Int(round(4^(4/N)))
                      isodd(lowest) || (lowest += 1) # there's an energy test relying on odd sizes
                      highest = Int(round(7^(4/N)))
                      randomsize = ntuple(i->rand(lowest:2:highest), N)
                      w = WaveOrthotope(rand(), rand(), randomsize...)
                      WaveSim.interior(w.u) .= rand((randomsize.-2)...)
                      WaveSim.interior(w.v) .= rand((randomsize.-2)...)
                      w
                  end
                  for N in 1:4
                  if all(hasmethod.((energy, step!), Tuple{WaveOrthotope{<:Real, N}})))
            f(w)
        end
    end
end



@testset "WaveSim" begin
    test_on_random_WaveOrthotopes("Constructors", w->begin
        m = size(w)
        # Default type is correct
        @test typeof(w) == WaveOrthotope{WaveSim.defaultT, length(size(w.u))}
        # Specify u and v
        c = rand()
        t = rand()
        u = rand(m...)
        v = rand(m...)
        w1 = WaveOrthotope(c, t, u, v)
        @test c == dampingcoef(w1)
        @test t == simtime(w1)
        @test u == w1.u
        @test v == w1.v
        # Specify a size
        w2 = WaveOrthotope(c, t, size(w)...)
        @test w2.u == w2.v == zeros(m...)
        # Make sure an ArgumentError is thrown if a size component is negative
        m_negative = ntuple(n->isodd(n) ? -m[n] : m[n], ndims(w))
        @test_throws ArgumentError WaveOrthotope(rand(), rand(), rand(m_negative...),
                                                                 rand(m_negative...))
        # Make sure an ArgumentError is thrown if sizes differ
        m_bigger = ntuple(n->isodd(n) ? m[n]+1 : m[n], ndims(w))
        @test_throws DimensionMismatch WaveOrthotope(rand(), rand(), rand(m...),
                                                                     rand(m_bigger...))
    end)



    test_on_random_WaveOrthotopes("Equality and approximate equality", w->begin
        # Basic equality
        w1 = deepcopy(w)
        @test w == w1
        @test simtime(w)     == simtime(w1)
        @test dampingcoef(w) == dampingcoef(w1)
        @test w.u            == w1.u
        @test w.v            == w1.v
        # Perturb one value and check that w and w1 are not equal, but may be about equal
        for reference in (w1.t, view(w1.u, rand(CartesianIndices(w1))),
                                view(w1.v, rand(CartesianIndices(w1))))
            # Perturb
            original = reference[]
            reference[] += 1e-4
            # Not equal
            @test w != w1
            # Only approximately equal with sufficiently large tolerance
            @test !isapprox(w, w1, atol=1e-6)
            @test  isapprox(w, w1, atol=1e-2)
            # Reset
            reference[] = original
            @test w == w1
        end
    end)



    test_on_random_WaveOrthotopes("AbstractArray interface", w->begin
        # size
        @test size(w) == size(w.u) == size(w.v)
        # getindex
        @test all(w[i] == w.u[i] for i in eachindex(w))
        # setindex!
        w[begin] = 2
        @test w.u[begin] == 2
    end)



    test_on_random_WaveOrthotopes("energy", w->begin
        # Fill the WaveOrthotope with specific values
        m = size(w)
        fillvalue = rand()
        w.u .= 0
        w.u[begin:2:end] .= fillvalue
        WaveSim.interior(w.v)[begin:end] .= sqrt.(range(0, 2*fillvalue, prod(m.-2)))
        # Check that dynamic and potentail components are as expected
        DE = energy(w, :dynamic)
        PE = energy(w, :potential)
        @test DE ≈ *((m.-2)...)/2*fillvalue
        @test PE ≈ sum(*(ntuple(j->i==j ? m[j]-1 : m[j]-2, length(m))...)
                       for i in 1:length(m))/4 * fillvalue^2
        # Check that total energy is calculated correctly
        @test energy(w, :total) ≈ DE+PE
        @test energy(w) ≈ DE+PE
        # Ensure that invalid arguments to energy result in an ArgumentError
        @test_throws ArgumentError energy(w, :noncomponent)
    end)



    test_on_random_WaveOrthotopes("step!", w->begin
        # Create and step a copy of w
        dt = rand()
        w1 = deepcopy(w)
        step!(w1, dt)
        # Convenience views
        u1, v1, u, v = WaveSim.interior.((w1.u, w1.v, w.u, w.v))
        uₐ = WaveSim.adjacents(u)
        # Check stepped orthotope against actual algorithm
        ∇²u1 = +(uₐ...)/2 - ndims(w)*u
        @test v1 ≈ @. (1-dt*w.c)*v+dt*∇²u1
        @test u1 ≈ @. u+dt*v1
        @test simtime(w1) ≈ simtime(w)+dt
        # Backward step works (if supported)
        try
            step!(w1, -dt)
            @test w1 ≈ w
        catch e
            e isa ArgumentError || rethrow(e)
        end
    end)



    test_on_random_WaveOrthotopes("solve!", w->begin
        # Solve
        solve!(w)
        # Make sure energy is about right
        stoppingenergy = prod(size(w).-2)/1000
        @test energy(w) ≈ stoppingenergy || energy(w) < stoppingenergy
        # If we step back by one time step, energy should go back up (if supported)
        try
            step!(w, -WaveSim.defaultdt)
            @test energy(w) > stoppingenergy
        catch e
            e isa ArgumentError || rethrow(e)
        end
    end)



    test_on_random_WaveOrthotopes("I/O", w->begin
        T = WaveSim.defaultT
        I = WaveSim.defaultI
        N = ndims(w)
        header = (I(N), I.(size(w))..., T(dampingcoef(w)), T(simtime(w)))
        # Writing to a stream results in the right number of bytes written
        buf = IOBuffer()
        headersize = sum(sizeof.(typeof.(header)))
        bodysize = sizeof(typeof(w.u[begin])) * length(w) * 2
        @test write(buf, w) == headersize + bodysize
        # Reading from a stream results in a correct WaveOrthotope
        seekstart(buf)
        @test w == WaveOrthotope(buf)
        # Reading from an empty stream throws
        @test_throws WaveSim.WaveOrthotopeReadException WaveOrthotope{T}(buf)
        # WaveOrthotope serialization is consistent with the standard
        buf = IOBuffer()
        write(buf, w)
        seekstart(buf)
        for h in header
            @test read(buf, typeof(h)) == h
        end
        # Permutation is required to go between C and Fortran array order
        perm = ntuple(n->N-n+1, N)
        uₚ = permutedims(w.u, perm)
        vₚ = permutedims(w.v, perm)
        @test all(read(buf, T) == a[i] for a in (uₚ, vₚ) for i in 1:length(w))
        # Reading too few bytes throws
        buf = IOBuffer()
        for h in header
            write(buf, h)
        end
        write(buf, uₚ)
        write(buf, vₚ[begin:end-1])
        seekstart(buf)
        @test_throws WaveSim.WaveOrthotopeReadException WaveOrthotope{T}(buf)
        # Reading too many bytes throws only if checkstreamlength is true
        buf = IOBuffer()
        write(buf, w)
        write(buf, '.')
        seekstart(buf)
        @test w == WaveOrthotope{T}(buf, checkstreamlength=false)
        seekstart(buf)
        @test_throws WaveSim.WaveOrthotopeReadException WaveOrthotope{T}(buf,
                                                        checkstreamlength=true)
    end)



    @testset "wavefiles" begin # also tests *.gz constructor
        # Make sure that trying to access a non-existent file throws
        @test_throws ArgumentError wavefiles(9, :small, :in)
        @test_throws ArgumentError wavefiles(5, :tiny, :out)
        @test_throws ArgumentError wavefiles(3, :wrongsymbol, :in)
        @test_throws ArgumentError wavefiles("2D/3d-medium-out.wo")
        # Make sure accessing directories works
        for dir in ("bin", ntuple(N->"$(N)D", 8)...)
            @test isdir(wavefiles(dir))
        end
        # Make sure accessing binaries works
        for binary in ("wavediff", "waveshow", "wavesolve")
            @test isfile(wavefiles("bin/$binary"))
        end
        # Test all provided files for existence and correctness
        for N in 1:8, filesize in (:tiny, :small, :medium), in_or_out in (:in, :out)
            # Short circuit if the file doesn't exist
            if filesize==:tiny && N>4
                continue
            end
            # Make sure both forms of the function give identical results
            filename = joinpath("$(N)D",
                                "$(N)d-$(String(filesize))-$(String(in_or_out)).wo")
            @test wavefiles(N, filesize, in_or_out) == wavefiles(filename)
            # Make sure a valid WaveOrthotope can be constructed
            if all(hasmethod.((energy, step!), Tuple{WaveOrthotope{<:Real, N}}))
                wo = WaveOrthotope(wavefiles(filename))
                if in_or_out == :in
                    @test simtime(wo) == 0
                    if filesize == :tiny || filesize == :small
                        @test WaveOrthotope(wavefiles(N, filesize, :out)) ≈ solve!(wo)
                    end
                else
                    @test isapprox(prod(size(wo).-2)/1000, energy(wo), rtol=1e-2)
                end
            end
        end
    end
end

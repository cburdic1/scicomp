"""
    interior(A::AbstractArray{T, N}, offsets::Vararg=ntuple(i->(1, 1), N)...) where {T, N}

Return a view that encompasses all but the outer cells of `A`.

`offset` is a `Vararg` of 2-tuples of numbers of length `N`; the first number is the width
of the "upper" edge that is cut off, the second the width of the "lower" edge; the position
of each offset corresponds to that axis in `A`. By default, a single cell's width is cut off
from the beginning and end along each axis.

# Examples

```jldoctest
julia> x = [10i+j for i=1:4, j=1:5]
4×5 Matrix{Int64}:
 11  12  13  14  15
 21  22  23  24  25
 31  32  33  34  35
 41  42  43  44  45

julia> interior(x)
2×3 view(::Matrix{Int64}, 2:3, 2:4) with eltype Int64:
 22  23  24
 32  33  34

julia> interior(x, (0, 1), (1, 2))
3×2 view(::Matrix{Int64}, 1:3, 2:3) with eltype Int64:
 12  13
 22  23
 32  33
```
"""
function interior(A::AbstractArray{T, N},
                  offsets::Vararg{NTuple{2, Integer}, N}=ntuple(n->(1, 1), N)...
                 ) where {T, N}
    return view(A, ntuple(n->firstindex(A, n)+offsets[n][1]:lastindex(A, n)-offsets[n][2],
                          N)...)
end



"""
    adjacents(A::SubArray[, direction::Real])

Return a tuple of views that correspond to a shift by a positive and/or negative unit vector
along each axis of `a`.

If `direction` is specified, `adjacents` will return a tuple of length equal to the
dimension of `A`. If `direction` is positive, the views will be offset by +1 along each
axis; if it's negative, the offset will be -1 along each axis; if it's 0, the views within
the returned tuple will be identical to `A`. If `direction` isn't specified, the tuples that
would have resulted from both negative and positive arguments are concatenated.

# Examples

```jldoctest
julia> x = [10i+j for i=1:4, j=1:5]
4×5 Matrix{Int64}:
 11  12  13  14  15
 21  22  23  24  25
 31  32  33  34  35
 41  42  43  44  45

julia> xi = interior(x)
2×3 view(::Matrix{Int64}, 2:3, 2:4) with eltype Int64:
 22  23  24
 32  33  34

julia> adjacents(xi, 1)[1]
2×3 view(::Matrix{Int64}, 3:4, 2:4) with eltype Int64:
 32  33  34
 42  43  44

julia> adjacents(xi, -1)[2]
2×3 view(::Matrix{Int64}, 2:3, 1:3) with eltype Int64:
 21  22  23
 31  32  33

julia> x22 = view(x, 2, 2)
0-dimensional view(::Matrix{Int64}, 2, 2) with eltype Int64:
22

julia> getindex.(adjacents(x22), 1)
(12, 21, 32, 23)
```
"""
function adjacents(A::SubArray{T, M, <:AbstractArray{T, N}, I, L}, direction::Real
                  ) where {T, M, N, I, L}
    return ntuple(i->view(A.parent, ntuple(j->A.indices[j].+(j==i ? sign(direction) : 0),
                                           N)...), N)
end
adjacents(A) = (adjacents(A, -1)..., adjacents(A, 1)...)

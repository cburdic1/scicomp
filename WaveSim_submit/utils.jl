export adjacents, interior

interior(A) = @view A[2:end-1, 2:end-1]

function adjacents(A)
    return (
        @view(A[1:end-2, 2:end-1]),
        @view(A[3:end,   2:end-1]),
        @view(A[2:end-1, 1:end-2]),
        @view(A[2:end-1, 3:end])
    )
end

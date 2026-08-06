function get_piece_iterator(N::Int)::Vector{Tuple{Int, Int}}
    pairs = Tuple{Int, Int}[]
    for i in 1:N
        for j in i:min(N, i + 1)
            for k in j:min(N, i + j - 1)
                push!(pairs, (i - (k - j), k))
            end
        end
    end
    return pairs
end

function get_pieces_dict(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Dict{Tuple{Int, Int}, Set{UInt32}}
    pieces = pieces_left(s, bs, bbs)
    N = pieces[1]
    visited = Dict(pair => Set{UInt32}() for pair in get_piece_iterator(N))
    push!(visited[pieces], s)
    return visited
end

function bfs(
    s_start::UInt32,
    bs::Vector{UInt32},
    bbs::Vector{UInt32};
    max_iter::Int=1_000_000_000,
    verbose::Bool=false,
    progress_interval::Int=1_000_000,
)
    frontier = UInt32[s_start]
    leaf_nodes = Set{UInt32}()
    visited = get_pieces_dict(s_start, bs, bbs)
    ns = zeros(UInt32, 7)

    for i in 1:max_iter
        if isempty(frontier)
            return visited, leaf_nodes
        end
        if verbose && i % progress_interval == 0
            println(i)
        end

        s = pop!(frontier)
        for roll in 0:4
            if roll == 0
                fill!(ns, 0)
                ns[1] = flip_turn(s, bs, bbs) - bs[32]
            else
                neighbours!(ns, s, roll, bs, bbs)
            end

            for neighbour in ns
                neighbour == 0 && break
                neighbour, _ = turn_change(neighbour, bs)
                if has_won(neighbour, bs, bbs)
                    push!(leaf_nodes, neighbour)
                    continue
                end

                pieces = pieces_left(neighbour, bs, bbs)
                if neighbour ∉ visited[pieces]
                    push!(visited[pieces], neighbour)
                    pushfirst!(frontier, neighbour)
                end
            end
        end
    end

    throw(ErrorException("BFS iteration limit reached"))
end

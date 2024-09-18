"""
    get_piece_iterator(N)

Get list of (m, n) tuples in monotonic order.
"""
function get_piece_iterator(N::Int)::Vector{Tuple{Int, Int}}
    ls = Vector{Tuple{Int, Int}}()
    for i in 1:N
        for j in i:min(N, i+1)
            for k in j:min(N, i+j-1)
                push!(ls, (i-(k-j), k))
            end
        end
    end
    return ls
end

"""
    get_pieces_dict(s, bs, bbs)

Get closed lists for visited states in dict where key is (m, n).
"""
function get_pieces_dict(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Dict{Tuple{Int, Int}, Set{UInt32}}
    visited = Dict{Tuple{Int, Int}, Set{UInt32}}()
    pieces = pieces_left(s, bs, bbs)
    # Number of pieces in game
    N = pieces[1]
    # Generate independent state spaces
    pieces_order = get_piece_iterator(N)
    for pieces_target in pieces_order
        visited[pieces_target] = Set{UInt32}()
    end
    push!(visited[pieces], s)
    return visited
end

"""
    bfs(s_start, bs, bbs; max_iter=1_000_000_000)

Perform breadth-first-search starting at `s_start` and return closed list of visited states as well as terminal states.
"""
function bfs(s_start::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32}; max_iter=1_000_000_000)
    # Open list of nodes to expand next
    frontier = Vector{UInt32}()
    pushfirst!(frontier, s_start)
    # Closed list of leaf nodes
    leafs = Set{UInt32}()
    # Closed lists of visited nodes per (m, n) tuple
    visited = get_pieces_dict(s_start, bs, bbs)
    ns = zeros(UInt32, 7)
    # Main loop
    for i in 1:max_iter
        # Keep track of progress (flush is for cluster)
        if i % 1_000_000 == 0
            println("$(i)")
            flush(stdout)
        end
        # If search is done
        if isempty(frontier)
            return visited, leafs
        end
        # BFS next expansion
        s = pop!(frontier)
        for roll in 0:4
            if roll == 0
                fill!(ns, 0)
                ns[1] = flip_turn(s, bs, bbs) - bs[32]
            else
                neighbours!(ns, s, roll, bs, bbs)
            end
            for neighbour in ns
                if neighbour == 0
                    break
                end
                # Check for turn change
                neighbour, factor = turn_change(neighbour, bs)
                # Check if terminal state
                if has_won(neighbour, bs, bbs)
                    push!(leafs, neighbour)
                else
                    pieces = pieces_left(neighbour, bs, bbs)
                    if neighbour ∉ visited[pieces]
                        push!(visited[pieces], neighbour)
                        pushfirst!(frontier, neighbour)
                    end
                end
            end
        end
    end
    throw(ErrorException("Iteration limit reached"))
end
"""
    get_value_dict(visited, leafs)

Initialize value map at zeros for non-terminal states, and -100 at terminal states.
"""
function get_value_dict(visited::Dict{Tuple{Int64, Int64}, Set{UInt32}}, leafs::Set{UInt32})::Dict{UInt32, Float64}
    vs = values(visited)
    # Initialize dict at zeros
    V = Dict{UInt32, Float64}(union(vs...) .=> zeros(sum(length.(vs))))
    # Negative because other player wins
    for leaf in leafs
        V[leaf] = -100
    end
    return V
end

"""
    heuristic_initialize(visited, leafs, h)

Initialize value map with heuristic values for non-terminal states, and -100 at terminal states.
"""
function heuristic_initialize(visited::Dict{Tuple{Int64, Int64}, Set{UInt32}}, leafs::Set{UInt32}, h::Function, bs, bbs; N::Int=7)#::Dict{UInt32, Float64}
    vs = union(values(visited)...)
    max_advancement = 15*N
    # Initialize dict at zeros
    V = Dict{UInt32, Float64}()
    z = zeros(max_advancement, max_advancement)
    counts = zeros(max_advancement, max_advancement)
    for s in vs
        V[s] = h(s)
        a1, a2 = advancement(s, bs, bbs; N=N)
        a1 = max_advancement - a1
        a2 = max_advancement - a2
        counts[a1, a2] += 1
        n = counts[a1, a2]
        z[a1, a2] = (z[a1, a2]*(n-1) + V[s])/n
    end
    # Negative because other player wins
    for leaf in leafs
        V[leaf] = -100
    end
    return V, z, counts
end


"""
    random_split(no_leafs, n)

Randomly split state space into n equally sized sets (WARNING: may not produce `n` sets if no_leafs is small).
"""
function random_split(no_leafs::Set{UInt32}, n::Int)::Vector{Set{UInt32}}
    # Randomly split up state space
    no_leafs = shuffle(collect(no_leafs))
    bin_size = ceil(Int, length(no_leafs) / n)
    idxs = push!(collect(1:bin_size:length(no_leafs)), length(no_leafs))
    # Independent state spaces
    subsets = Vector{Set{UInt32}}()
    for n in 1:length(idxs)-1
        # Edge of bin
        offset = n == length(idxs)-1 ? 0 : 1
        push!(subsets, Set(no_leafs[idxs[n]:(idxs[n+1]-offset)]))
    end
    return subsets
end


"""
    get_new_value!(ns, s, bs, bbs, V, Ps)

Find new value for given state `s` by calculating max value averaged over dice rolls. Operation is in place on vector of length 7 `ns`.
"""
function get_new_value!(ns::Vector{UInt32}, s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32}, V::Dict{UInt32, Float64}, Ps::Vector{Float64})::Float64
    nv = zero(Float64)
    # Rolling a 0 (negative because turn change)
    sp = flip_turn(s, bs, bbs)
    sp -= bs[32]
    nv += -Ps[1]*V[sp]
    # Other rolls don't just flip turn
    for roll in 1:4
        neighbours!(ns, s, roll, bs, bbs)
        # Get max across possible actions
        nvv = -Inf
        for neighbour in ns
            # Not all states have 7 possible actions
            if neighbour == 0
                break
            end
            neighbour, factor = turn_change(neighbour, bs)
            # Negate if turn change
            nvv = max(nvv, factor*V[neighbour])
        end
        # Expectation of value (+1 because julia 1 indexes)
        nv += Ps[roll+1]*nvv
    end
    return nv
end


"""
    value_iteration(V, no_leafs, θ, bs, bbs, Ps; max_iter=100)

Perform value iteration over state space with leaf nodes removed (and values initiated). θ is threshold
"""
function value_iteration(V::Dict{UInt32, Float64}, no_leafs::Set{UInt32}, θ::Float64, bs::Vector{UInt32}, bbs::Vector{UInt32}, Ps::Vector{Float64}; max_iter=100, N=7)#::Nothing
    start_time = now()
    # Inline neighbour list to remove memory allocations, max_a is found empirically.
    max_a = 7
    ns = zeros(UInt32, max_a)
    Δ = 0
    zs = []
    max_advancement = 15*N
    # Outer loop until convergence
    for _ in ProgressBar(1:max_iter)
        z = zeros(max_advancement, max_advancement)
        counts = zeros(max_advancement, max_advancement)
        Δ = 0
        # Inner loop sweeping over states
        for s in no_leafs
            v = V[s]
            nv = get_new_value!(ns, s, bs, bbs, V, Ps)
            Δ = max(Δ, abs(v-nv))
            V[s] = nv
            a1, a2 = advancement(s, bs, bbs; N=N)
            a1 = max_advancement - a1
            a2 = max_advancement - a2
            counts[a1, a2] += 1
            n = counts[a1, a2]
            z[a1, a2] = (z[a1, a2]*(n-1) + V[s])/n
        end
        push!(zs, z)
        if Δ < θ
            println("Regular Value iteration took $(now() - start_time)")
            flush(stdout)
            println("Delta: $(Δ)")
            flush(stdout)
            return zs
        end
    end
    println("Regular Value iteration took $(now() - start_time)")
    flush(stdout)
    println("Maximum number of iterations reached (Delta: $(Δ))")
    flush(stdout)
    return zs
end

"""
    value_iteration_parallel(V, no_leafs, θ, bs, bbs, Ps; max_iter=100)

Perform asynchronous value iteration over state space with leaf nodes removed (and values initiated). θ is threshold
"""
function value_iteration_parallel(V::Dict{UInt32, Float64}, no_leafs::Set{UInt32}, θ::Float64, bs::Vector{UInt32}, bbs::Vector{UInt32}, Ps::Vector{Float64}; max_iter=100)::Nothing
    start_time = now()
    # Randomly split up state space
    subsets = random_split(no_leafs, Threads.nthreads());
    # Value iteration outer loop
    Threads.@threads for n in eachindex(subsets)
        ns = zeros(UInt32, 7)
        for _ in 1:max_iter
            Δ = 0
            # Inner loop over subset of state space
            for s in subsets[n]
                v = V[s]
                nv = get_new_value!(ns, s, bs, bbs, V, Ps)
                Δ = max(Δ, abs(v-nv))
                V[s] = nv
            end
            if Δ < θ
                break
            end
        end
    end
    println("Parallel Value iteration took $(now() - start_time)")
    flush(stdout)
    return nothing
end

"""
    get_value_map(visited, leafs, θ; max_iter=100, N=7)

Solve Royal Game of Ur using divided state space. `visited` and `leafs` are dictionaries containins divisions. θ is threshold
"""
function get_value_map(visited::Dict{Tuple{Int64, Int64}, Set{UInt32}}, leafs::Set{UInt32}, θ::Float64; max_iter=100, N::Int=7)#::Dict{UInt32, Float64}
    start_time = now()
    # Initialise base 2 and base 3
    bs, bbs = get_bases()
    # Initialize map
    h = (x) -> h_advancement(x, bs, bbs; N=N)
    V, z, counts = heuristic_initialize(visited, leafs, h, bs, bbs; N=N)
    #V = get_value_dict(visited, leafs)
    println("Initializing map took $(now() - start_time)")
    flush(stdout)
    # Dice probabilities
    Ps = get_Ps()
    # Generate independent state spaces
    pieces_order = get_piece_iterator(N)
    Zs = []
    for pieces_target in pieces_order
        println("Pieces on the board $(pieces_target)")
        flush(stdout)
        current_states = visited[pieces_target]
        current_iteration = now()
        # value_iteration_parallel(V, current_states, θ, bs, bbs, Ps; max_iter=max_iter)
        # value_iteration_parallel(V, current_states, θ, bs, bbs, Ps; max_iter=max_iter)
        # value_iteration_parallel(V, current_states, θ, bs, bbs, Ps; max_iter=max_iter)
        zs = value_iteration(V, current_states, θ, bs, bbs, Ps; max_iter=max_iter)
        push!(Zs, zs...)
        println("Took $(now() - current_iteration)")
        flush(stdout)
    end
    # Full sweep
    all_states = union(values(visited)...)
    zs = value_iteration(V, all_states, θ, bs, bbs, Ps; max_iter=max_iter)
    push!(Zs, zs...)
    println("Full value iteration took $(now() - start_time)")
    flush(stdout)
    return V, Zs
end

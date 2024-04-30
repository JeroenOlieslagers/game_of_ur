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
    get_shared_array_pointers(visited, leafs)

Initialize dict of pointers and array of values, -100 at terminal states.
"""
function get_shared_array_pointers(visited::Dict{Tuple{Int64, Int64}, Set{UInt32}}, leafs::Set{UInt32})::Tuple{Dict{UInt32, Int}, SharedArray{Float64}}
    vs = values(visited)
    L = sum(length.(vs))
    # Pointers from state to index in array
    p = Dict{UInt32, Int}(union(vs...) .=> 1:L)
    # Array containing all values
    V = SharedVector{Float64}(L + length(leafs))
    # Negative because other player wins
    for (n, leaf) in enumerate(leafs)
        id = L + n
        p[leaf] = id
        V[id] = -100
    end
    return p, V
end

function get_reduced_pointer_sets(visited, p, bs, bbs)
    ppp = Dict{Tuple{Int, Int}, Dict{UInt32, Int}}()
    for k in keys(visited)
        println(k)
        pp = Dict{UInt32, Int}()
        for s in visited[k]
            if s ∉ keys(pp)
                pp[s] = p[s]
            end
            for roll in 1:4
                neighs = possible_neighbours(s, roll, bs, bbs)
                for neighbour in neighs
                    neighbour, factor = turn_change(neighbour, bs)
                    if neighbour ∉ keys(pp)
                        pp[neighbour] = p[neighbour]
                    end
                end
            end
        end
        ppp[k] = pp
    end
    return ppp
end

"""
    random_split(no_leafs, n)

Randomly split state space into n equally sized sets.
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
    get_new_value(s, bs, bbs, V, Ps)

Find new value for given state `s` by calculating max value averaged over dice rolls.
"""
function get_new_value(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32}, p::Dict{UInt32, Int}, V::SharedArray{Float64}, Ps::Vector{Float64})::Float64
    nv = 0
    # Rolling a 0 (negative because turn change)
    sp = flip_turn(s, bs, bbs)
    sp -= bs[32]
    nv += -Ps[1]*V[p[sp]]
    # Other rolls don't just flip turn
    for roll in 1:4
        neighs = possible_neighbours(s, roll, bs, bbs)
        # Get max across possible actions
        nvv = -Inf
        for neighbour in neighs
            neighbour, factor = turn_change(neighbour, bs)
            # Negate if turn change
            nvv = max(nvv, factor*V[p[neighbour]])
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
function value_iteration(p::Dict{UInt32, Int}, V::SharedArray{Float64}, no_leafs::Set{UInt32}, θ::Float64, bs::Vector{UInt32}, bbs::Vector{UInt32}, Ps::Vector{Float64}; max_iter=100)::Nothing
    start_time = now()
    Δ = 0
    for _ in 1:max_iter
        Δ = 0
        for s in no_leafs
            pt = p[s]
            v = V[pt]
            nv = get_new_value(s, bs, bbs, p, V, Ps)
            Δ = max(Δ, abs(v-nv))
            V[pt] = nv
        end
        if Δ < θ
            println("Regular Value iteration took $(now() - start_time)")
            flush(stdout)
            println("Delta: $(Δ)")
            flush(stdout)
            return nothing
        end
    end
    println("Regular Value iteration took $(now() - start_time)")
    flush(stdout)
    println("Maximum number of iterations reached (Delta: $(Δ))")
    flush(stdout)
    return nothing
end

"""
    value_iteration_parallel(V, no_leafs, θ, bs, bbs, Ps; max_iter=100)

Perform asynchronous value iteration over state space with leaf nodes removed (and values initiated). θ is threshold
"""
function value_iteration_parallel(p::Dict{UInt32, Int}, V::SharedVector{Float64}, no_leafs::Set{UInt32}, θ::Float64, bs::Vector{UInt32}, bbs::Vector{UInt32}, Ps::Vector{Float64}; max_iter=100)::Nothing
    start_time = now()
    # Randomly split up state space
    subsets = random_split(no_leafs, nprocs()-1);
    # Value iteration loop
    @sync @distributed for n in 1:(nprocs()-1)#Threads.@threads
        for i in 1:max_iter
            Δ = 0
            for s in subsets[n]
                pt = p[s]
                v = V[pt]
                nv = get_new_value(s, bs, bbs, p, V, Ps)
                Δ = max(Δ, abs(v-nv))
                V[pt] = nv
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
    value_iteration_smart(visited, leafs, θ, bs, bbs, Ps; max_iter=100, N=7)

Value iteration for full problem. `visited` and `leafs` are non-intersecting. θ is threshold
"""
function value_iteration_smart(visited::Dict{Tuple{Int64, Int64}, Set{UInt32}}, leafs::Set{UInt32}, θ::Float64, bs::Vector{UInt32}, bbs::Vector{UInt32}; max_iter=100, N=7)::Tuple{Dict{UInt32, Int}, SharedArray{Float64}}
    start_time = now()
    # Initialize map
    #V = get_value_dict(visited, leafs)
    p, V = get_shared_array_pointers(visited, leafs)
    ppp = get_reduced_pointer_sets(visited, p, bs, bbs)
    println("Initializing map took $(now() - start_time)")
    flush(stdout)
    # Dice probabilities
    Ps = get_Ps()
    # Generate independent state spaces
    pieces_order = get_piece_iterator(N)
    for pieces_target in pieces_order
        println("Pieces on the board $(pieces_target)")
        flush(stdout)
        current_states = visited[pieces_target]
        pp = ppp[pieces_target]
        current_iteration = now()
        value_iteration_parallel(pp, V, current_states, θ, bs, bbs, Ps; max_iter=max_iter)
        value_iteration(pp, V, current_states, θ, bs, bbs, Ps; max_iter=max_iter)
        println("Took $(now() - current_iteration)")
    end
    # Full sweep
    value_iteration(p, V, union(values(visited)...), θ, bs, bbs, Ps; max_iter=max_iter)
    println("Full value iteration took $(now() - start_time)")
    flush(stdout)
    return p, V
end
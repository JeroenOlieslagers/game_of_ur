
function initialize_value(h::Function, ind_to_state::Vector{UInt32}, boundaries::Dict{Tuple{Int64, Int64}, Tuple{Int32, Int32}}, bs::Vector{UInt32}, bbs::Vector{UInt32}; N::Int64=7)::Vector{Float64}
    Random.seed!(0)
    N_s = length(ind_to_state)
    V = zeros(Float64, N_s)
    leaf_ind = boundaries[(N, N)][2]
    for n in eachindex(ind_to_state)
        s = ind_to_state[n]
        if n > leaf_ind
            continue
        end
        V[n] = h(s, bs, bbs)
    end
    # for leaf in leaf_nodes
    #     n = state_to_ind[leaf]
    #     V[n] = -100
    # end
    for leaf in (leaf_ind+1):N_s
        V[leaf] = -100
    end
    return V
end

#function bellman_equation(s::Int32, V::Vector{Float64}, neigh_tensor::Array{Int32, 3}, mirror_states::Vector{Int32}, Ps::Vector{Float64})::Float64
function bellman_equation(s, V, neigh_tensor, mirror_states, Ps)::Float64
    # rolling a 0 just flips the state
    sp = mirror_states[s]
    nv = -Ps[1]*V[sp]
    # look at all possible rolls
    for d in 1:4
        nvv = -Inf
        # a maximum of 7 moves
        for i in 1:7
            neigh = neigh_tensor[i, d, s]
            if neigh == 0
                break
            end
            # if < 0, means the turn changed and so need to negate value
            if neigh < 0
                nvv = max(nvv, -V[-neigh])
            else
                nvv = max(nvv, V[neigh])
            end
        end
        nv += Ps[d+1]*nvv
    end
    return nv
end

function iteration!(V::Vector{Float64}, rang::UnitRange{Int32}, neigh_tensor::Array{Int32, 3}, mirror_states::Vector{Int32}, Ps::Vector{Float64})::Nothing
    @batch for s in rang#
        nv = bellman_equation(s, V, neigh_tensor, mirror_states, Ps)
        V[s] = nv
    end
    return nothing
end

function calculate_delta(V::Vector{Float64}, rang::UnitRange{Int32}, neigh_tensor::Array{Int32, 3}, mirror_states::Vector{Int32}, Ps::Vector{Float64})::Float64
    delta = 0.0
    for s in rang#Threads.@threads 
        nv = bellman_equation(s, V, neigh_tensor, mirror_states, Ps)
        delta = max(delta, abs(nv - V[s]))
        V[s] = nv
    end
    return delta
end


function value_iteration!(V::Vector{Float64}, rang::UnitRange{Int32}, neigh_tensor::Array{Int32, 3}, mirror_states::Vector{Int32}, Ps::Vector{Float64}; n_epochs::Int=10, n_iters::Int=100, θ::Float64=1e-3)::Nothing
    for k in 1:n_epochs
        for j in 1:n_iters
            iteration!(V, rang, neigh_tensor, mirror_states, Ps)
        end
        delta = calculate_delta(V, rang, neigh_tensor, mirror_states, Ps)
        if delta < θ
            break
        end
    end
    return nothing
end

function solve_game!(V::Vector{Float64}, boundaries::Dict{Tuple{Int64, Int64}, Tuple{Int32, Int32}}, neigh_tensor::Array{Int32, 3}, mirror_states::Vector{Int32}; n_epochs::Int=10, n_iters::Int=100, θ::Float64=1e-3)
    Ps = get_Ps()
    nms = get_piece_iterator(maximum(keys(boundaries))[2])
    agents = zeros(Float64, length(V), length(nms)+1)
    agents[:, 1] = deepcopy(V)
    for (n, nm) in ProgressBar(enumerate(nms))
        bounds = boundaries[nm]
        rang = bounds[1]:bounds[2]
        value_iteration!(V, rang, neigh_tensor, mirror_states, Ps; n_epochs=n_epochs, n_iters=n_iters, θ=θ)
        agents[:, n+1] = deepcopy(V)
    end
    return agents
end

function solve_game_slow!(V::Vector{Float64}, boundaries::Dict{Tuple{Int64, Int64}, Tuple{Int32, Int32}}, neigh_tensor::Array{Int32, 3}, mirror_states::Vector{Int32}; n_iters::Int=100, θ::Float64=1e-3)
    Ps = get_Ps()
    T = 100
    nms = get_piece_iterator(maximum(keys(boundaries))[2])
    agents = zeros(Float64, length(V), T)
    agents[:, 1] = deepcopy(V)
    agent_nm = [(0, 0) for _ in 1:T]
    agent_counter = zeros(Int64, T)
    agent_t = zeros(Int64, T)
    start_t = now()
    #rang = Int32(1):Int32(length(mirror_states))
    counter = 1
    counter_ = 1
    for nm in ProgressBar(nms)
        bounds = boundaries[nm]
        rang = bounds[1]:bounds[2]
        for k in 1:n_iters
            delta = calculate_delta(V, rang, neigh_tensor, mirror_states, Ps)
            counter_ += 1
            if (k-1) % 5 == 0
                counter += 1
                agents[:, counter] = deepcopy(V)
                agent_nm[counter] = nm
                agent_t[counter] = (now() - start_t).value
                agent_counter[counter] = counter_
            end
            if delta < θ
                break
            end
        end
    end
    println(counter)
    return agents, agent_nm, agent_counter, agent_t
end

# V = initialize_value(h_randn, ind_to_state, boundaries, bs, bbs; N=N);

# @time solve_game!(V, boundaries, neigh_tensor, mirror_states; θ=θ);

# @time agents, agent_nm, agent_counter, agent_t = solve_game_slow!(V, boundaries, neigh_tensor, mirror_states; θ=θ, n_iters=10000);


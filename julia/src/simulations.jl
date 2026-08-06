@inline function fast_roll(cPs::Vector{Float64})::Int
    ran = rand()
    @inbounds if ran < cPs[1]
        return 0
    elseif ran < cPs[2]
        return 1
    elseif ran < cPs[3]
        return 2
    elseif ran < cPs[4]
        return 3
    else
        return 4
    end
end

function value_to_policy(V::AbstractArray, neigh_tensor::Array{Int32, 3})::Matrix{Int32}
    max_neighs, _, max_ind = size(neigh_tensor)
    policy = zeros(Int32, 4, max_ind)

    @inbounds for s_ind in 1:max_ind
        for roll in 1:4
            best_state = Int32(s_ind)
            best_value = -Inf
            for i in 1:max_neighs
                neigh = neigh_tensor[i, roll, s_ind]
                neigh == 0 && break
                value = neigh < 0 ? -V[-neigh] : V[neigh]
                if value > best_value
                    best_value = value
                    best_state = neigh
                end
            end
            policy[roll, s_ind] = best_state
        end
    end

    return policy
end

function simulate_game(
    policy1::Matrix{Int32},
    policy2::Matrix{Int32},
    s_start::Int32,
    mirror_states::Vector{Int32},
    cPs::Vector{Float64};
    max_iter::Int=1000,
)::Bool
    s = s_start
    _, max_ind = size(policy1)
    light_turn = true

    for _ in 1:max_iter
        roll = fast_roll(cPs)
        if roll == 0
            s = mirror_states[s]
            light_turn = !light_turn
            continue
        end

        s = light_turn ? policy1[roll, s] : policy2[roll, s]
        if s < 0
            s = -s
            light_turn = !light_turn
        end
        if s > max_ind
            return !light_turn
        end
    end

    throw(ErrorException("Simulation iteration limit reached"))
end

function simulate_game(
    V1::Vector{Float64},
    V2::Vector{Float64},
    s_start::Int32,
    mirror_states::Vector{Int32},
    cPs::Vector{Float64},
    neigh_tensor::Array{Int32, 3};
    max_iter::Int=1000,
)::Bool
    s = s_start
    max_neighs, _, max_ind = size(neigh_tensor)
    light_turn = true

    for _ in 1:max_iter
        roll = fast_roll(cPs)
        if roll == 0
            s = mirror_states[s]
            light_turn = !light_turn
            continue
        end

        best_states = Int32[]
        best_value = -Inf
        for i in 1:max_neighs
            neigh = neigh_tensor[i, roll, s]
            neigh == 0 && break
            value = if light_turn
                neigh < 0 ? -V1[-neigh] : V1[neigh]
            else
                neigh < 0 ? -V2[-neigh] : V2[neigh]
            end
            if value > best_value
                best_value = value
                empty!(best_states)
                push!(best_states, neigh)
            elseif value == best_value
                push!(best_states, neigh)
            end
        end

        s = rand(best_states)
        if s < 0
            s = -s
            light_turn = !light_turn
        end
        if s > max_ind
            return !light_turn
        end
    end

    throw(ErrorException("Simulation iteration limit reached"))
end

function simulate_game_with_random(
    policy1::Matrix{Int32},
    policy2::Matrix{Int32},
    ϵ1::Float64,
    ϵ2::Float64,
    s_start::Int32,
    mirror_states::Vector{Int32},
    cPs::Vector{Float64},
    neigh_tensor::Array{Int32, 3};
    max_iter::Int=1000,
)::Bool
    s = s_start
    _, max_ind = size(policy1)
    light_turn = true

    for _ in 1:max_iter
        roll = fast_roll(cPs)
        if roll == 0
            s = mirror_states[s]
            light_turn = !light_turn
            continue
        end

        explore = rand() < (light_turn ? ϵ1 : ϵ2)
        if explore
            neighs = neigh_tensor[:, roll, s]
            zero_idx = findfirst(iszero, neighs)
            s = zero_idx === nothing ? rand(neighs) : rand(neighs[1:zero_idx - 1])
        else
            s = light_turn ? policy1[roll, s] : policy2[roll, s]
        end

        if s < 0
            s = -s
            light_turn = !light_turn
        end
        if s > max_ind
            return !light_turn
        end
    end

    throw(ErrorException("Simulation iteration limit reached"))
end

function duel(
    n_games::Int64,
    policy1::Matrix{Int32},
    policy2::Matrix{Int32},
    s_start::Int32,
    mirror_states::Vector{Int32},
)::Int64
    cPs = cumsum(get_Ps())
    return sum(simulate_game(policy1, policy2, s_start, mirror_states, cPs) for _ in 1:n_games)
end

function duel(
    n_games::Int64,
    V1::Vector{Float64},
    V2::Vector{Float64},
    s_start::Int32,
    mirror_states::Vector{Int32},
    neigh_tensor::Array{Int32, 3},
)::Int64
    cPs = cumsum(get_Ps())
    return sum(simulate_game(V1, V2, s_start, mirror_states, cPs, neigh_tensor) for _ in 1:n_games)
end

function duel_with_random(
    n_games::Int64,
    policy1::Matrix{Int32},
    policy2::Matrix{Int32},
    ϵ1::Float64,
    ϵ2::Float64,
    s_start::Int32,
    mirror_states::Vector{Int32},
    neigh_tensor::Array{Int32, 3},
)::Int64
    cPs = cumsum(get_Ps())
    return sum(
        simulate_game_with_random(policy1, policy2, ϵ1, ϵ2, s_start, mirror_states, cPs, neigh_tensor)
        for _ in 1:n_games
    )
end

function tournament(
    n_games::Int64,
    agents::Vector{Matrix{Int32}},
    s_start::Int32,
    mirror_states::Vector{Int32},
)::Matrix{Int64}
    results = zeros(Int64, length(agents), length(agents))
    for i in eachindex(agents), j in eachindex(agents)
        i == j && continue
        results[i, j] = duel(n_games, agents[i], agents[j], s_start, mirror_states)
    end
    return results
end

function tournament(
    n_games::Int64,
    agents::Vector{Vector{Float64}},
    s_start::Int32,
    mirror_states::Vector{Int32},
    neigh_tensor::Array{Int32, 3},
)::Matrix{Int64}
    results = zeros(Int64, length(agents), length(agents))
    for i in eachindex(agents), j in eachindex(agents)
        i == j && continue
        results[i, j] = duel(n_games, agents[i], agents[j], s_start, mirror_states, neigh_tensor)
    end
    return results
end

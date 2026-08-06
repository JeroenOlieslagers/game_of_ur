function initialize_value(
    h::Function,
    ind_to_state::Vector{UInt32},
    boundaries::Dict{Tuple{Int64, Int64}, Tuple{Int32, Int32}},
    bs::Vector{UInt32},
    bbs::Vector{UInt32};
    N::Int64=7,
)::Vector{Float64}
    Random.seed!(0)
    V = zeros(Float64, length(ind_to_state))
    leaf_ind = boundaries[(N, N)][2]

    for n in eachindex(ind_to_state)
        if n <= leaf_ind
            V[n] = h(ind_to_state[n], bs, bbs)
        else
            V[n] = -100
        end
    end
    return V
end

function bellman_equation(
    s::Int32,
    V::Vector{Float64},
    neigh_tensor::Array{Int32, 3},
    mirror_states::Vector{Int32},
    Ps::Vector{Float64},
)::Float64
    nv = -Ps[1] * V[mirror_states[s]]
    for roll in 1:4
        best = -Inf
        for i in axes(neigh_tensor, 1)
            neigh = neigh_tensor[i, roll, s]
            neigh == 0 && break
            best = max(best, neigh < 0 ? -V[-neigh] : V[neigh])
        end
        nv += Ps[roll + 1] * best
    end
    return nv
end

function iteration!(
    V::Vector{Float64},
    rang::UnitRange{Int32},
    neigh_tensor::Array{Int32, 3},
    mirror_states::Vector{Int32},
    Ps::Vector{Float64},
)::Nothing
    for s in rang
        V[s] = bellman_equation(Int32(s), V, neigh_tensor, mirror_states, Ps)
    end
    return nothing
end

function calculate_delta(
    V::Vector{Float64},
    rang::UnitRange{Int32},
    neigh_tensor::Array{Int32, 3},
    mirror_states::Vector{Int32},
    Ps::Vector{Float64},
)::Float64
    delta = 0.0
    for s in rang
        nv = bellman_equation(Int32(s), V, neigh_tensor, mirror_states, Ps)
        delta = max(delta, abs(nv - V[s]))
        V[s] = nv
    end
    return delta
end

function value_iteration!(
    V::Vector{Float64},
    rang::UnitRange{Int32},
    neigh_tensor::Array{Int32, 3},
    mirror_states::Vector{Int32},
    Ps::Vector{Float64};
    n_epochs::Int=10,
    n_iters::Int=100,
    θ::Float64=1e-3,
)::Nothing
    for _ in 1:n_epochs
        for _ in 1:n_iters
            iteration!(V, rang, neigh_tensor, mirror_states, Ps)
        end
        calculate_delta(V, rang, neigh_tensor, mirror_states, Ps) < θ && break
    end
    return nothing
end

function solve_game!(
    V::Vector{Float64},
    boundaries::Dict{Tuple{Int64, Int64}, Tuple{Int32, Int32}},
    neigh_tensor::Array{Int32, 3},
    mirror_states::Vector{Int32};
    n_epochs::Int=10,
    n_iters::Int=100,
    θ::Float64=1e-3,
)
    Ps = get_Ps()
    nms = get_piece_iterator(maximum(last.(keys(boundaries))))
    agents = zeros(Float64, length(V), length(nms) + 1)
    agents[:, 1] = V

    for (n, nm) in enumerate(nms)
        bounds = boundaries[nm]
        value_iteration!(V, bounds[1]:bounds[2], neigh_tensor, mirror_states, Ps;
                         n_epochs=n_epochs, n_iters=n_iters, θ=θ)
        agents[:, n + 1] = V
    end

    return agents
end

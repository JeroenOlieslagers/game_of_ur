#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "GameOfUr.jl"))
using .GameOfUr

function main()
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
    θ = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-3

    bs, bbs = get_bases()
    s_start = start_state(bs, bbs; N=N)

    println("Searching state space for N=$N")
    visited, leaf_nodes = bfs(s_start, bs, bbs)
    ind_to_state, state_to_ind, boundaries = get_conversions(visited, leaf_nodes; N=N)
    states = setdiff(union(values(visited)...), leaf_nodes)
    neigh_tensor, mirror_states = get_neigh_tensor(states, state_to_ind, bs, bbs)

    println("Solving with θ=$θ")
    V = initialize_value(h_0, ind_to_state, boundaries, bs, bbs; N=N)
    solve_game!(V, boundaries, neigh_tensor, mirror_states; θ=θ)

    start_index = state_to_ind[s_start]
    println("Chance of first player winning: $((V[start_index] + 100) / 2)%")
end

main()

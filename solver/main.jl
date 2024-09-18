using Dates
using Random
using StatsBase
using JLD2
using ProgressBars
using Random
using Polyester
include("game_logic.jl")
include("search.jl")
include("value_iteration.jl")
include("matrices.jl")
include("heuristics.jl")

# if visited states have been pre-computed
LOAD_VISITED = true
# if matrices have been pre-computed
LOAD_MATRICES = true
# if value map has been pre-computed
LOAD_VMAP = true

# number of pieces, 7 is full game
N = 7;
# convergence threshold
θ = 0.001;

# arrays of first 32 numbers in base 2 and base 3
bs, bbs = get_bases();
# initial board state
s_start = start_state(bs, bbs; N=N);

if LOAD_MATRICES || ~LOAD_VMAP
    if LOAD_MATRICES
        ind_to_state = load("jld2_files/ind_to_state.jld2")["data"]
        state_to_ind = load("jld2_files/state_to_ind.jld2")["state_to_ind"]
        boundaries = load("jld2_files/boundaries.jld2")["boundaries"]
        neigh_tensor = load("jld2_files/neigh_tensor.jld2")["data"]
        mirror_states = load("jld2_files/mirror_states.jld2")["data"]
    else
        # This computes the entire state space (self-other representation)
        if LOAD_VISITED
            visited = load("jld2_files/visited.jld2")["visited"]
            leaf_nodes = load("jld2_files/leaf_nodes.jld2")["leaf_nodes"]
        else
            visited, leaf_nodes = bfs(s_start, bs, bbs)
        end
        ind_to_state, state_to_ind, boundaries = get_conversions(visited, leaf_nodes);
        states = setdiff(union(values(visited)...), leaf_nodes)
        neigh_tensor, mirror_states = get_neigh_tensor(states, state_to_ind);
    end
end


# This performs value iteration
if LOAD_VMAP
    V = load("jld2_files/V.jld2")["data"]
else
    V = initialize_value(h_randn, ind_to_state, boundaries, bs, bbs)
    solve_game!(V, boundaries, neigh_tensor, mirror_states; θ=θ)
end

# Print chance of winning as light from start state
println("Chance of light winning: $((V[s_start]+100)/2)%")



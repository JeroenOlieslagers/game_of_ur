using Dates
using Random
using JLD2
include("game_logic.jl")
include("search.jl")
include("value_iteration.jl")

# if visited states have been pre-computed
LOAD_VISITED = false
# if value map has been pre-computed
LOAD_VMAP = false

# number of pieces, 7 is full game
N = 7;
# convergence threshold
θ = 0.001;

# arrays of first 32 numbers in base 2 and base 3
bs, bbs = get_bases();
# initial board state
s_start = start_state(bs, bbs; N=N);

# This computes the entire state space (self-other representation)
if LOAD_VISITED
    visited = load("visited.jld2")["visited"]
    leafs = load("leafs.jld2")["leaf_nodes"]
else
    visited, leafs = bfs(s_start, bs, bbs)
end

# This performs value iteration
if LOAD_VMAP
    V = load("V.jld2")["V"]
else
    V = get_value_map(visited, leafs, θ; N=N)
end

# Print chance of winning as light from start state
println("Chance of light winning: $((V[s_start]+100)/2)%")
using Dates
using Random
using JLD2
using Plots
include("../solver/game_logic.jl")
include("../solver/search.jl")
include("../solver/value_iteration.jl")
include("../solver/heuristics.jl")

bs, bbs = get_bases();
s_start = start_state(bs, bbs);

visited = load("visited.jld2")["visited"]
all_visited = union(values(visited)...)
leafs = load("leafs.jld2")["leaf_nodes"]
V = load("V.jld2")["V"]


N = 4
s_start = start_state(bs, bbs; N=N);
visited, leafs = bfs(s_start, bs, bbs)
VV, ZZs = get_value_map(visited, leafs, θ; N=N)

h = (x) -> h_advancement(x, bs, bbs; N=N)
V, z, counts = heuristic_initialize(visited, leafs, h, bs, bbs)
V = get_value_dict(visited, leafs)
zs = value_iteration(V, union(values(visited)...), θ, bs, bbs, get_Ps(); max_iter=100, N=N)

anim = @animate for i in eachindex(zs)
    surface(1:60, 1:60, zs[i], yflip=true, xflip=true, zlim=(-110, 110), cbar=false, zlabel="Value", xlabel="Dark remaining", ylabel="              Light remaining", colorbar_title="\nValue", title="Iteration $(i)", dpi=100, xticks=[1, 50], yticks=[1, 50], zticks=[-100, 0, 100])
end

gif(anim, "training_value_4_pieces_heur_init.gif", fps=20)

max_advancement = 105

countss = []
zs = []
for a in ProgressBar(get_piece_iterator(7))
    counts = zeros(max_advancement, max_advancement)
    z = zeros(max_advancement, max_advancement)
    for s in visited[a]
        a1, a2 = advancement(s, bs, bbs)
        a1 = max_advancement - a1
        a2 = max_advancement - a2
        counts[a1, a2] += 1
        #N = counts[a1, a2]
        N = total_counts[a1, a2]
        z[a1, a2] = (z[a1, a2]*(N) + V[s])/N
    end
    push!(countss, counts)
    push!(zs, z)
end

total_counts = sum(countss)
anim = @animate for i in eachindex(countss)
    heatmap(1:105, 1:105, countss[i] ./ total_counts, yflip=true, xflip=true, clim=(0, 1), xlabel="Dark remaining", ylabel="Light remaining", colorbar_title="\nProportion of all states", title="(m, n) = $(get_piece_iterator(7)[i])", dpi=300, size=(350, 300), xticks=[1, 50, 100], yticks=[1, 50, 100], right_margin=5Plots.mm)
end

gif(anim, "all_counts.gif", fps=3)

anim = @animate for i in eachindex(zs)
    surface(1:105, 1:105, sum(zs[1:i]), yflip=true, xflip=true, zlim=(-110, 110), cbar=false, zlabel="Value", xlabel="Dark remaining", ylabel="              Light remaining", colorbar_title="\nValue", title="(m, n) = $(get_piece_iterator(7)[i])", dpi=300, xticks=[1, 50, 100], yticks=[1, 50, 100], zticks=[-100, 0, 100])
end

gif(anim, "true_value_evolution.gif", fps=3)


s_start = start_state(bs, bbs; N=4);
V_h, z, counts = heuristic_initialize(visited, leafs, h_advancement, bs, bbs; N=4)
heuristic = (x) -> greedy(x[1], x[2], V_h, bs)
remaining = []
a = simulate_game(heuristic, ran, s_start, bs, bbs; remaining=remaining)

xs = [remaining[i][1] for i in eachindex(remaining)]
ys = [remaining[i][2] for i in eachindex(remaining)]
zs = [z[remaining[i][1], remaining[i][2]] for i in eachindex(remaining)]

plot(xs, ys, zs)
scatter!(xs[1], ys[1], zs[1], c=:red)

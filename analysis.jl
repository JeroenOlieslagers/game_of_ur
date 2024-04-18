using JLD2


function min_max(visited, dark_turns, rolls, V, bs, bbs)
    N_games = length(visited)
    mins = []
    maxs = []
    roll_vs = []
    diffs = []
    for j in ProgressBar(1:N_games)
        _visited = visited[j]
        _dark_turns = dark_turns[j]
        _rolls = rolls[j]
        N = length(_visited)-1
        _mins = zeros(N)
        _maxs = zeros(N)
        _roll_vs = zeros(N, 5)
        _diffs = zeros(N, 2)
        for i in 1:N
            s = _visited[i]
            roll = _rolls[i]
            factor = -1
            # skill
            neighs = possible_neighbours(s, roll, bs, bbs)
            v = ((_dark_turns[i] != _dark_turns[i+1])*2 - 1)*V[_visited[i+1]]
            diff = 0
            for sp in neighs
                neigh = sp
                f = factor*(1 - check_bit(neigh, 32)*2)
                if check_bit(sp, 32)
                    neigh = sp-bs[32]
                end
                vp = f*V[neigh]
                diff = maximum((diff, vp-v))
            end
            _diffs[i, :] = [diff, length(neighs)]
            # max and min
            __mins = zeros(5)
            __maxs = zeros(5)
            for r in 0:4
                neighs = possible_neighbours(s, r, bs, bbs)
                vss = []
                for sp in neighs
                    neigh = sp
                    f = factor*(1 - check_bit(neigh, 32)*2)
                    if check_bit(sp, 32)
                        neigh = sp-bs[32]
                    end
                    vp = f*V[neigh]
                    push!(vss, vp)
                end
                __mins[r+1] = minimum(vss)
                __maxs[r+1] = maximum(vss)
            end
            _mins[i] = minimum(__mins)
            _maxs[i] = maximum(__maxs)
            _roll_vs[i, :] = __maxs
        end
        # push!(mins, _mins)
        # push!(maxs, _maxs)
        push!(roll_vs, _roll_vs)
        push!(diffs, _diffs)
    end
    return roll_vs, diffs#mins, maxs, 
end

function advantages(visited, moves, dark_turns, rolls, V, bs, bbs)
    captures = []
    capture_v = []
    for n in ProgressBar(eachindex(visited))
        N = length(visited[n])-1
        for i in 1:N
            s = visited[n][i]
            roll = rolls[n][i]
            move = moves[n][i]
            dt = dark_turns[n][i]
            neighs, as = possible_neighs_moves(s, roll, bs, bbs)
            factor = dt*2 - 1
            vs = []
            v = (dark_turns[n][i+1]*2 - 1)*V[visited[n][i+1]]
            for (sp, ap) in zip(neighs, as)
                neigh = sp
                f = factor*(1 - check_bit(neigh, 32)*2)
                if check_bit(sp, 32)
                    neigh = sp-bs[32]
                end
                if ap[1] == 8 && ap[3] && abs(v) > 80
                    push!(vs, v - f*V[neigh])
                end
            end
            if length(vs) > 0
                push!(captures, (move[1] == 8) && move[3] && abs(v) > 80)
                if dt
                    if -abs(maximum(vs)) < -5
                        println(s)
                        println(roll)
                        println(move)
                    end
                    push!(capture_v, -abs(maximum(vs)))
                else
                    if -abs(minimum(vs)) < -5
                        println(s)
                        println(roll)
                        println(move)
                    end
                    push!(capture_v, -abs(minimum(vs)))
                end
            end
        end
    end
    return captures, capture_v
end

subset = 1:100000

roll_vs, diffs = min_max(visitedd, dark_turns, rolls, V, bs, bbs);
from_centers_and_capture_and_winlose, from_center_and_capture_and_winlose_v = advantages(visited[subset], moves[subset], dark_turns[subset], rolls[subset], V, bs, bbs);

plot(layout=(2,2), grid=false, foreground_color_legend=nothing, dpi=300)
histogram!(capture_v[capture_v .< 0]/2, sp=1, title="Captures", label="N="*string(round(Int, length(capture_v)/10000)/100)*"M f="*string(round(Int, sum(capture_v .< 0)/1000))*"k", xlabel=latexstring("\\Delta")*"%", bins=100)
histogram!((rosette_v[rosette_v .< 0])[1:1000000]/2, sp=2, title="Move to rosette", label="N="*string(round(Int, length(rosette_v)/10000)/100)*"M f="*string(round(Int, sum(rosette_v .< 0)/1000))*"k", xlabel=latexstring("\\Delta")*"%", bins=100)
histogram!(center_v[center_v .< 0]/2, sp=3, title="Move to center", label="N="*string(round(Int, length(center_v)/1000))*"k f="*string(round(Int, sum(center_v .< 0)/1000))*"k", xlabel=latexstring("\\Delta")*"%", bins=100)
histogram!((from_center_v[from_center_v .< 0])[1:1000000]/2, sp=4, title="Move from center", label="N="*string(round(Int, length(from_center_v)/10000)/100)*"M f="*string(round(Int, sum(from_center_v .< 0)/10000)/100)*"M", xlabel=latexstring("\\Delta")*"%", bins=100)

histogram((from_center_and_capture_v[from_center_and_capture_v .< 0])/2, title="Move from center AND capture", label="N="*string(round(Int, length(from_center_and_capture_v)/1000))*"k f="*string(round(Int, sum(from_center_and_capture_v .< 0)/1000))*"k", xlabel=latexstring("\\Delta")*"%", bins=100)
histogram((from_center_and_capture_and_winlose_v[from_center_and_capture_and_winlose_v .< 0])/2, title="Move from center AND capture AND |V|>80", label="N="*string(round(Int, length(from_center_and_capture_and_winlose_v)/1000))*"k f="*string(round(Int, sum(from_center_and_capture_and_winlose_v .< 0))), xlabel=latexstring("\\Delta")*"%", bins=100)

savefig("advantages.png")

#@save "subsets.jld2" subsets

visited = load("visited_half_smart.jld2")["visited"];
leafs = load("leafs.jld2")["leafs"];
#leaf_nodes = load("leaf_nodes_half_smart.jld2")["leaf_nodes"];
V = load("V_half_smart.jld2")["V"];


bs = UInt32.(2 .^ (0:31));
bbs = UInt32.(3 .^ (0:7));

s = start_state_int(bs, bbs);



include("../solver/game_logic.jl")
using ProgressBars
using StatsBase

function simulate_game(π₁, π₂, s_start, bs, bbs; max_iter=100000, visited=nothing, dark_turns=nothing, rolls=nothing, moves=nothing, remaining=nothing)
    Ps = [binomial(4, k) for k in 0:4]*(0.5^4)
    current_dark = false
    s = s_start
    a = nothing
    if visited !== nothing
        push!(visited, s)
    end
    if dark_turns !== nothing
        push!(dark_turns, current_dark)
    end
    if remaining !== nothing
        push!(remaining, 15*7 .- advancement(s, bs, bbs))
    end
    for i in 1:max_iter
        roll = wsample(Ps) - 1
        if current_dark
            s, a = π₂(neighbours_moves(s, roll, bs, bbs))
        else
            s, a = π₁(neighbours_moves(s, roll, bs, bbs))
        end
        if check_bit(s, 32)
            s -= bs[32]
            current_dark = !current_dark
        end
        if visited !== nothing
            push!(visited, s)
        end
        if dark_turns !== nothing
            push!(dark_turns, current_dark)
        end
        if rolls !== nothing
            push!(rolls, roll)
        end
        if moves !== nothing
            push!(moves, a)
        end
        if remaining !== nothing
            push!(remaining, 15*7 .- advancement(s, bs, bbs))
        end
        if has_won(s, bs, bbs)
            return current_dark
        end
    end
    throw(ErrorException("Max iter reached"))
end

function greedy(neighs, moves, V, bs)
    min = Inf
    s_p = UInt32(0)
    a_p = (UInt8(0), UInt8(0), false, false)
    for (neigh, move) in zip(neighs, moves)
        v = nothing
        if check_bit(neigh, 32)
            v = -V[neigh-bs[32]]
        else
            v = V[neigh]
        end
        if v < min
            min = v
            s_p = neigh
            a_p = move
        end
    end
    return s_p, a_p
end

function greedyy(neighs, moves, V, bs)
    min = Inf
    s_p = UInt32(0)
    a_p = (UInt8(0), UInt8(0), false, false)
    for (neigh, move) in zip(neighs, moves)
        v = nothing
        if check_bit(neigh, 32)
            v = V[neigh-bs[32]]
        else
            v = -V[neigh]
        end
        if v < min
            min = v
            s_p = neigh
            a_p = move
        end
    end
    return s_p, a_p
end

function ran(x)
    neighs, moves = x
    i = rand(1:length(neighs))
    return neighs[i], moves[i]
end

bs = UInt32.(2 .^ (0:31));
bbs = UInt32.(3 .^ (0:7));

s = start_state(bs, bbs);

#@btime simulate_game(ran, ran, s, bs, bbs)

s=0x08800000
s=0x04401116

opt = (x) -> greedy(x[1], x[2], V, bs)
optt = (x) -> greedyy(x[1], x[2], V3, bs)

V_h = heuristic_initialize(visited, leafs, h_advancement, bs, bbs)
heuristic = (x) -> greedy(x[1], x[2], V_h, bs)

n_games = 10000
results = []
visitedd = []
dark_turns = []
vss = []
rolls = []
moves = []
for i in ProgressBar(1:n_games)
    _visited = []
    _dark_turns = []
    _rolls = []
    _moves = []
    push!(results, simulate_game(optt, opt, s, bs, bbs; visited=_visited, dark_turns=_dark_turns, rolls=_rolls, moves=_moves))
    push!(visitedd, _visited)
    push!(dark_turns, _dark_turns)
    push!(rolls, _rolls)
    push!(moves, _moves)
    _vss = zeros(length(_visited))
    for i in eachindex(_visited)
        if _dark_turns[i]
            _vss[i] = V[_visited[i]]
        else
            _vss[i] = -V[_visited[i]]
        end
    end
    push!(vss, _vss)
end

draw_boards(reverse(k[sp[end-39:end]]), bs, bbs; titles=round.(reverse(10000 * v[sp[end-39:end]] / sum(v)))/100)

plot(cumsum(v[reverse(sp)][1:end])[1:1000:end] ./ sum(v), xlabel="Rank (x1000)", ylabel="CDF", label=false, dpi=300)

occupied = zeros(14)
for v in ProgressBar(k)
    # p_on = 0
    # for i in 1:8
    #     p_on += check_trit(v, i, bs, bbs) > 0
    # end
    # for i in 14:25
    #     p_on += check_bit(v, i)
    # end
    # push!(on_the_board, p_on)
    # p_home = 0
    # for i in 26:28
    #     p_home += 2^(28-i)*check_bit(v, i)
    # end
    # for i in 29:31
    #     p_home += 2^(31-i)*check_bit(v, i)
    # end
    # push!(home, p_home)
    p_occupied = zeros(14)
    for i in 14:17
        occupied[i-13] += check_bit(v, i)
    end
    for i in 1:8
        occupied[i+4] += check_trit(v, i, bs, bbs) > 0
    end
    for i in 18:19
        occupied[i-17+12] += check_bit(v, i)
    end
end

to_center_rosette_rolls = []
for i in ProgressBar(1:100000)
    for (n, m) in enumerate(moves[i])
        if m[2] == 8
            push!(to_center_rosette_rolls, rolls[i][n])
        end
    end
end
histogram(to_center_rosette_rolls[1:100:end], normalize=true, bins=4, xticks=([1.5, 2.5, 3.5, 4.5], [1, 2, 3, 4]), label=nothing, xlabel="Roll", title="Moves leading to center rosette")



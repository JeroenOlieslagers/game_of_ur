using Accessors
using DataStructures
using ProgressBars
using StatsBase

# Number of pieces
N = 3

wp = Tuple(zeros(Bool, 14))
wh = (UInt8(N), UInt8(0))
bp = Tuple(zeros(Bool, 14))
bh = (UInt8(N), UInt8(0))
roll = UInt8(0)
turn = UInt8(2)

s_starts = make_move((wp, bp, wh, bh, roll, turn), (0, 0, 0))
s_start = (wp, bp, wh, bh, UInt8(1))

function possible_moves(s; r=nothing)
    roll = nothing
    if r === nothing
        roll = s[5]
    else
        roll = r
    end
    if roll == 0
        return []
    end
    player = s[end]
    pieces = s[player]
    other_pieces = s[1 + (player % 2)]
    homes = s[player+2]
    L = length(pieces)
    moves = Vector{Tuple{UInt8, UInt8, UInt8}}()
    if homes[1] > 0 && pieces[roll] == false
        push!(moves, (player, 0, roll))
    end
    for p_loc in findall(pieces)
        p_target = p_loc + roll
        if p_target == 8 && other_pieces[p_target]
            continue
        end
        if p_target > L
            if p_target == L+1
                push!(moves, (player, p_loc, p_target))
            end
        elseif pieces[p_target] == false
            push!(moves, (player, p_loc, p_target))
        end
    end
    return moves
end

function make_move(s, move)
    player, p_loc, p_target = move
    wp, bp, wh, bh, roll, turn = s
    next_player = UInt8(1 + (turn % 2))
    if move == (0, 0, 0)
        return Tuple((wp, bp, wh, bh, UInt8(r), next_player) for r in 0:4)
    end
    if player != turn
        throw(ErrorException("State and move don't agree on whose turn it is."))
    end
    pieces = s[player]
    homes = s[player+2]
    L = length(pieces)
    if p_loc == 0
        homes = @set homes[1] -= UInt8(1)
        pieces = @set pieces[p_target] = true
    elseif p_target == L+1
        homes = @set homes[2] += UInt8(1)
        pieces = @set pieces[p_loc] = false
    else
        pieces = @set pieces[p_loc] = false
        pieces = @set pieces[p_target] = true
    end
    if p_target == 4 || p_target == 8 || p_target == 14
        next_player = player
    end
    if player == 1
        return Tuple((pieces, bp, homes, bh, UInt8(r), next_player) for r in 0:4)
    else
        return Tuple((wp, pieces, wh, homes, UInt8(r), next_player) for r in 0:4)
    end
end

function make_move_no_roll(s, move)
    player, p_loc, p_target = move
    wp, bp, wh, bh, turn = s
    next_player = UInt8(1 + (turn % 2))
    if move == (0, 0, 0)
        return (wp, bp, wh, bh, next_player)
    end
    if player != turn
        throw(ErrorException("State and move don't agree on whose turn it is."))
    end
    pieces = s[player]
    homes = s[player+2]
    L = length(pieces)
    if p_loc == 0
        homes = @set homes[1] -= UInt8(1)
        pieces = @set pieces[p_target] = true
    elseif p_target == L+1
        homes = @set homes[2] += UInt8(1)
        pieces = @set pieces[p_loc] = false
    else
        pieces = @set pieces[p_loc] = false
        pieces = @set pieces[p_target] = true
    end
    if p_target == 4 || p_target == 8 || p_target == 14
        next_player = player
    end
    if player == 1
        return (pieces, bp, homes, bh, next_player)
    else
        return (wp, pieces, wh, homes, next_player)
    end
end

function has_won(s)
    if s[3][1] == 0 && sum(s[1]) == 0
        return 1
    elseif s[4][1] == 0 && sum(s[2]) == 0
        return 2
    else
        return 0
    end
end

function exhaustive_tree(s_starts; n_iter=100000)
    L = length(s_starts[1][1])
    p_type = NTuple{L, Bool}
    h_type = Tuple{UInt8, UInt8}
    s_type = Tuple{p_type, p_type, h_type, h_type, UInt8, UInt8}
    move_type = Tuple{UInt8, UInt8, UInt8}
    tree = DefaultDict{s_type, Vector{Tuple{move_type, NTuple{5, s_type}}}}([])
    frontier = Vector{s_type}()
    visited = Set{s_type}()
    # win_black = Vector{s_type}()
    # win_white = Vector{s_type}()
    for s_start in s_starts
        pushfirst!(frontier, s_start)
    end
    for i in ProgressBar(1:n_iter)
        if isempty(frontier)
            return tree, visited
        end
        s = pop!(frontier)
        winner = has_won(s)
        if winner > 0
            # if winner == 1
            #     push!(win_black, s)
            # else
            #     push!(win_white, s)
            # end
            continue
            tree[s] = []
        end
        #for r in 0:4
        moves = possible_moves(s)
        if isempty(moves)
            moves = [(UInt8(0), UInt8(0), UInt8(0))]
        end
        for move in moves
            s_nexts = make_move(s, move)
            push!(tree[s], (move, s_nexts))
            for s_next in s_nexts
                if s_next ∉ visited
                    push!(visited, s)
                    pushfirst!(frontier, s_next)
                end
            end
        end
        #end
    end
    return tree, visited
    throw(ErrorException("Maximum number of iterations reached"))
end

function exhaustive_tree_no_dice_in_state(s_start; n_iter=100000)
    L = length(s_start[1])
    p_type = NTuple{L, Bool}
    h_type = Tuple{UInt8, UInt8}
    s_type = Tuple{p_type, p_type, h_type, h_type, UInt8}
    #move_type = Tuple{UInt8, UInt8, UInt8}
    #tree = DefaultDict{s_type, Vector{Tuple{move_type, NTuple{5, s_type}}}}([])
    tree = DefaultDict{s_type, NTuple{5, Vector{s_type}}}(([], [], [], [], []))
    frontier = Vector{s_type}()
    visited = Set{s_type}()
    #win_black = Vector{s_type}()
    #win_white = Vector{s_type}()
    pushfirst!(frontier, s_start)
    for i in ProgressBar(1:n_iter)
        if isempty(frontier)
            return tree, visited#, win_black, win_white
        end
        s = pop!(frontier)
        winner = has_won(s)
        if winner > 0
            # if winner == 1
            #     push!(win_black, s)
            # else
            #     push!(win_white, s)
            # end
            continue
            tree[s] = []
        end
        for r in 0:4
            moves = possible_moves(s; r=r)
            if isempty(moves)
                moves = [(UInt8(0), UInt8(0), UInt8(0))]
            end
            for move in moves
                s_next = make_move_no_roll(s, move)
                push!(tree[s][r+1], s_next)
                if s_next ∉ visited
                    push!(visited, s_next)
                    pushfirst!(frontier, s_next)
                end
            end
        end
    end
    return tree, visited#, win_black, win_white
    throw(ErrorException("Maximum number of iterations reached"))
end

function value_iteration(tree, visited, γ, θ; max_iter=100)
    s_type = typeof(first(visited))
    V = DefaultDict{s_type, Float64}(0.0)
    Ps = [binomial(4, k) for k in 0:4]*(0.5^4)
    for i in ProgressBar(1:max_iter)
        Δ = 0
        for s in visited
            v = V[s]
            nv = 0
            for (a, ss) in tree[s]
                nvv = 0
                for (n, sp) in enumerate(ss)
                    winner = has_won(sp)
                    R = winner == 1 ? -100 : winner == 2 ? 100 : 0
                    nvv += Ps[n]*(R + γ*V[sp])
                end
                nv = max(nv, nvv)
            end
            Δ = max(Δ, abs(v-nv))
            V[s] = nv
        end
        if Δ < θ
            return V
        end
    end
    throw(ErrorException("Maximum number of iterations reached"))
end

function value_iteration_averaged(tree, visited, γ, θ; max_iter=100)
    s_type = typeof(first(visited))
    V = DefaultDict{s_type, Float64}(0.0)
    Ps = [binomial(4, k) for k in 0:4]*(0.5^4)
    for i in ProgressBar(1:max_iter)
        Δ = 0
        for s in visited
            v = V[s]
            nv = 0
            for (n, ls) in enumerate(tree[s])
                nvv = 0
                for sp in ls
                    winner = has_won(sp)
                    R = winner == 1 ? 100 : winner == 2 ? -100 : 0
                    nvv = max(nvv, R + γ*V[sp])
                end
                nv += Ps[n]*nvv
            end
            Δ = max(Δ, abs(v-nv))
            V[s] = nv
        end
        if Δ < θ
            return V
        end
    end
    throw(ErrorException("Maximum number of iterations reached"))
end

function greedy_policy(V, tree, γ)
    s_type = typeof(first(keys(V)))
    a_type = typeof(first(values(tree))[1][1])
    policy = Dict{s_type, a_type}()
    Ps = [binomial(4, k) for k in 0:4]*(0.5^4)
    for s in keys(V)
        if length(tree[s]) > 0
            max_v = -10000000000
            max_a = (UInt8(0), UInt8(0), UInt8(0))
            for (a, ss) in tree[s]
                v = 0
                for (n, sp) in enumerate(ss)
                    winner = has_won(sp)
                    R = winner == 1 ? -100 : winner == 2 ? 100 : 0
                    v += Ps[n]*(R + γ*V[sp])
                end
                if v > max_v
                    max_a = a
                    max_v = v
                end
            end
            policy[s] = max_a
        end
    end
    return policy
end

function random_policy(V, tree)
    s_type = typeof(first(keys(V)))
    a_type = typeof(first(values(tree))[1][1])
    policy = Dict{s_type, a_type}()
    for s in keys(V)
        if length(tree[s]) > 0
            a = sample(tree[s])[1]
            policy[s] = a
        end
    end
    return policy
end

function tournament(N_games, policy1, policy2, tree; max_iter=10000)
    wp = Tuple(zeros(Bool, 14))
    bp = Tuple(zeros(Bool, 14))
    wh = (UInt8(N), UInt8(0))
    bh = (UInt8(N), UInt8(0))
    start_turn = UInt8(1)
    Ps = [binomial(4, k) for k in 0:4]*(0.5^4)
    dice = () -> wsample(UInt8.(0:4), Ps)
    wins = zeros(Int, 2)
    for n in ProgressBar(1:N_games)
        roll = dice()
        s = (bp, wp, bh, wh, roll, start_turn)
        winner = nothing
        for i in 1:max_iter
            if i == max_iter
                @warn "Maximum number of moves for a game reached"
            end
            turn = s[end]
            move = nothing
            if turn == 1
                move = policy1[s]
            else
                move = policy2[s]
            end
            ss = make_move(s, move)
            roll = dice()
            s = ss[roll+1]
            won = has_won(s)
            if won > 0
                winner = won
                break
            end
        end
        if winner !== nothing
            wins[winner] += 1
        end
    end
    return wins
end

tree, visited = exhaustive_tree(s_starts; n_iter=10000000);
tree, visited = exhaustive_tree_no_dice_in_state(s_start; n_iter=10000000);
V_white_win_av = value_iteration_averaged(tree, visited, 0.99, 1.0)
V_white_win = value_iteration(tree, visited, 0.99, 1.0)
V_black_win = value_iteration(tree, visited, 0.99, 1.0)
opt_policy_white_win = greedy_policy(V_white_win, tree, 0.99)
opt_policy_black_win = greedy_policy(V_black_win, tree, 0.99)
rand_policy = random_policy(V, tree)

wins = tournament(10000, opt_policy_white_win, opt_policy_black_win, tree)
wins = tournament(10000, rand_policy, rand_policy, tree)

wins = tournament(10000, opt_policy_white_win, rand_policy, tree)
wins = tournament(10000, rand_policy, opt_policy_white_win, tree)

wins = tournament(10000, rand_policy, opt_policy_black_win, tree)
wins = tournament(10000, opt_policy_black_win, rand_policy, tree)

wins = tournament(10000, opt_policy_white_win, opt_policy_black_win, tree)
wins = tournament(10000, opt_policy_white_win, opt_policy_white_win, tree)

wins = tournament(10000, opt_policy_white_win, opt_policy_black_win, tree)
wins = tournament(10000, opt_policy_black_win, opt_policy_black_win, tree)


using Dates
using JLD2

function start_state_int(bs, bbs)
    # if N > 7
    #     throw(ErrorException("Cannot encode >7 pieces in 32 bit encoding"))
    # end
    s = UInt32(0)
    z = UInt32(0)
    o = UInt32(1)
    # Shared squares
    for i in 1:8
        s += bbs[i]*z
    end
    # white safe
    for i in 14:19
        s += bs[i]*z
    end
    # black safe
    for i in 20:25
        s += bs[i]*z
    end
    # white home
    for i in 27:27#26:28
        s += bs[i]*o
    end
    # black home
    for i in 30:30#29:31
        s += bs[i]*o
    end
    # turn
    s += bs[32]*z
    return s
end

function how_many_home(s)
    return (4*check_bit(s, 26) + 2*check_bit(s, 27) + check_bit(s, 28), 4*check_bit(s, 29) + 2*check_bit(s, 30) + check_bit(s, 31))
end

function has_home(s, bt)
    if bt
        return check_bit(s, 29) || check_bit(s, 30) || check_bit(s, 31)
    else
        return check_bit(s, 26) || check_bit(s, 27) || check_bit(s, 28)
    end
end

function black_turn(s)
    return check_bit(s, 32)
end

function check_bit(s, n)
    return ((s >> (n-1)) & 1) == 1
end

function check_trit(s, n, bs, bbs)
    a = s % bs[14]
    # if n == 8
    #     return a ÷ bbs[n]
    # else
    return (a ÷ bbs[n]) % bbs[2]
    #end
end

function move_out(s, bs, bt)
    # subtracts 1 from 3 bit encoded home (111 -> 110, 100 -> 011, etc)
    if bt
        if check_bit(s, 31)
            s -= bs[31]
        else
            if check_bit(s, 30)
                s -= bs[30]
            else
                if !check_bit(s, 29)
                    throw(ErrorException("Cannot move out (1)"))
                end
                s -= bs[29]
                s += bs[30]
            end
            s += bs[31]
        end
    else
        if check_bit(s, 28)
            s -= bs[28]
        else
            if check_bit(s, 27)
                s -= bs[27]
            else
                if !check_bit(s, 26)
                    throw(ErrorException("Cannot move out (2)"))
                end
                s -= bs[26]
                s += bs[27]
            end
            s += bs[28]
        end
    end
    return s
end

function move_in(s, bs, bt)
    # adds 1 to 3 bit encoded home (110 -> 111, 011 -> 100, etc)
    if bt
        if !check_bit(s, 28)
            s += bs[28]
        else
            if !check_bit(s, 27)
                s += bs[27]
            else
                if check_bit(s, 26)
                    throw(ErrorException("Cannot move in (2)"))
                end
                s += bs[26]
                s -= bs[27]
            end
            s -= bs[28]
        end
    else
        if !check_bit(s, 31)
            s += bs[31]
        else
            if !check_bit(s, 30)
                s += bs[30]
            else
                if check_bit(s, 29)
                    throw(ErrorException("Cannot move in (1)"))
                end
                s += bs[29]
                s -= bs[30]
            end
            s -= bs[31]
        end
    end
    return s
end

function flip_turn(s, bs, bbs, bt; half=false)
    if half
        # Shared squares
        for i in 1:8
            ct = check_trit(s, i, bs, bbs)
            if ct == 1
                s += bbs[i]
            elseif ct == 2
                s -= bbs[i]
            end
        end
        remove = UInt32(0)
        add = UInt32(0)
        # white safe
        for i in 14:19
            if check_bit(s, i)
                if !check_bit(s, i+6)
                    remove += bs[i]
                    add += bs[i+6]
                end
            end
        end
        # black safe
        for i in 20:25
            if check_bit(s, i)
                if !check_bit(s, i-6)
                    remove += bs[i]
                    add += bs[i-6]
                end
            end
        end
        # white home
        for i in 26:28
            if check_bit(s, i)
                remove += bs[i]
                add += bs[i+3]
            end
        end
        # black home
        for i in 29:31
            if check_bit(s, i)
                remove += bs[i]
                add += bs[i-3]
            end
        end
        s -= remove
        s += add
    end
    if bt
        s -= bs[32]
    else
        s += bs[32]
    end
    #end
    return s
end

function place_piece(s, to, bs, bbs, bt)
    if to == 0
        return s
    elseif to < 9
        if bt
            # capture
            if check_trit(s, to, bs, bbs) == 0x1
                s = move_in(s, bs, bt)
                s -= bbs[to]
            end
            if check_trit(s, to, bs, bbs) == 0x2
                throw(ErrorException("Piece alrdy in 'to' pos (1)"))
            end
            s += 0x2*bbs[to]
        else
            # capture
            if check_trit(s, to, bs, bbs) == 0x2
                s = move_in(s, bs, bt)
                s -= 0x2*bbs[to]
            end
            if check_trit(s, to, bs, bbs) == 0x1
                throw(ErrorException("Piece alrdy in 'to' pos (2)"))
            end
            s += bbs[to]
        end
    else
        if check_bit(s, to)
            throw(ErrorException("Piece alrdy in 'to' pos (3)"))
        end
        s += bs[to]
    end
    return s
end

function take_piece(s, from, bs, bbs, bt)
    if from == 0
        s = move_out(s, bs, bt)
    elseif from < 9
        if black_turn(s)
            if check_trit(s, from, bs, bbs) != 0x2
                throw(ErrorException("No piece to take from (1)"))
            end
            s -= 0x2*bbs[from]
        else
            if check_trit(s, from, bs, bbs) != 0x1
                throw(ErrorException("No piece to take from (2)"))
            end
            s -= bbs[from]
        end
    else
        if !check_bit(s, from)
            throw(ErrorException("No piece to take from (3)"))
        end
        s -= bs[from]
    end
    return s
end

function move_piece(s, from, to, bs, bbs, bt; half=false)
    s = take_piece(s, from, bs, bbs, bt)
    s = place_piece(s, to, bs, bbs, bt)
    # get another roll
    if to == 4 || to == 17 || to == 23 || to == 19 || to == 25
        return s
    else
        s = flip_turn(s, bs, bbs, bt; half=half)
        return s
    end
end

#function possible_moves(s, roll, bs, bbs)
function possible_neighbours(s, roll, bs, bbs; half=false)
    # moves = Int16[]
    bt = black_turn(s)
    if roll == 0
        return [flip_turn(s, bs, bbs, bt; half=half)]
    end
    neighs = UInt32[]#Set{UInt32}()
    if roll < 1 || roll > 4
        throw(ErrorException("Wrong roll"))
    end
    # from home
    if has_home(s, bt)
        if bt
            if !check_bit(s, 19+roll)
                # push!(moves, to_move(bt, 0, 19+roll))
                push!(neighs, move_piece(s, 0, 19+roll, bs, bbs, bt; half=half))
            end
        else
            if !check_bit(s, 13+roll)
                # push!(moves, to_move(bt, 0, 13+roll))
                push!(neighs, move_piece(s, 0, 13+roll, bs, bbs, bt; half=half))
            end
        end
    end
    # from start safe
    for i in 1:4
        if bt
            if check_bit(s, 19+i)
                if i+roll < 5
                    if !check_bit(s, 19+i+roll)
                        # push!(moves, to_move(bt, 19+i, 19+i+roll))
                        push!(neighs, move_piece(s, 19+i, 19+i+roll, bs, bbs, bt; half=half))
                    end
                else
                    if check_trit(s, i+roll-4, bs, bbs) != 0x2
                        # push!(moves, to_move(bt, 19+i, i+roll-4))
                        push!(neighs, move_piece(s, 19+i, i+roll-4, bs, bbs, bt; half=half))
                    end
                end
            end
        else
            if check_bit(s, 13+i)
                if i+roll < 5
                    if !check_bit(s, 13+i+roll)
                        # push!(moves, to_move(bt, 13+i, 13+i+roll))
                        push!(neighs, move_piece(s, 13+i, 13+i+roll, bs, bbs, bt; half=half))
                    end
                else
                    if check_trit(s, i+roll-4, bs, bbs) != 0x1
                        # push!(moves, to_move(bt, 13+i, i+roll-4))
                        push!(neighs, move_piece(s, 13+i, i+roll-4, bs, bbs, bt; half=half))
                    end
                end
            end
        end
    end
    # # from unsafe
    for i in 1:8
        if bt
            if check_trit(s, i, bs, bbs) == 0x2
                if i+roll < 9
                    # central safe square
                    if i+roll == 4
                        if check_trit(s, i+roll, bs, bbs) == 0x0
                            # push!(moves, to_move(bt, i, i+roll))
                            push!(neighs, move_piece(s, i, i+roll, bs, bbs, bt; half=half))
                        end
                    elseif check_trit(s, i+roll, bs, bbs) != 0x2
                        # push!(moves, to_move(bt, i, i+roll))
                        push!(neighs, move_piece(s, i, i+roll, bs, bbs, bt; half=half))
                    end
                elseif i+roll < 11
                    if !check_bit(s, 24+i+roll-9)
                        # push!(moves, to_move(bt, i, 24+i+roll-9))
                        push!(neighs, move_piece(s, i, 24+i+roll-9, bs, bbs, bt; half=half))
                    end
                elseif i+roll == 11
                    push!(neighs, move_piece(s, i, 0, bs, bbs, bt; half=half))
                end
            end
        else
            if check_trit(s, i, bs, bbs) == 0x1
                if i+roll < 9
                    # central safe square
                    if i+roll == 4
                        if check_trit(s, i+roll, bs, bbs) == 0x0
                            # push!(moves, to_move(bt, i, i+roll))
                            push!(neighs, move_piece(s, i, i+roll, bs, bbs, bt; half=half))
                        end
                    elseif check_trit(s, i+roll, bs, bbs) != 0x1
                        # push!(moves, to_move(bt, i, i+roll))
                        push!(neighs, move_piece(s, i, i+roll, bs, bbs, bt; half=half))
                    end
                elseif i+roll < 11
                    if !check_bit(s, 18+i+roll-9)
                        # push!(moves, to_move(bt, i, 18+i+roll-9))
                        push!(neighs, move_piece(s, i, 18+i+roll-9, bs, bbs, bt; half=half))
                    end
                elseif i+roll == 11
                    push!(neighs, move_piece(s, i, 0, bs, bbs, bt; half=half))
                end
            end
        end
    end
    # from end safe
    for i in 1:2
        if bt
            if check_bit(s, 23+i)
                if i+roll == 3
                    # push!(moves, to_move(bt, 23+i, 0))
                    push!(neighs, move_piece(s, 23+i, 0, bs, bbs, bt; half=half))
                elseif i+roll < 3
                    if !check_bit(s, 23+i+roll)
                        # push!(moves, to_move(bt, 23+i, 23+i+roll))
                        push!(neighs, move_piece(s, 23+i, 23+i+roll, bs, bbs, bt; half=half))
                    end
                end
            end
        else
            if check_bit(s, 17+i)
                if i+roll == 3
                    # push!(moves, to_move(bt, 17+i, 0))
                    push!(neighs, move_piece(s, 17+i, 0, bs, bbs, bt; half=half))
                elseif i+roll < 3
                    if !check_bit(s, 17+i+roll)
                        # push!(moves, to_move(bt, 17+i, 17+i+roll))
                        push!(neighs, move_piece(s, 17+i, 17+i+roll, bs, bbs, bt; half=half))
                    end
                end
            end
        end
    end
    if isempty(neighs)
        return [flip_turn(s, bs, bbs, bt; half=half)]
    else
        # return moves
        return neighs
    end
end


function has_won(s, bs, bbs; half=false)
    if !half
        white_win = true
        for i in 26:28
            if check_bit(s, i)
                white_win = false
                break
            end
        end
        if !white_win
            @goto black
        end
        for i in 14:19
            if check_bit(s, i)
                white_win = false
                break
            end
        end
        if !white_win
            @goto black
        end
        for i in 1:8
            if check_trit(s, i, bs, bbs) == 0x1
                white_win = false
                break
            end
        end
        if white_win
            return 1
        end
    end
    @label black
    black_win = true
    for i in 29:31
        if check_bit(s, i)
            black_win = false
            break
        end
    end
    if !black_win
        @goto finish
    end
    for i in 20:25
        if check_bit(s, i)
            black_win = false
            break
        end
    end
    if !black_win
        @goto finish
    end
    for i in 1:8
        if check_trit(s, i, bs, bbs) == 0x2
            black_win = false
            break
        end
    end
    @label finish
    if black_win
        if half 
            return true
        else
            return 2
        end
    else
        if half
            return false
        else
            return 0
        end
    end
end

function pieces_on_the_board(s, bs, bbs)
    white_count = 0
    black_count = 0
    for i in 14:19
        white_count += check_bit(s, i)
    end
    for i in 20:25
        black_count += check_bit(s, i)
    end
    for i in 1:8
        white_count += (check_trit(s, i, bs, bbs) == 0x1)
    end
    for i in 1:8
        black_count += (check_trit(s, i, bs, bbs) == 0x2)
    end
    for i in 26:28
        white_count += check_bit(s, i)*2^(i-26)
    end
    for i in 29:31
        black_count += check_bit(s, i)*2^(i-29)
    end
    return minimum((white_count, black_count))
end

# begin
# s = move_out(s, bs, true)
# s = move_out(s, bs, true)
# s = move_out(s, bs, true)
# s = move_out(s, bs, true)
# s = move_out(s, bs, true)
# s = move_out(s, bs, true)
# s = move_out(s, bs, false)
# s = move_out(s, bs, false)
# s = move_out(s, bs, false)
# s = move_out(s, bs, false)
# s = move_out(s, bs, false)
# s = move_out(s, bs, false)
# end

# bt = black_turn(s)
# s = move_piece(s, 0, 4, bs, bbs, bt)
# bt = black_turn(s)
# s = move_piece(s, 0, 3, bs, bbs, bt)
# bt = black_turn(s)
# s = move_piece(s, 0, 6, bs, bbs, bt)
# bt = black_turn(s)
# s = move_piece(s, 0, 7, bs, bbs, bt)


# check_trit(s, 4, bs, bbs)

# ss = possible_neighbours(s, 2, bs, bbs)
# possible_moves(s, 2, bs, bbs)

function bfs(s_start, bs, bbs; half=false, max_iter=1000000000)
    frontier = Vector{UInt32}()
    pushfirst!(frontier, s_start)
    visited = Set{UInt32}()
    leaf_nodes = Set{UInt32}()
    # inv_graph = DefaultDict{UInt32, Vector{UInt32}}([])
    # graph = DefaultDict{UInt32, Vector{UInt32}}([])
    # depths = Dict{UInt32, Int}()
    # depths[s_start] = 0
    push!(visited, s_start)
    for i in 1:max_iter
        if isempty(frontier)
            return visited, leaf_nodes#, inv_graph, depths, graph
        end
        s = pop!(frontier)
        for roll in 0:4
            neighs = possible_neighbours(s, roll, bs, bbs; half=half)
            for neighbour in neighs
                # if s ∉ inv_graph[neighbour]
                #     push!(inv_graph[neighbour], s)
                # end
                # if neighbour ∉ graph[s]
                #     push!(graph[s], neighbour)
                # end
                if neighbour ∉ visited
                    push!(visited, neighbour)
                    #depths[neighbour] = depths[s] + 1
                    if has_won(neighbour, bs, bbs; half=half) > 0
                        push!(leaf_nodes, neighbour)
                    else
                        pushfirst!(frontier, neighbour)
                    end
                end
            end
        end
    end
    throw(ErrorException("Iteration limit reached"))
end

function get_neighs(visited, bs, bbs)
    # function fun(chunk, graph, lk, bs, bbs)
    #     sm = 0
    #     for s in chunk
    #         sm += 1
    #         if sm % 1000000 == 0
    #             println("=======$(sm)=======")
    #             println("Took $(now() - last_stamp)")
    #             flush(stdout)
    #             last_stamp = now()
    #         end
    #         s_next = []
    #         for roll in 0:4
    #             neighs = possible_neighbours(s, roll, bs, bbs)
    #             roll_uint = UInt8(roll)
    #             for neigh in neighs
    #                 push!(s_next, (roll_uint, neigh))
    #             end
    #         end
    #         lock(lk) do
    #             graph[s] = Tuple(s_next)
    #         end
    #     end
    # end
    # graph = Dict{UInt32, Tuple{Vararg{Tuple{UInt8, UInt32}, T}} where T}()
    # chunks = Iterators.partition(visited, length(visited) ÷ Threads.nthreads())
    # lk = ReentrantLock()
    # tasks = map(chunks) do chunk
    #     Threads.@spawn fun(chunk, graph, lk, bs, bbs)
    # end
    # fetched_tasks = fetch.(tasks)
    
    graph = Dict{UInt32, Tuple{Vararg{Tuple{UInt8, UInt32}, T}} where T}()
    #graph = Dict{UInt32, NTuple{Tuple{UInt8, UInt32}}}()
    sm = 0
    last_stamp = now()
    N = length(visited)
    #lk = ReentrantLock()

    for s in visited#Threads.@threads
        sm += 1
        #s = visited[n]
        if sm % 1000000 == 0
            println("=======$(sm)=======$(round(100*100*sm/N)/100)%")
            println("Took $(now() - last_stamp)")
            flush(stdout)
            last_stamp = now()
        end
        s_next = []
        for roll in 0:4
            neighs = possible_neighbours(s, roll, bs, bbs)
            roll_uint = UInt8(roll)
            for neigh in neighs
                push!(s_next, (roll_uint, neigh))
            end
        end
        #lock(lk) do
        graph[s] = Tuple(s_next)
        #end
        if sm == 138000000#69000000
            @save "/scratch/jo2229/graph1.jld2" graph
            graph = Dict{UInt32, Tuple{Vararg{Tuple{UInt8, UInt32}, T}} where T}()
        end
    end
    @save "/scratch/jo2229/graph2.jld2" graph
    #return graph
end

function inv_bfs(leaf_nodes, inv_graph; max_iter=1000000000)
    frontier = Vector{UInt32}()
    inv_depths = Dict{UInt32, UInt16}()
    visited = Set{UInt32}()
    o = one(UInt16)
    for leaf in leaf_nodes
        pushfirst!(frontier, leaf)
        inv_depths[leaf] = zero(UInt16)
        push!(visited, leaf)
    end
    for i in 1:max_iter
        if isempty(frontier)
            println(length(visited))
            return inv_depths
        end
        s = pop!(frontier)
        d = inv_depths[s]
        for inv_neigh in inv_graph[s]
            new_d = d + o
            if inv_neigh ∉ visited
                inv_depths[inv_neigh] = new_d
                push!(visited, inv_neigh)
                pushfirst!(frontier, inv_neigh)
            end
        end
    end
    throw(ErrorException("Iteration limit reached"))
end

function pretty_print(s, bs, bbs)
    pretty = []
    dummy = Int64[]
    for i in 26:28
        push!(dummy, check_bit(s, i))
    end
    push!(pretty, dummy)
    dummy = Int64[]
    for i in 29:31
        push!(dummy, check_bit(s, i))
    end
    push!(pretty, dummy)
    dummy = Int64[]
    for i in 14:17
        push!(dummy, check_bit(s, i))
    end
    for i in 20:23
        push!(dummy, check_bit(s, i))
    end
    for i in 1:8
        push!(dummy, check_trit(s, i, bs, bbs))
    end
    for i in 18:19
        push!(dummy, check_bit(s, i))
    end
    for i in 24:25
        push!(dummy, check_bit(s, i))
    end
    push!(pretty, dummy)
    push!(pretty, black_turn(s))
    return pretty
end

function get_new_value(s, bs, bbs, V, Ps; half=false)
    bt = black_turn(s)
    nv = 0
    sp = flip_turn(s, bs, bbs, bt; half=half)
    if half
        sp -= bs[32]
        nv += -Ps[1]*V[sp]
    else
        nv += Ps[1]*V[sp]
    end
    for roll in 1:4
        neighs = possible_neighbours(s, roll, bs, bbs; half=half)
        nvv = nothing
        if half
            nvv = -Inf
        else
            if bt
                nvv = Inf
            else
                nvv = -Inf
            end
        end
        for neighbour in neighs
            if half
                btt = black_turn(neighbour)
                if btt
                    neighbour -= bs[32]
                end
            end
            # winner = has_won(neighbour, bs, bbs)
            # R = winner == 1 ? 100 : winner == 2 ? -100 : 0
            # Vp = V[neighbour]
            if half
                factor = btt ? -1 : 1
                nvv = max(nvv, factor*V[neighbour])
            else
                if bt
                    nvv = min(nvv, V[neighbour])# R +
                else
                    nvv = max(nvv, V[neighbour])# R +
                end
            end
        end
        nv += Ps[roll+1]*nvv
    end
    return nv
end

function value_iteration(visited, leaf_nodes, θ, bs, bbs; half=false, max_iter=100)
    start_time = now()
    times = []
    ss = []
    deltas = []
    V = Dict{UInt32, Float64}(visited .=> zeros(length(visited)))
    #V = zeros(2^32)
    for leaf in leaf_nodes
        if has_won(leaf, bs, bbs; half=half) == 1
            V[leaf] = 100
        else
            V[leaf] = -100
        end
    end
    println("Initializing map took $(now() - start_time)")
    flush(stdout)
    Ps = [binomial(4, k) for k in 0:4]*(0.5^4) 
    Δ = 0   
    for i in 1:max_iter
        println("=======$(i)=======")
        flush(stdout)
        last_stamp = now()
        Δ = 0
        ssum = 0
        #old_V = copy(V)
        for s in visited#ProgressBar
        #for s in ProgressBar(_seed)
            # if s in leaf_nodes
            #     continue
            # end
            # if inv_depths[s] > i
            #     continue
            # end
            # if pieces_on_the_board(s, bs, bbs) > i
            #     continue
            # end
            #v = i == 1 ? 0 : V[s]
            v = V[s]
            nv = get_new_value(s, bs, bbs, V, Ps; half=half)
            #ssum += 1
            # if nv != 0
            #     ssum += 1
            # end
            Δ = max(Δ, abs(v-nv))
            V[s] = nv
        end
        println("Took $(now() - last_stamp)")
        push!(times, now() - last_stamp)
        push!(ss, ssum)
        push!(deltas, Δ)
        flush(stdout)
        println("Δ: $(Δ)")
        flush(stdout)
        if Δ < θ
            # println("Value iteration took $(now() - start_time)")
            # flush(stdout)
            # return V, times, ss, deltas
            break
        end
    end
    println("Maximum number of iterations reached (Delta: $(Δ))")
    flush(stdout)
    println("Value iteration took $(now() - start_time)")
    flush(stdout)
    return V, times, ss, deltas
end


bs = UInt32.(2 .^ (0:31));
bbs = UInt32.(3 .^ (0:7));

s = start_state_int(bs, bbs);
s_start = s;
# println("Starting BFS...")
# start_bfs_time = now()
# flush(stdout)
#@time visited, leaf_nodes, inv_graph, depths, graph = bfs(s_start, bs, bbs);
@time visited, leaf_nodes = bfs(s_start, bs, bbs; half=true);

#@time inv_depths = inv_bfs(leaf_nodes, inv_graph; max_iter=10000000);
#@time inv_depths = inv_bfs([s_start], graph; max_iter=10000000);
# @save "visited.jld2" visited
# @save "leaf_nodes.jld2" leaf_nodes

println("Loading states...")
loading_states_time = now()
flush(stdout)
visited = load("visited.jld2")["visited"]
leaf_nodes = load("leaf_nodes.jld2")["leaf_nodes"]

println("Finished getting all states. Took $(now()-loading_states_time)")
flush(stdout)
println("Starting value iteration...")
flush(stdout)
epsilon = 0.0000001;
max_iter = 400;#400;
println("epsilon: $(epsilon), max_iter: $(max_iter)")
flush(stdout)


no_leafs = collect(setdiff(visited, leaf_nodes));
n_threads = 8;
bin_size = ceil(Int, length(no_leafs) / n_threads)
idxs = push!(collect(1:bin_size:length(no_leafs)), length(no_leafs))


V, times, ss, deltas = value_iteration(no_leafs, leaf_nodes, epsilon, bs, bbs; half=true, max_iter=max_iter);
println("Everything took $(now()-loading_states_time)")
flush(stdout)

function EXTRA_STUFF()
    # println("Writing keys to binary file model_keys_in_loop_eps=1.0.dat...")
    # flush(stdout)
    # bytes1 = write("model_keys_in_loop_eps=1.0.dat", collect(keys(V)))
    # println("Writing values to binary file model_values_in_loop_eps=1.0.dat...")
    # flush(stdout)
    # bytes2 = write("model_values_in_loop_eps=1.0.dat", collect(values(V)))


    # function greedy_policy(V, γ)
    #     s_type = typeof(first(keys(V)))
    #     a_type = typeof(first(values(tree))[1][1])
    #     policy = Dict{s_type, a_type}()
    #     Ps = [binomial(4, k) for k in 0:4]*(0.5^4)
    #     for s in keys(V)
    #         if length(tree[s]) > 0
    #             max_v = -10000000000
    #             max_a = (UInt8(0), UInt8(0), UInt8(0))
    #             for (a, ss) in tree[s]
    #                 v = 0
    #                 for (n, sp) in enumerate(ss)
    #                     winner = has_won(sp)
    #                     R = winner == 1 ? -100 : winner == 2 ? 100 : 0
    #                     v += Ps[n]*(R + γ*V[sp])
    #                 end
    #                 if v > max_v
    #                     max_a = a
    #                     max_v = v
    #                 end
    #             end
    #             policy[s] = max_a
    #         end
    #     end
    #     return policy
    # end

    # function to_move(bt, from, to)
    #     move = Int16(0)
    #     move += bt#*bs16[1]

    #     #from = pos32_to_pos16(from, bt)
    #     #to = pos32_to_pos16(to, bt)
        
    #     #move += bs16[from+1]
    #     #move += bs16[to+1]
    #     move += from*Int16(10)
    #     move += to*Int16(1000)
    #     return move
    # end

    # function pos32_to_pos16(pos, bt)
    #     if pos > 8
    #         if bt
    #             if pos < 20 || pos > 25
    #                 throw(ErrorException("Wrong to_move format (1)"))
    #             end
    #             pos -= 11
    #         else
    #             if pos < 14 || pos > 19
    #                 throw(ErrorException("Wrong to_move format (2)"))
    #             end
    #             pos -= 5
    #         end
    #         # split safe tiles
    #         if pos < 13
    #             pos -= 8
    #         end
    #     else
    #         pos += 4
    #     end
    #     return pos
    # end

    # function pos16_to_pos32(pos, bt)
    #     if pos < 5
    #         if bt
    #             pos += 19
    #         else
    #             pos += 13
    #         end
    #     elseif pos > 12
    #         pos += 5
    #         if bt
    #             pos += 6
    #         end
    #     else
    #         pos -= 4
    #     end
    #     return pos
    # end

    # function check_conversions()
    #     for i in 1:14
    #         if i != pos32_to_pos16(pos16_to_pos32(i, true), true)
    #             println("wot")
    #         end
    #         if i != pos32_to_pos16(pos16_to_pos32(i, false), false)
    #             println("wot")
    #         end
    #     end
    # end
    return nothing
end

a = UInt32[]
for (k, v) in depths
    if k ∉ keys(inv_depths)
        push!(a, k)
    end
end
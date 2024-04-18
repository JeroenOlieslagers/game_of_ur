using Dates
using JLD2

function start_state_int(bs, bbs; N=7)
    s = UInt32(0)
    z = UInt32(0)
    o = UInt32(1)
    bits = bitstring(N)[end-2:end]
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
    for i in 26:28#26:28
        if bits[i+1-26] == '1'
            s += bs[i]*o
        end
    end
    # black home
    for i in 29:31#29:31
        if bits[i+1-29] == '1'
            s += bs[i]*o
        end
    end
    # turn
    #s += bs[32]*z
    return s
end

function how_many_home(s)
    return 4*check_bit(s, 26) + 2*check_bit(s, 27) + check_bit(s, 28)
end

function has_home(s)
    return check_bit(s, 26) || check_bit(s, 27) || check_bit(s, 28)
end

function check_bit(s, n)
    return ((s >> (n-1)) & 1) == 1
end

function check_trit(s, n, bs, bbs)
    a = s % bs[14]
    return (a ÷ bbs[n]) % bbs[2]
end

function move_out(s, bs)
    # subtracts 1 from 3 bit encoded home (111 -> 110, 100 -> 011, etc)
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
    return s
end

function move_in(s, bs; dark=false)
    # adds 1 to 3 bit encoded home (110 -> 111, 011 -> 100, etc)
    if !dark
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

function flip_turn(s, bs, bbs)
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

    s += bs[32]
    return s
end

function place_piece(s, to, bs, bbs)
    if to == 0
        return s
    elseif to < 9
        # capture
        if check_trit(s, to, bs, bbs) == 0x2
            s = move_in(s, bs; dark=true)
            s -= 0x2*bbs[to]
        end
        if check_trit(s, to, bs, bbs) == 0x1
            throw(ErrorException("Piece alrdy in 'to' pos (2)"))
        end
        s += bbs[to]
    else
        if check_bit(s, to)
            throw(ErrorException("Piece alrdy in 'to' pos (3)"))
        end
        s += bs[to]
    end
    return s
end

function take_piece(s, from, bs, bbs)
    if from == 0
        s = move_out(s, bs)
    elseif from < 9
        if check_trit(s, from, bs, bbs) != 0x1
            throw(ErrorException("No piece to take from (2)"))
        end
        s -= bbs[from]
    else
        if !check_bit(s, from)
            throw(ErrorException("No piece to take from (3)"))
        end
        s -= bs[from]
    end
    return s
end

function move_piece(s, from, to, bs, bbs)
    s = take_piece(s, from, bs, bbs)
    s = place_piece(s, to, bs, bbs)
    # get another roll
    if to == 4 || to == 17 || to == 23 || to == 19 || to == 25
        return s
    else
        s = flip_turn(s, bs, bbs)
        return s
    end
end

function possible_neighbours(s, roll, bs, bbs)
    if roll == 0
        return [flip_turn(s, bs, bbs)]
    end
    neighs = UInt32[]#Set{UInt32}()
    if roll < 1 || roll > 4
        throw(ErrorException("Wrong roll"))
    end
    # from home
    if has_home(s)
        if !check_bit(s, 13+roll)
            # push!(moves, to_move(bt, 0, 13+roll))
            push!(neighs, move_piece(s, 0, 13+roll, bs, bbs))
        end
    end
    # from start safe
    for i in 1:4
        if check_bit(s, 13+i)
            if i+roll < 5
                if !check_bit(s, 13+i+roll)
                    # push!(moves, to_move(bt, 13+i, 13+i+roll))
                    push!(neighs, move_piece(s, 13+i, 13+i+roll, bs, bbs))
                end
            else
                if check_trit(s, i+roll-4, bs, bbs) != 0x1
                    # cant capture central rosette
                    if !(check_trit(s, i+roll-4, bs, bbs) == 0x2 && i+roll-4 == 4)
                        # push!(moves, to_move(bt, 13+i, i+roll-4))
                        push!(neighs, move_piece(s, 13+i, i+roll-4, bs, bbs))
                    end
                end
            end
        end
    end
    # # from unsafe
    for i in 1:8
        if check_trit(s, i, bs, bbs) == 0x1
            if i+roll < 9
                # central safe square
                if i+roll == 4
                    if check_trit(s, i+roll, bs, bbs) == 0x0
                        # push!(moves, to_move(bt, i, i+roll))
                        push!(neighs, move_piece(s, i, i+roll, bs, bbs))
                    end
                elseif check_trit(s, i+roll, bs, bbs) != 0x1
                    # push!(moves, to_move(bt, i, i+roll))
                    push!(neighs, move_piece(s, i, i+roll, bs, bbs))
                end
            elseif i+roll < 11
                if !check_bit(s, 18+i+roll-9)
                    # push!(moves, to_move(bt, i, 18+i+roll-9))
                    push!(neighs, move_piece(s, i, 18+i+roll-9, bs, bbs))
                end
            elseif i+roll == 11
                push!(neighs, move_piece(s, i, 0, bs, bbs))
            end
        end
    end
    # from end safe
    for i in 1:2
        if check_bit(s, 17+i)
            if i+roll == 3
                # push!(moves, to_move(bt, 17+i, 0))
                push!(neighs, move_piece(s, 17+i, 0, bs, bbs))
            elseif i+roll < 3
                if !check_bit(s, 17+i+roll)
                    # push!(moves, to_move(bt, 17+i, 17+i+roll))
                    push!(neighs, move_piece(s, 17+i, 17+i+roll, bs, bbs))
                end
            end
        end
    end
    if isempty(neighs)
        return [flip_turn(s, bs, bbs)]
    else
        return neighs
    end
end

function has_won(s, bs, bbs)
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
    return black_win
end

function bfs(s_start, bs, bbs; max_iter=1000000000)
    frontier = Vector{UInt32}()
    pushfirst!(frontier, s_start)
    leaf_nodes = Set{UInt32}()
    ####### NO SMART
    # visited = Set{UInt32}()
    ####### SMART
    pieces = pieces_on_the_board(s_start, bs, bbs)
    visited = Dict{Tuple{Int, Int}, Set{UInt32}}()
    N = pieces[1]
    for i in 1:N
        for j in i:min(N, i+1)
            for k in j:min(N, i+j-1)
                pieces_target = (i-(k-j), k)
                visited[pieces_target] = Set{UInt32}()
            end
        end
    end
    # inv_graph = DefaultDict{UInt32, Vector{UInt32}}([])
    # graph = DefaultDict{UInt32, Vector{UInt32}}([])
    # depths = Dict{UInt32, Int}()
    # depths[s_start] = 0
    push!(visited[pieces], s_start)
    for i in 1:max_iter
        if i % 1000000 == 0
            println("$(i)")
            flush(stdout)
        end
        if isempty(frontier)
            return Dict(visited), leaf_nodes#, inv_graph, depths, graph
        end
        s = pop!(frontier)
        for roll in 0:4
            neighs = possible_neighbours(s, roll, bs, bbs)
            for neighbour in neighs
                bt = check_bit(neighbour, 32)
                if bt
                    neighbour -= bs[32]
                end
                # if s ∉ inv_graph[neighbour]
                #     push!(inv_graph[neighbour], s)
                # end
                # if neighbour ∉ graph[s]
                #     push!(graph[s], neighbour)
                # end
                if has_won(neighbour, bs, bbs)
                    push!(leaf_nodes, neighbour)
                else
                    pieces = pieces_on_the_board(neighbour, bs, bbs)
                    if neighbour ∉ visited[pieces]
                        push!(visited[pieces], neighbour)
                        #depths[neighbour] = depths[s] + 1
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
    push!(pretty, check_bit(s, 32))
    return pretty
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
        white_count += check_bit(s, i)*2^(28-i)
    end
    for i in 29:31
        black_count += check_bit(s, i)*2^(31-i)
    end
    if white_count < black_count
        return (white_count, black_count)
    else
        return (black_count, white_count)
    end
end

function get_new_value(s, bs, bbs, V, Ps)
    nv = 0
    sp = flip_turn(s, bs, bbs)
    sp -= bs[32]
    nv += -Ps[1]*V[sp]
    for roll in 1:4
        neighs = possible_neighbours(s, roll, bs, bbs)
        #nvv = Inf
        nvv = -Inf
        for neighbour in neighs
            bt = check_bit(neighbour, 32)
            if bt
                neighbour -= bs[32]
            end
            factor = bt ? -1 : 1
            #nvv = min(nvv, factor*V[neighbour])
            nvv = max(nvv, factor*V[neighbour])
        end
        nv += Ps[roll+1]*nvv
    end
    return nv
end

function value_iteration(V, n_pieces, no_leafs, θ, bs, bbs, Ps; max_iter=100)
    start_time = now()
    Δ = 0
    ss = 0
    for i in 1:max_iter
        println("=======$(i)=======")
        flush(stdout)
        last_stamp = now()
        Δ = 0
        ss = 0
        for s in no_leafs
            if n_pieces !== nothing
                pieces = pieces_on_the_board(s, bs, bbs)
                if pieces != n_pieces
                    continue
                end
            end
            v = V[s]
            nv = get_new_value(s, bs, bbs, V, Ps)
            Δ = max(Δ, abs(v-nv))
            V[s] = nv
            ss += 1
        end
        println("Took $(now() - last_stamp)")
        flush(stdout)
        println("Δ: $(Δ)")
        flush(stdout)
        if Δ < θ
            println("Value iteration took $(now() - start_time)")
            flush(stdout)
            #return V, Δ, ss
            return nothing
        end
    end
    println("Maximum number of iterations reached (Delta: $(Δ))")
    flush(stdout)
    println("Value iteration took $(now() - start_time)")
    flush(stdout)
    #return V, Δ, ss
    return nothing
end

function value_iteration(V, no_leafs, θ, bs, bbs, Ps; max_iter=100)
    start_time = now()
    Δ = 0
    ss = 0
    for i in 1:max_iter
        # println("=======$(i)=======")
        # flush(stdout)
        # last_stamp = now()
        Δ = 0
        ss = 0
        for s in no_leafs
            # if n_pieces !== nothing
            #     pieces = pieces_on_the_board(s, bs, bbs)
            #     if pieces != n_pieces
            #         continue
            #     end
            # end
            v = V[s]
            nv = get_new_value(s, bs, bbs, V, Ps)
            Δ = max(Δ, abs(v-nv))
            V[s] = nv
            ss += 1
        end
        # println("Took $(now() - last_stamp)")
        # flush(stdout)
        # println("Δ: $(Δ)")
        # flush(stdout)
        if Δ < θ
            println("Non parallel Value iteration took $(now() - start_time) (Delta: $(Δ))")
            flush(stdout)
            #return V, Δ, ss
            return nothing
        end
    end
    println("Maximum number of iterations reached (Delta: $(Δ))")
    flush(stdout)
    println("Non parallel Value iteration took $(now() - start_time)")
    flush(stdout)
    #return V, Δ, ss
    return nothing
end

function value_iteration_parallel(V, no_leafs, θ, bs, bbs, Ps; max_iter=100)
    start_time = now()
    n_threads = Threads.nthreads();
    no_leafs = collect(no_leafs)
    bin_size = ceil(Int, length(no_leafs) / n_threads)
    idxs = push!(collect(1:bin_size:length(no_leafs)), length(no_leafs))
    Δ = 0
    Threads.@threads for n in 1:length(idxs)-1#
        offset = n == length(idxs)-1 ? 0 : 1
        #println("Indices $((idxs[n],idxs[n+1]-offset)) or $(idxs[n+1]-offset-idxs[n]+1) elements")
        #flush(stdout)
        subset = Set(no_leafs[idxs[n]:(idxs[n+1]-offset)]);
        for i in 1:max_iter
            #last_stamp = now()
            Δ = 0
            for s in subset
                # if n_pieces !== nothing
                #     pieces = pieces_on_the_board(s, bs, bbs)
                #     if pieces != n_pieces
                #         continue
                #     end
                # end
                v = V[s]
                nv = get_new_value(s, bs, bbs, V, Ps)
                Δ = max(Δ, abs(v-nv))
                V[s] = nv
            end
            #println("Thread $(n) iteration $(i) took $(now() - last_stamp) with Δ: $(Δ)")
            #flush(stdout)
            if Δ < θ
                #println("Thread $(n) took $(now() - start_time)")
                #flush(stdout)
                break
            end
        end
    end
    value_iteration(V, no_leafs, θ, bs, bbs, Ps; max_iter=max_iter)
    println("Parallel Value iteration took $(now() - start_time)")
    flush(stdout)
    return nothing
end

function value_iteration_smart(no_leafs, leaf_nodes, θ, bs, bbs; max_iter=100, N=7)
    start_time = now()
    #V = Dict{UInt32, Float64}(no_leafs .=> zeros(length(no_leafs)))
    V = Dict{UInt32, Float64}(union(values(no_leafs)...) .=> zeros(sum(length.(values(no_leafs)))))
    for leaf in leaf_nodes
        V[leaf] = -100
    end
    #no_leafs = collect(no_leafs)
    println("Initializing map took $(now() - start_time)")
    flush(stdout)
    Ps = [binomial(4, k) for k in 0:4]*(0.5^4) 
    #sss = []
    for i in 1:N
        for j in i:min(N, i+1)
            for k in j:min(N, i+j-1)#Threads.@threads 
                pieces_target = (i-(k-j), k)
                println("Pieces on the board $(pieces_target)")
                flush(stdout)
                current_states = no_leafs[pieces_target]
                #value_iteration(V, pieces_target, no_leafs, θ, bs, bbs, Ps; max_iter=max_iter)
                #value_iteration_parallel(V, current_states, θ, bs, bbs, Ps; max_iter=max_iter)
                value_iteration(V, current_states, θ, bs, bbs, Ps; max_iter=max_iter)
                #push!(sss, ss)
            end
        end
    end
    value_iteration(V, union(values(no_leafs)...), θ, bs, bbs, Ps; max_iter=max_iter)

    # println("Maximum number of iterations reached (Delta: $(Δ))")
    # flush(stdout)
    println("Value iteration took $(now() - start_time)")
    flush(stdout)
    return V#, sss
end

function value_iteration_full(visited, leaf_nodes, θ, bs, bbs; max_iter=100)
    start_time = now()
    V = Dict{UInt32, Float64}(visited .=> zeros(length(visited)))
    #V = zeros(2^32)
    for leaf in leaf_nodes
        V[leaf] = 100
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
        for s in visited
            v = V[s]
            nv = get_new_value(s, bs, bbs, V, Ps)
            Δ = max(Δ, abs(v-nv))
            V[s] = nv
        end
        println("Took $(now() - last_stamp)")
        flush(stdout)
        println("Δ: $(Δ)")
        flush(stdout)
        if Δ < θ
            println("Value iteration took $(now() - start_time)")
            flush(stdout)
            return V
        end
    end
    println("Maximum number of iterations reached (Delta: $(Δ))")
    flush(stdout)
    println("Value iteration took $(now() - start_time)")
    flush(stdout)
    return V
end

function value_iteration_parallel_full(visited, leaf_nodes, θ, bs, bbs, idxs; max_iter=100)
    start_time = now()
    V = Dict{UInt32, Float64}(visited .=> zeros(length(visited)))
    #V = zeros(2^32)
    for leaf in leaf_nodes
        V[leaf] = 100
    end
    visited = collect(visited);
    println("Initializing map took $(now() - start_time)")
    flush(stdout)
    Ps = [binomial(4, k) for k in 0:4]*(0.5^4)
    Δ = 0
    Threads.@threads for n in 1:length(idxs)-1
        offset = n == length(idxs)-1 ? 0 : 1
        println("Indices $((idxs[n],idxs[n+1]-offset)) or $(idxs[n+1]-offset-idxs[n]+1) elements")
        flush(stdout)
        subset = Set(visited[idxs[n]:(idxs[n+1]-offset)]);
        for i in 1:max_iter
            #println("=======$(i)=======")
            #flush(stdout)
            last_stamp = now()
            Δ = 0
            #ssum = 0
            #old_V = copy(V)
            for s in subset#visited[idxs[n]:(idxs[n+1]-offset)]##ProgressBar
            #for idx in idxs[n]:(idxs[n+1]-offset)
                #s = visited[idx]
                v = V[s]
                nv = get_new_value(s, bs, bbs, V, Ps)
                Δ = max(Δ, abs(v-nv))
                V[s] = nv
            end
            println("Thread $(n) iteration $(i) took $(now() - last_stamp) with Δ: $(Δ)")
            flush(stdout)
            if Δ < θ
                # println("Value iteration took $(now() - start_time)")
                # flush(stdout)
                # return V, times, ss, deltas
                println("Thread $(n) took $(now() - start_time)")
                flush(stdout)
                break
            end
        end
    end
    vi_start_time = now()
    for i in 1:max_iter
        println("=======$(i)=======")
        flush(stdout)
        last_stamp = now()
        Δ = 0
        for s in visited
            v = V[s]
            nv = get_new_value(s, bs, bbs, V, Ps)
            Δ = max(Δ, abs(v-nv))
            V[s] = nv
        end
        println("Took $(now() - last_stamp)")
        flush(stdout)
        println("Δ: $(Δ)")
        flush(stdout)
        if Δ < θ
            println("Value iteration took $(now() - vi_start_time)")
            flush(stdout)
            println("Whole thing took $(now() - start_time)")
            return V
        end
    end
    println("Maximum number of iterations reached (Delta: $(Δ))")
    flush(stdout)
    println("Value iteration took $(now() - vi_start_time)")
    flush(stdout)
    println("Whole thing took $(now() - start_time)")
    return V
end


bs = UInt32.(2 .^ (0:31));
bbs = UInt32.(3 .^ (0:7));

s = start_state_int(bs, bbs; N=7);
s_start = s;
# println("Starting BFS...")
# start_bfs_time = now()
# flush(stdout)
#@time visited, leaf_nodes, inv_graph, depths, graph = bfs(s_start, bs, bbs);
@time visited, leaf_nodes = bfs(s_start, bs, bbs);

#@time inv_depths = inv_bfs(leaf_nodes, inv_graph; max_iter=10000000);
#@time inv_depths = inv_bfs([s_start], graph; max_iter=10000000);
# @save "visited.jld2" visited
# @save "leaf_nodes.jld2" leaf_nodes

println("Loading states...")
loading_states_time = now()
flush(stdout)
visited = load("visited_half_smart.jld2")["visited"]
leaf_nodes = load("leaf_nodes_half_smart.jld2")["leaf_nodes"]

println("Finished getting all states. Took $(now()-loading_states_time)")
flush(stdout)
println("Starting value iteration...")
flush(stdout)
epsilon = 0.0000001;
max_iter = 400;#400;
println("epsilon: $(epsilon), max_iter: $(max_iter)")
flush(stdout)


no_leafs = collect(setdiff(visited, leaf_nodes));
n_threads = Threads.nthreads();
bin_size = ceil(Int, length(no_leafs) / n_threads)
idxs = push!(collect(1:bin_size:length(no_leafs)), length(no_leafs))

#V = iterate_smart(idxs, no_leafs, leaf_nodes, epsilon, bs, bbs; max_iter=max_iter);
V3 = value_iteration_smart(visited, leaf_nodes, epsilon, bs, bbs; max_iter=max_iter, N=2);
#V = value_iteration(no_leafs, leaf_nodes, epsilon, bs, bbs; max_iter=max_iter);
VV = value_iteration_parallel_full(no_leafs, leaf_nodes, epsilon, bs, bbs, idxs; max_iter=max_iter);
println("Everything took $(now()-loading_states_time)")
flush(stdout)

mm = []
sp = 0
spp = 0
for s in no_leafs
    ss = possible_neighbours(s, 0, bs, bbs)[1]
    ss -= bs[32]
    d = V[s]+V[ss]
    if d == -79.53145755929853
        sp = s
    end
    if d == 8.622942736193263
        spp = s
    end
    push!(mm, d)
end

pretty_print(sss, bs, bbs)
pretty_print(ss, bs, bbs)
pretty_print(sp, bs, bbs)
pretty_print(spp, bs, bbs)

sss = possible_neighbours(sp, 0, bs, bbs)[1]
sss -= bs[32]

V = load("V_half_smart.jld2")["V"]



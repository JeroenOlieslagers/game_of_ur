using StaticArrays

function optimal_move(ss, moves, V, bs)
    max_V = -Inf
    opt_move = nothing
    #vs =[]
    for (s, m) in zip(ss, moves)
        if m == -1
            break
        end
        if check_bit(s, 32)
            s -= bs[32]
            v = -V[s]
        else
            v = V[s]
        end
        if v > max_V
            max_V = v
            opt_move = m
        end
        #push!(vs, v)
    end
    if opt_move === nothing
        throw(ErrorException("No optimal move found???"))
    else
        return opt_move#, vs
    end
end

function get_blockages(states, bs, bbs)
    blockages = [[] for _ in 1:4]
    blockages_s = [Set{UInt32}() for _ in 1:4]
    for s in ProgressBar(states)
        for d in 1:4
            ss, moves = neighbours_moves(s, d, bs, bbs)
            if moves[1] == (0, 0, 0, 0)
                self_stuck, _ = piece_locs(s, bs, bbs; N=4)
                push!(blockages[d], self_stuck[self_stuck .!= 0])
                push!(blockages_s[d], s)
            end
        end
    end
    return blockages, blockages_s
end

function get_other_blockages(s, bs, bbs, blockages_s)
    # calculate which ways other is blocked
    other_blocked = Int[]
    sp = flip_turn(s, bs, bbs)
    sp -= bs[32]
    for i in 1:4
        if sp in blockages_s[i]
            push!(other_blocked, i)
        end
    end
    return other_blocked
end

function blockage_analysis(states, bs, bbs, blockages_s, V)
    to_blockage_tile = zeros(Int, 16, 4, 6)
    for s in ProgressBar(states)
        other_blocked = get_other_blockages(s, bs, bbs, blockages_s)
        for d in 1:4
            ss, moves = neighbours_moves(s, d, bs, bbs)
            # skip if blocked or only one move available
            if length(moves) == 1
                continue
            end
            opt_move = optimal_move(ss, moves, V, bs)
            for (sp, move) in zip(ss, moves)
                # if it is other's turn (not on rosette)
                if ~move[4]
                    # find which of other's blockages have been cleared
                    for ii in other_blocked
                        if sp - bs[32] ∉ blockages_s[ii]
                            to_blockage_tile[move[2]+1, ii, 4] += 1
                            if move == opt_move
                                to_blockage_tile[move[2]+1, ii, 5] += 1
                                if move[3]
                                    to_blockage_tile[move[2]+1, ii, 6] += 1
                                end
                            end
                        end
                    end
                    sp = flip_turn(sp, bs, bbs)
                end
                # find which blockages each move causes
                for ii in 1:4
                    if sp in blockages_s[ii]
                        to_blockage_tile[move[2]+1, ii, 1] += 1
                        if move == opt_move
                            to_blockage_tile[move[2]+1, ii, 2] += 1
                            if move[3]
                                to_blockage_tile[move[2]+1, ii, 3] += 1
                            end
                        end
                    end
                end
            end
        end
    end
    return to_blockage_tile
end

function is_run(m, other_locs)::Bool
    unsafe_other_locs = other_locs[4 .< other_locs .< 13]
    # if current move starts in safe tile, is not a run
    if length(unsafe_other_locs) == 0 || m[1] > 12
        return false
    else
        # if other piece in unsafe tiles is before current move from tile,
        # it is a run
        return maximum(unsafe_other_locs) < m[1]
    end
end

function is_chase(m, other_locs)::Bool
    unsafe_other_locs = other_locs[4 .< other_locs .< 13]
    # if current move starts in safe tile, is not a chase
    if length(unsafe_other_locs) == 0 || m[1] < 5
        return false
    else
        # if other piece in unsafe tiles is ahead of current move to tile,
        # it is a chase
        return maximum(unsafe_other_locs) >= m[2]
    end
end

function is_new(m, other_locs)::Bool
    return m[1] == 0
end

function run_chase_new_analysis(states, bs, bbs, V)
    comps = zeros(Int, 16, 16, 3, 3, 3)
    alternative_moves = zeros(Int, 16, 4, 3, 2)
    comparison_funcs = [is_run, is_chase, is_new]
    Ns = zeros(Int, 4, 3, 3)
    for s in ProgressBar(states)
        for d in 1:4
            ss, moves = neighbours_moves(s, d, bs, bbs)
            # skip if blocked or only one move available
            if length(moves) == 1
                continue
            end
            # check if current state has any available moves that capture
            # or move onto rosette
            capture_avail = false
            rosette_avail = false
            for move in moves
                if move[3]
                    capture_avail = true
                end
                if move[4]
                    rosette_avail = true
                end
            end
            avails = [capture_avail, rosette_avail, true]
            self_locs, other_locs = piece_locs(s, bs, bbs; N=4)
            opt_move = optimal_move(ss, moves, V, bs)
            for move in moves
                # check if each condition holds
                for (i, comparison_func) in enumerate(comparison_funcs)
                    # check whether a capture and rosette move is available
                    for (j, avail) in enumerate(avails)
                        if comparison_func(move, other_locs)
                            if avail
                                comps[move[1]+1, move[2]+1, i, j, 1] += 1
                                if move == opt_move
                                    comps[move[1]+1, move[2]+1, i, j, 2] += 1
                                    if move[3]
                                        comps[move[1]+1, move[2]+1, i, j, 3] += 1
                                    end
                                    # skip next part if available is true for all actions
                                    if j == 3
                                        continue
                                    end
                                    # check alternative moves
                                    for move_ in moves
                                        if move_ == move
                                            continue
                                        end
                                        if move_[j+2]
                                            Ns[d, i, j] += 1
                                            alternative_moves[move_[1]+1, d, i, j] += 1
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return comps, alternative_moves, Ns
end

function get_tile_transition_matrix(states, bs, bbs, V)
    M = zeros(Int, 16, 16, 4)
    moves = zeros(Int8, 7)
    ns = zeros(UInt32, 7)
    for s in ProgressBar(states)
        for d in 1:4
            neighbours_moves!(ns, moves, s, d, bs, bbs)
            # skip if blocked or only one move available
            if moves[2] == -1
                continue
            end
            opt_move = optimal_move(ns, moves, V, bs)
            to = opt_move+d
            M[opt_move+1, to+1, 2] += 1
            if is_capture(s, to, bs, bbs)
                M[opt_move+1, to+1, 4] += 1
            end
            for move in moves
                if move == -1
                    break
                end
                to = move+d
                M[move+1, to+1, 1] += 1
                if is_capture(s, to, bs, bbs)
                    M[move+1, to+1, 3] += 1
                end
            end
        end
    end
    return M
end

function value_of_random_action!(ns, s, bs, bbs, V, Ps)
    nv = zero(Float64)
    # Rolling a 0 (negative because turn change)
    sp = flip_turn(s, bs, bbs)
    sp -= bs[32]
    nv += -Ps[1]*V[sp]
    # Other rolls don't just flip turn
    for d in 1:4
        neighbours!(ns, s, d, bs, bbs)
        # Get mean across possible actions (random policy)
        nvv = 0
        counter = 0
        for neighbour in ns
            # Not all states have 7 possible actions
            if neighbour == 0
                break
            end
            neighbour, factor = turn_change(neighbour, bs)
            # Negate if turn change
            counter += 1
            nvv += factor*V[neighbour]
        end
        # Expectation of value (+1 because julia 1 indexes)
        nv += Ps[d+1]*(nvv/counter)
    end
    return nv
end

### CAPTURING CENTRAL ROSETTE

central_rosette = zeros(Int, 51, 51, 4, 4, 6);
for s in ProgressBar(states)
    for d in 1:4
        ss, moves = neighbours_moves(s, d, bs, bbs)
        # skip if blocked or only one move available
        if length(moves) == 1
            continue
        end
        self, other = advancement(s, bs, bbs; N=4, score=true, def=0)
        self_score, other_score = player_score(s, bs, bbs; N=4)
        opt_move = optimal_move(ss, moves, V, bs)
        for move in moves
            # the "~" is to make sure if there are two captures and one is optimal,
            # the other doesn't get counted
            if move[2] == 8
                if ~(opt_move[2] == 8)
                    central_rosette[self+1, other+1, self_score+1, other_score+1, 2] += 1
                end
                if move == opt_move
                    central_rosette[self+1, other+1, self_score+1, other_score+1, 2] += 1
                    central_rosette[self+1, other+1, self_score+1, other_score+1, 1] += 1
                end
            end
            if move[3]
                if ~opt_move[3]
                    central_rosette[self+1, other+1, self_score+1, other_score+1, 4] += 1
                end
                if move == opt_move
                    central_rosette[self+1, other+1, self_score+1, other_score+1, 4] += 1
                    central_rosette[self+1, other+1, self_score+1, other_score+1, 3] += 1
                end
            end
            if move[4]
                if ~opt_move[4]
                    central_rosette[self+1, other+1, self_score+1, other_score+1, 6] += 1
                end
                if move == opt_move
                    central_rosette[self+1, other+1, self_score+1, other_score+1, 6] += 1
                    central_rosette[self+1, other+1, self_score+1, other_score+1, 5] += 1
                end
            end
        end
    end
end


plot(layout=grid(4, 3), size=(400, 500))
for i in 1:3
    for j in 1:4
        # CHANGE THESE TWO LINES BELOW, THE LAST INDEX
        a = sum(central_rosette[:, :, i, j, 2], dims=1)
        b = sum(central_rosette[:, :, i, j, 2], dims=2)
        # xx = length(a[a .> 0])
        # yy = min(length(b[b .> 0]), 50)
        # xx = min(length(a[a .> 0]), 44)
        # yy = min(length(b[b .> 0]), 50)
        xx = length(a[a .> 0])
        yy = min(length(b[b .> 0]), 46)
        #heatmap!(1:yy, 0:xx-1, (central_rosette[2:yy+1, 1:xx, i, j, 5] ./ central_rosette[2:yy+1, 1:xx, i, j, 6])', sp=(j-1)*3 + i, colorbar=false, clim=(0, 1), xticks=[], yticks=[], tick_direction=:out, ylims=(-1, xx), xlims=(0, yy+2))
        #heatmap!(1:yy, 4:xx+5, (central_rosette[2:yy+1, 5:xx+6, i, j, 3] ./ central_rosette[2:yy+1, 5:xx+6, i, j, 4])', sp=(j-1)*3 + i, colorbar=false, clim=(0, 1), xticks=[], yticks=[], tick_direction=:out, ylims=(4, xx+6), xlims=(0, yy+2))
        heatmap!(5:yy+4, 0:xx-1, (central_rosette[6:yy+5, 1:xx, i, j, 1] ./ central_rosette[6:yy+5, 1:xx, i, j, 2])', sp=(j-1)*3 + i, colorbar=false, clim=(0, 1), xticks=[], yticks=[], tick_direction=:out, ylims=(-1, xx), xlims=(4, yy+5))
        if j == 4
            #plot!(xticks=[1, yy], sp=(j-1)*3 + i)
            #plot!(xticks=[1, yy], sp=(j-1)*3 + i)
            plot!(xticks=[5, yy+4], sp=(j-1)*3 + i)
        end
        if i == 1
            #plot!(yticks=[0, xx-1], sp=(j-1)*3 + i)
            #plot!(yticks=[5, xx+5], sp=(j-1)*3 + i)
            plot!(yticks=[0, xx-1], sp=(j-1)*3 + i)
        end
        #heatmap!(0:50, 0:50, ZZ[1:51, 1:51, i, j, 1], sp=(i-1)*4 + j, colorbar=false)#, clim=(0, 15))
    end
end
plot!()

heatmap(0:9, 0:9, (central_rosette[1:10, 1:10, 3, 4, 5] ./ central_rosette[1:10, 1:10, 3, 4, 6])')
heatmap(0:9, 0:9, central_rosette[1:10, 1:10, 3, 4, 6]')

##### TRANSITION STRUCTURE
M = get_tile_transition_matrix(states, bs, bbs, V);

#heatmap(0:15, 0:15, M[:, :, 1] / maximum(M[:, :, 1]), yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
heatmap(0:15, 0:15, M[:, :, 2] ./ M[:, :, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
#heatmap(0:15, 0:15, M[:, :, 3] / maximum(M[:, :, 3]), yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
heatmap(0:15, 0:15, M[:, :, 4] ./ M[:, :, 3], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))

function transition_plot(M)
    l = @layout [grid(1, 4, widths=[0.45, 0.25, 0.25, 0.05])]
    plot(layout=l, size=(712, 300), top_margin=3Plots.mm, guidefont=font(14), dpi=300, ylims=(-1, 14.5))
    heatmap!(0:15, 0:15, M[:, :, 2] ./ M[:, :, 1], title="opt / all", sp=1, yflip=true, xmirror=true, axis=([], false), clim=(0, 1), colorbar=false)
    heatmap!(0:15, 0:15, M[:, :, 4] ./ M[:, :, 3], title="opt cap / cap", xlim=(4, 14), sp=2, yflip=true, xmirror=true, axis=([], false), clim=(0, 1), colorbar=false ,colorbar_title="Proportion")
    f = M[:, :, 4] ./ M[:, :, 2]
    f[f .== 0] .= NaN
    heatmap!(0:15, 0:15, f, title="opt cap / opt", xlim=(4, 14), sp=3, yflip=true, xmirror=true, axis=([], false), clim=(0, 1), colorbar=false)
    scatter!([], [], zcolor=[NaN], sp=4, clim=(0, 1), axis=([], false), label=false, framestyle=:none, colorbar_title="Proportion", colorbar_titlefontsize=12)
    annotate!(-0.5, 7.5, sp=1, text("from", 14, rotation=90))
    annotate!(1, 7.5, sp=1, ("→", 14))
    for j in 1:3
        if j > 1
            annotate!(9, -0.5, sp=j, ("to", 14))
            annotate!(9, 0.5, sp=j, ("↓", 14))
            annotate!(4, 1, sp=j, "1")
            annotate!(4, 2, sp=j, "2")
            annotate!(4, 3, sp=j, "3")
            annotate!(4, 4, sp=j, "4")
            for i in 5:11
                annotate!(i, i-5, sp=j, "$(i)")
                annotate!(i, i, sp=j, "$(i)")
            end
            annotate!(12, 7, sp=j, "12")
        else
            annotate!(7.5, -0.5, sp=j, ("to", 14))
            annotate!(7.5, 0.5, sp=j, ("↓", 14))
            annotate!(0, 0, sp=j, "0")
            for i in 1:14
                annotate!(i, max(-1, i-5), sp=j, "$(i)")
                annotate!(i, i, sp=j, "$(i)")
            end
            annotate!(15, 10, sp=j, "15")
        end
    end
    display(plot!())
end

transition_plot(M)

ls = []
for s in states
    sp = flip_turn(s, bs, bbs)
    sp -= bs[32]
    sm = V[s] + V[sp]
    if sm < 0
        push!(ls, s)
    end
end

s = locs_to_s(Int[1, 5], Int[], 0, 1, bs, bbs)

s = rand(states)

sp = flip_turn(s, bs, bbs)
sp -= bs[32]

draw_board(s, bs, bbs)

ns, moves = neighbours_moves(s, 3, bs, bbs)
move, vs = optimal_move(ns, moves, V, bs)

####
bs, bbs = get_bases()
s_start = start_state(bs, bbs; N=4)
visited, leafs = bfs(s_start, bs, bbs)
states = union(values(visited)...)
# V = load("V.jld2")["V"]

######## RUN CHASE NEW STORY

comps, dice_tiles, Ns = run_chase_new_analysis(states, bs, bbs, V);

# run
heatmap(0:15, 0:15, comps[:, :, 1, 3, 2] ./ comps[:, :, 1, 3, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
# run & capture avail
heatmap(0:15, 0:15, comps[:, :, 1, 1, 1] ./ comps[:, :, 1, 3, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
heatmap(0:15, 0:15, comps[:, :, 1, 1, 2] ./ comps[:, :, 1, 1, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
# run & rosette avail
heatmap(0:15, 0:15, comps[:, :, 1, 2, 2] ./ comps[:, :, 1, 2, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))

# chase
heatmap(0:15, 0:15, comps[:, :, 2, 3, 2] ./ comps[:, :, 2, 3, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
# chase & capture avail
heatmap(0:15, 0:15, comps[:, :, 2, 1, 1] ./ comps[:, :, 2, 3, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
heatmap(0:15, 0:15, comps[:, :, 2, 1, 2] ./ comps[:, :, 2, 1, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
heatmap(0:15, 0:15, comps[:, :, 2, 1, 3] ./ comps[:, :, 2, 1, 2], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
# chase & rosette avail
heatmap(0:15, 0:15, comps[:, :, 2, 2, 2] ./ comps[:, :, 2, 2, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
heatmap(0:15, 0:15, comps[:, :, 2, 2, 3] ./ comps[:, :, 2, 2, 2], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))

# new
heatmap(0:15, 0:15, comps[:, :, 3, 3, 2] ./ comps[:, :, 3, 3, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
# new & capture avail
heatmap(0:15, 0:15, comps[:, :, 3, 1, 1] ./ comps[:, :, 3, 3, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
heatmap(0:15, 0:15, comps[:, :, 3, 1, 2] ./ comps[:, :, 3, 1, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
# new & rosette avail
heatmap(0:15, 0:15, comps[:, :, 3, 2, 1] ./ comps[:, :, 3, 3, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))
heatmap(0:15, 0:15, comps[:, :, 3, 2, 2] ./ comps[:, :, 3, 2, 1], yflip=true, xmirror=true, xlabel="to", ylabel="from", xticks=0:15, yticks=0:15, size=(450, 400), clim=(0, 1))

N_ = 4
for j in 1:3
    plot(layout=grid(2, N_), size=(N_*130, 2*300), fontfamily="Helvetica", colorbar=false, yflip=true, xticks=:none, yticks=:none, legend=false, colorbar_title="", dpi=300, clim=(0, 1))#Occupancy (%)", dpi=300)
    for i in 1:N_
        plot!(title=i, sp=i)
        draw_board_heatmap_2(dice_tiles[:, i, j, 1] ./ Ns[i, j, 1], i)
        draw_board_heatmap_2(dice_tiles[:, i, j, 2] ./ Ns[i, j, 2], 4+i)
    end
    display(plot!())
end

####### BLOCKAGES STORY

blockages, blockages_s = get_blockages(states, bs, bbs);

to_blockage_tile = blockage_analysis(states, bs, bbs, blockages_s, V);

a = zeros(Float64, 16, 4)
for i in 1:4
    for blocks in blockages[i]
        a[blocks .+ 1, i] .+= 1
    end
    a[:, i] ./= length(blockages[i])
end

N_ = 4
plot(layout=grid(1, N_), size=(N_*130, 300), fontfamily="Helvetica", colorbar=false, yflip=true, xticks=:none, yticks=:none, legend=false, colorbar_title="", dpi=300, clim=(0, 1))#Occupancy (%)", dpi=300)
for i in 1:N_
    plot!(title=i, sp=i)
    draw_board_heatmap_2(a[:, i], i)
end
plot!()

N_ = 4
plot(layout=grid(2, N_), size=(N_*130, 2*300), fontfamily="Helvetica", colorbar=false, yflip=true, xticks=:none, yticks=:none, legend=false, colorbar_title="", dpi=300, clim=(0, 1))#Occupancy (%)", dpi=300)
for i in 1:N_
    plot!(title=i, sp=i)
    draw_board_heatmap_2(to_blockage_tile[:, i, 2] ./ to_blockage_tile[:, i, 1], i)
    draw_board_heatmap_2(to_blockage_tile[:, i, 3] ./ to_blockage_tile[:, i, 2], 4+i)
end
plot!()

# unblocking the enemy
N_ = 4
plot(layout=grid(2, N_), size=(N_*130, 2*300), fontfamily="Helvetica", colorbar=false, yflip=true, xticks=:none, yticks=:none, legend=false, colorbar_title="", dpi=300, clim=(0, 1))#Occupancy (%)", dpi=300)
for i in 1:N_
    plot!(title=i, sp=i)
    draw_board_heatmap_2(to_blockage_tile[:, i, 5] ./ to_blockage_tile[:, i, 4], i)
    draw_board_heatmap_2(to_blockage_tile[:, i, 6] ./ to_blockage_tile[:, i, 5], 4+i)
end
plot!()

### WHEN DOES OPTIMAL BECOME EFFECTIVE

ls = []
ss = zeros(Int, 16, 2, 4);
Ns = zeros(Int, 2, 4);
ns = zeros(UInt32, 7);
Ps = get_Ps();
m = 0;
Z = zeros(Float64, 51, 51, 4, 4, 2);
ZZ = zeros(Float64, 51, 51, 4, 4);
ZZZ = zeros(Float64, 61, 61, 2);
for s in ProgressBar(states)
    self, other = advancement(s, bs, bbs; N=4, score=true, def=0)
    self_, other_ = advancement(s, bs, bbs; N=4)
    self_score, other_score = player_score(s, bs, bbs; N=4)
    self_locs, other_locs = piece_locs(s, bs, bbs; N=4)
    # if player has no choice
    if maximum([length(neighbours(s, d, bs, bbs)) for d in 1:4]) < 2
        ZZ[self+1, other+1, self_score+1, other_score+1] += 1
        continue
    end
    nv = get_new_value!(ns, s, bs, bbs, V, Ps)
    vr = value_of_random_action!(ns, s, bs, bbs, V, Ps)
    dif = nv - vr
    Z[self+1, other+1, self_score+1, other_score+1, 1] += dif
    Z[self+1, other+1, self_score+1, other_score+1, 2] += 1
    ZZZ[self_+1, other_+1, 1] += dif
    ZZZ[self_+1, other_+1, 2] += 1
    part = 1
    if dif > 20
        part = 4
    elseif dif > 15
        part = 3
    elseif dif > 12.5
        part = 2
    end
    Ns[1, part] += 1
    Ns[2, part] += 1
    for self_loc in self_locs
        if self_loc == 0
            break
        end
        ss[self_loc+1, 1, part] += 1
    end
    for other_loc in other_locs
        if other_loc == 0
            break
        end
        ss[other_loc+1, 2, part] += 1
    end
    push!(ls, dif)
end

heatmap(0:60, 0:60, ZZZ[:, :, 1] ./ ZZZ[:, :, 2], size=(400, 400))

N_ = 4
plot(layout=grid(2, N_), size=(N_*130, 2*300), fontfamily="Helvetica", colorbar=false, yflip=true, xticks=:none, yticks=:none, legend=false, colorbar_title="", dpi=300, clim=(0, 1))#Occupancy (%)", dpi=300)
for i in 1:N_
    #plot!(title=i, sp=i)
    draw_board_heatmap_2(ss[:, 1, i] ./ Ns[1, i], i)
    draw_board_heatmap_2(ss[:, 2, i] ./ Ns[2, i], N_+i; other=true)
end
display(plot!())

plot(layout=grid(3, 4), size=(600, 400))
for i in 1:3
    for j in 1:4
        a = sum(Z[:, :, i, j, 2], dims=1)
        b = sum(Z[:, :, i, j, 2], dims=2)
        xx = length(a[a .> 0])
        yy = min(length(b[b .> 0]), 50)
        heatmap!(0:xx-1, 1:yy, Z[2:yy+1, 1:xx, i, j, 1] ./ Z[2:yy+1, 1:xx, i, j, 2], sp=(i-1)*4 + j, colorbar=false, clim=(0, 15), xticks=[], yticks=[])
        if i == 3
            plot!(xticks=[0, xx-1], sp=(i-1)*4 + j)
        end
        if j == 1
            plot!(yticks=[1, yy], sp=(i-1)*4 + j)
        end
        #heatmap!(0:50, 0:50, ZZ[1:51, 1:51, i, j, 1], sp=(i-1)*4 + j, colorbar=false)#, clim=(0, 15))
    end
end
plot!()


plot(layout=grid(4, 1), size=(600, 600), ylim=(0, 10))
for i in 1:4
    for j in 1:4
        denom = sum(Z[:, :, i, j, 2], dims=2)
        num = sum(Z[:, :, i, j, 1], dims=2)
        plot!(0:50, (num ./ denom), sp=i, label=nothing)
        #plot!(0:50, denom', sp=j, label=nothing)
    end
end
plot!()
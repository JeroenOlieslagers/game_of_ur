### TACTICS


function captures_possible(s, tile, bs, bbs)
    captures = 0
    if tile > 0x04 && tile < UInt8(13) && tile !== 0x08
        if tile < 0x08
            for i in max(20, 20+tile-5):23
                if check_bit(s, i)
                    captures += 1
                end
            end
        end
        if tile > 0x05
            for i in max(5, tile-4):(tile-1)
                if check_trit(s, i-4, bs, bbs) == 2
                    captures += 1
                end
            end
        end
    end
    return captures
end

function is_rosette(tile::UInt8)::Bool
    if tile in [4, 8, 14]
        return true
    else
        return false
    end
end

function capture!(stats::stats_type, sp::UInt32, move::Tuple{UInt8, UInt8, Bool, Bool}, is_optimal::Bool)::Nothing
    from = move[1]
    if move[3]
        stats[from+1, 1, 1] += 1
        if is_optimal
            stats[from+1, 2, 1] += 1
        end
    end
    return nothing
end

function uncaptured!(stats::stats_type, sp::UInt32, move::Tuple{UInt8, UInt8, Bool, Bool}, is_optimal::Bool)::Nothing
    from = move[1]
    to = move[2]
    captures_from = captures_possible(sp, from, bs, bbs)
    captures_to = captures_possible(sp, to, bs, bbs)
    if captures_to < captures_from
        stats[from+1, 1, 2] += 1
        if is_optimal
            stats[from+1, 2, 2] += 1
        end
    end
    return nothing
end

function rosette!(stats::stats_type, sp::UInt32, move::Tuple{UInt8, UInt8, Bool, Bool}, is_optimal::Bool)::Nothing
    to_rosette = move[4]
    from = move[1]
    if to_rosette
        stats[from+1, 1, 3] += 1
        if is_optimal
            stats[from+1, 2, 3] += 1
        end
    end
    if is_rosette(from)
        stats[from+1, 1, 4] += 1
        if is_optimal
            stats[from+1, 2, 4] += 1
        end
    end
    return nothing
end

function unsafe!(stats::stats_type, sp::UInt32, move::Tuple{UInt8, UInt8, Bool, Bool}, is_optimal::Bool)::Nothing
    to = move[2]
    from = move[1]
    onto_unsafe = 0x04 < to < UInt8(13)
    from_unsafe = 0x04 < from < UInt8(13)
    if onto_unsafe && ~from_unsafe
        stats[from+1, 1, 5] += 1
        if is_optimal
            stats[from+1, 2, 5] += 1
        end
    end
    if ~onto_unsafe && from_unsafe
        stats[from+1, 1, 6] += 1
        if is_optimal
            stats[from+1, 2, 6] += 1
        end
    end
    return nothing
end

function from_rosette_and_capture!(stats::stats_type, sp::UInt32, move::Tuple{UInt8, UInt8, Bool, Bool}, is_optimal::Bool)::Nothing
    to = move[2]
    from = move[1]
    if is_rosette(from) && move[3]
        stats[from+1, 1, 7] += 1
        if is_optimal
            stats[from+1, 2, 7] += 1
        end
    end
    onto_unsafe = 0x04 < to < UInt8(13)
    from_unsafe = 0x04 < from < UInt8(13)
    if onto_unsafe && ~from_unsafe && move[3]
        stats[from+1, 1, 8] += 1
        if is_optimal
            stats[from+1, 2, 8] += 1
        end
    end
    return nothing
end

function get_all_tactics(states, tactics, bs, bbs, V)
    stats = @MArray zeros(Int, 15, 2, 8)
    dice_stats = @MArray zeros(Int, 15, 4, 4, 4, 4)
    total = zeros(Int, 15)
    total_opt = zeros(Int, 15)
    stuck = zeros(Int, 15, 4, 4)
    to_all_tiles = zeros(Int, 16, 2, 16)
    to_all_unsafe_captures = zeros(Int, 16, 2, 16)
    trans_matrix = zeros(Int, 16, 16, 4)
    for s in ProgressBar(states)
        for d in 1:4
            ss, moves = neighbours_moves(s, d, bs, bbs)
            self, other = player_score(s, bs, bbs; N=4)
            if moves[1] == (0, 0, 0, 0)
                self_stuck, _ = piece_locs(s, bs, bbs; N=4)
                for piece in self_stuck
                    if piece == 0
                        break
                    end
                    stuck[piece+1, self+1, other+1] += 1
                end
                continue
            end
            if length(moves) == 1
                continue
            end
            opt_move = optimal_move(ss, moves, V)
            total_opt[opt_move[1]+1] += 1
            dice_stats[opt_move[1]+1, 2, d, self+1, other+1] += 1
            to_all_tiles[opt_move[1]+1, 2, opt_move[2]+1] += 1
            trans_matrix[opt_move[1]+1, opt_move[2]+1, 2] += 1
            if opt_move[3]
                trans_matrix[opt_move[1]+1, opt_move[2]+1, 4] += 1
                to_all_unsafe_captures[opt_move[1]+1, 2, opt_move[2]+1] += 1
            end
            for (sp, move) in zip(ss, moves)
                total[move[1]+1] += 1
                dice_stats[move[1]+1, 1, d, self+1, other+1] += 1
                to_all_tiles[move[1]+1, 1, move[2]+1] += 1
                trans_matrix[move[1]+1, move[2]+1, 1] += 1
                if move[3]
                    trans_matrix[move[1]+1, move[2]+1, 3] += 1
                    to_all_unsafe_captures[move[1]+1, 1, move[2]+1] += 1
                end
                if move[3]
                    dice_stats[move[1]+1, 3, d, self+1, other+1] += 1
                    if move == opt_move
                        dice_stats[move[1]+1, 4, d, self+1, other+1] += 1
                    end
                end
                for tactic! in tactics
                    tactic!(stats, sp, move, move == opt_move)
                end
            end
        end
    end
    return stats, total, total_opt, dice_stats, stuck, to_all_tiles, to_all_unsafe_captures, trans_matrix
end


tactics = [capture!, uncaptured!, rosette!, unsafe!, from_rosette_and_capture!]


stats_type = MArray{Tuple{15, 2, 8}, Int, 3, 240}

stats, total, total_opt, dice_stats, stuck, to_all_tiles, to_all_unsafe_captures, trans_matrix = get_all_tactics(states, tactics, bs, bbs, V)

plot(layout=grid(1, N_), size=(N_*130, 300), fontfamily="Helvetica", colorbar=false, yflip=true, xticks=:none, yticks=:none, legend=false, colorbar_title="", dpi=300, clim=(0, 1))#Occupancy (%)", dpi=300)

draw_board_heatmap_2(stuck[:, 1] ./ stuck[:, 1], 1)
draw_board_heatmap_2(stuck[:, 2] ./ stuck[:, 1], 2)
draw_board_heatmap_2(stuck[:, 3] ./ stuck[:, 1], 3)
draw_board_heatmap_2(stuck[:, 4] ./ stuck[:, 1], 4)

N_ = 8
plot(layout=grid(2, N_), size=(N_*130, 2*300), fontfamily="Helvetica", colorbar=false, yflip=true, xticks=:none, yticks=:none, legend=false, colorbar_title="", dpi=300, clim=(0, 1))#Occupancy (%)", dpi=300)
titles = ["Captures", "Uncapture", "Onto rosette", "From rosette", "To unsafe", "Away unsafe", "R & C", "To unsafe & C"]
for i in 1:N_
    plot!(title=titles[i], sp=i)
    draw_board_heatmap_2(stats[:, 1, i] ./ total, i)
    draw_board_heatmap_2(stats[:, 2, i] ./ stats[:, 1, i], N_ + i)
end
plot!()

N_ = 4
score = 4
for score in 1:4
    plot(layout=grid(2, N_), size=(N_*130, 2*300), fontfamily="Helvetica", colorbar=false, yflip=true, xticks=:none, yticks=:none, legend=false, colorbar_title="", dpi=300, clim=(0, 1))#Occupancy (%)", dpi=300)
    for i in 1:N_
        plot!(title=i, sp=i)
        # draw_board_heatmap_2(dice_stats[:, 3, i] ./ total, i)
        # draw_board_heatmap_2(dice_stats[:, 4, i] ./ dice_stats[:, 3, i], N_ + i)
        draw_board_heatmap_2(dice_stats[:, 2, i, score] ./ dice_stats[:, 1, i, score], i)
        draw_board_heatmap_2(dice_stats[:, 4, i, score] ./ dice_stats[:, 3, i, score], N_ + i)
    end
    display(plot!())
end


N_ = 16
plot(layout=grid(2, N_), size=(N_*130, 2*300), fontfamily="Helvetica", colorbar=false, yflip=true, xticks=:none, yticks=:none, legend=false, colorbar_title="", dpi=300, clim=(0, 1))
for i in 1:N_
    plot!(title=i, sp=i)
    draw_board_heatmap_2(to_all_tiles[:, 2, i] ./ to_all_tiles[:, 1, i], i)
    draw_board_heatmap_2(to_all_unsafe_captures[:, 2, i] ./ to_all_unsafe_captures[:, 1, i], N_ + i)
end
plot!()
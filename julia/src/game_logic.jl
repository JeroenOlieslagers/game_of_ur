const State = UInt32

function get_bases()::Tuple{Vector{UInt32}, Vector{UInt32}}
    return UInt32.(2 .^ (0:31)), UInt32.(3 .^ (0:7))
end

function start_state(bs::Vector{UInt32}, bbs::Vector{UInt32}; N::Int=7)::UInt32
    0 <= N <= 7 || throw(ArgumentError("N must be between 0 and 7"))

    s = UInt32(0)
    for i in 26:28
        if (N ÷ 2^(28 - i)) % 2 == 1
            s += bs[i]
        end
    end
    for i in 29:31
        if (N ÷ 2^(31 - i)) % 2 == 1
            s += bs[i]
        end
    end
    return s
end

check_bit(s::UInt32, n::Int)::Bool = ((s >> (n - 1)) & 1) == 1

function check_trit(s::UInt32, n::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::Int
    return Int((s % bs[14] ÷ bbs[n]) % bbs[2])
end

function how_many_home(s::UInt32)::Tuple{Int, Int}
    self_home = 4 * check_bit(s, 26) + 2 * check_bit(s, 27) + check_bit(s, 28)
    other_home = 4 * check_bit(s, 29) + 2 * check_bit(s, 30) + check_bit(s, 31)
    return self_home, other_home
end

has_home(s::UInt32)::Bool = check_bit(s, 26) || check_bit(s, 27) || check_bit(s, 28)

function move_out(s::UInt32, bs::Vector{UInt32})::UInt32
    if check_bit(s, 28)
        return s - bs[28]
    end
    if check_bit(s, 27)
        return s - bs[27] + bs[28]
    end
    check_bit(s, 26) || throw(ErrorException("Cannot move out; no self pieces are home"))
    return s - bs[26] + bs[27] + bs[28]
end

function move_in(s::UInt32, bs::Vector{UInt32})::UInt32
    if !check_bit(s, 31)
        return s + bs[31]
    end
    if !check_bit(s, 30)
        return s + bs[30] - bs[31]
    end
    !check_bit(s, 29) || throw(ErrorException("Cannot move in; other home is full"))
    return s + bs[29] - bs[30] - bs[31]
end

function flip_turn(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::UInt32
    for i in 1:8
        trit = check_trit(s, i, bs, bbs)
        if trit == 1
            s += bbs[i]
        elseif trit == 2
            s -= bbs[i]
        end
    end

    remove = UInt32(0)
    add = UInt32(0)

    for i in 14:19
        if check_bit(s, i) && !check_bit(s, i + 6)
            remove += bs[i]
            add += bs[i + 6]
        end
    end
    for i in 20:25
        if check_bit(s, i) && !check_bit(s, i - 6)
            remove += bs[i]
            add += bs[i - 6]
        end
    end
    for i in 26:28
        if check_bit(s, i)
            remove += bs[i]
            add += bs[i + 3]
        end
    end
    for i in 29:31
        if check_bit(s, i)
            remove += bs[i]
            add += bs[i - 3]
        end
    end

    return s - remove + add + bs[32]
end

function place_piece(s::UInt32, to::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::UInt32
    if to == 0
        return s
    elseif to < 9
        if check_trit(s, to, bs, bbs) == 2
            s = move_in(s, bs) - 0x2 * bbs[to]
        end
        return s + bbs[to]
    else
        return s + bs[to]
    end
end

function take_piece(s::UInt32, from::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::UInt32
    if from == 0
        return move_out(s, bs)
    elseif from < 9
        return s - bbs[from]
    else
        return s - bs[from]
    end
end

function move_piece(s::UInt32, from::Int, to::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::UInt32
    s = place_piece(take_piece(s, from, bs, bbs), to, bs, bbs)
    return to in (4, 17, 19, 23, 25) ? s : flip_turn(s, bs, bbs)
end

function neighbours(s::UInt32, roll::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::Vector{UInt32}
    ns = zeros(UInt32, 7)
    neighbours!(ns, s, roll, bs, bbs)
    last = findfirst(iszero, ns)
    return last === nothing ? ns : ns[1:last - 1]
end

function neighbours!(ns::Vector{UInt32}, s::UInt32, roll::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::Nothing
    0 <= roll <= 4 || throw(ArgumentError("roll must be between 0 and 4"))
    fill!(ns, 0)

    if roll == 0
        ns[1] = flip_turn(s, bs, bbs)
        return nothing
    end

    idx = 0
    if has_home(s) && !check_bit(s, 13 + roll)
        idx += 1
        ns[idx] = move_piece(s, 0, 13 + roll, bs, bbs)
    end

    for i in 1:4
        if check_bit(s, 13 + i)
            if i + roll < 5
                if !check_bit(s, 13 + i + roll)
                    idx += 1
                    ns[idx] = move_piece(s, 13 + i, 13 + i + roll, bs, bbs)
                end
            else
                to = i + roll - 4
                if check_trit(s, to, bs, bbs) != 1 &&
                   !(check_trit(s, to, bs, bbs) == 2 && to == 4)
                    idx += 1
                    ns[idx] = move_piece(s, 13 + i, to, bs, bbs)
                end
            end
        end
    end

    for i in 1:8
        if check_trit(s, i, bs, bbs) == 1
            to = i + roll
            if to < 9
                if (to == 4 && check_trit(s, to, bs, bbs) == 0) ||
                   (to != 4 && check_trit(s, to, bs, bbs) != 1)
                    idx += 1
                    ns[idx] = move_piece(s, i, to, bs, bbs)
                end
            elseif to < 11
                end_safe = 18 + to - 9
                if !check_bit(s, end_safe)
                    idx += 1
                    ns[idx] = move_piece(s, i, end_safe, bs, bbs)
                end
            elseif to == 11
                idx += 1
                ns[idx] = move_piece(s, i, 0, bs, bbs)
            end
        end
    end

    for i in 1:2
        if check_bit(s, 17 + i)
            if i + roll == 3
                idx += 1
                ns[idx] = move_piece(s, 17 + i, 0, bs, bbs)
            elseif i + roll < 3 && !check_bit(s, 17 + i + roll)
                idx += 1
                ns[idx] = move_piece(s, 17 + i, 17 + i + roll, bs, bbs)
            end
        end
    end

    if idx == 0
        ns[1] = flip_turn(s, bs, bbs)
    end
    return nothing
end

function has_won(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Bool
    for i in 29:31
        check_bit(s, i) && return false
    end
    for i in 20:25
        check_bit(s, i) && return false
    end
    for i in 1:8
        check_trit(s, i, bs, bbs) == 2 && return false
    end
    return true
end

function pieces_on_board(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Tuple{Int, Int}
    self_count = 0
    other_count = 0
    for i in 14:19
        self_count += check_bit(s, i)
    end
    for i in 20:25
        other_count += check_bit(s, i)
    end
    for i in 1:8
        trit = check_trit(s, i, bs, bbs)
        self_count += trit == 1
        other_count += trit == 2
    end
    return self_count, other_count
end

function player_score(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32}; N::Int=7)::Tuple{Int, Int}
    self_home, other_home = how_many_home(s)
    self_board, other_board = pieces_on_board(s, bs, bbs)
    return N - self_home - self_board, N - other_home - other_board
end

function pieces_left(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Tuple{Int, Int}
    self_board, other_board = pieces_on_board(s, bs, bbs)
    self_home, other_home = how_many_home(s)
    self_count = self_board + self_home
    other_count = other_board + other_home
    return min(self_count, other_count), max(self_count, other_count)
end

function turn_change(s::UInt32, bs::Vector{UInt32})::Tuple{UInt32, Int}
    return check_bit(s, 32) ? (s - bs[32], -1) : (s, 1)
end

get_Ps()::Vector{Float64} = [binomial(4, k) for k in 0:4] .* (0.5 ^ 4)

function piece_locs(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32}; N::Int=7)::Tuple{Vector{Int}, Vector{Int}}
    self_pos = zeros(Int, N)
    other_pos = zeros(Int, N)
    self_counter = 1
    other_counter = 1

    for i in 14:17
        if check_bit(s, i)
            self_pos[self_counter] = i - 13
            self_counter += 1
        end
    end
    for i in 20:23
        if check_bit(s, i)
            other_pos[other_counter] = i - 19
            other_counter += 1
        end
    end
    for i in 1:8
        trit = check_trit(s, i, bs, bbs)
        if trit == 1
            self_pos[self_counter] = i + 4
            self_counter += 1
        elseif trit == 2
            other_pos[other_counter] = i + 4
            other_counter += 1
        end
    end
    for i in 18:19
        if check_bit(s, i)
            self_pos[self_counter] = i - 5
            self_counter += 1
        end
    end
    for i in 24:25
        if check_bit(s, i)
            other_pos[other_counter] = i - 11
            other_counter += 1
        end
    end
    return self_pos, other_pos
end

function board_loc_to_state_loc(loc::Int)::Int
    if loc < 5
        return loc + 13
    elseif loc > 12
        return loc + 5
    else
        return loc - 4
    end
end

function locs_to_s(
    self_locs::Vector{Int},
    other_locs::Vector{Int},
    self_home::Int,
    other_home::Int,
    bs::Vector{UInt32},
    bbs::Vector{UInt32},
)::UInt32
    s = UInt32(0)
    for _ in 1:other_home
        s = move_in(s, bs)
    end
    for loc in self_locs
        loc == 0 && break
        s = place_piece(s, board_loc_to_state_loc(loc), bs, bbs)
    end

    s = flip_turn(s, bs, bbs)
    for _ in 1:self_home
        s = move_in(s, bs)
    end
    for loc in other_locs
        loc == 0 && break
        s = place_piece(s, board_loc_to_state_loc(loc), bs, bbs)
    end
    return flip_turn(s, bs, bbs)
end

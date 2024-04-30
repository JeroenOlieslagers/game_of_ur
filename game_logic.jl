"""
    bs, bbs = get_bases()

Get the first 32 powers of 2 and 3.
"""
function get_bases()::Tuple{Vector{UInt32}, Vector{UInt32}}
    return UInt32.(2 .^ (0:31)), UInt32.(3 .^ (0:7))
end

"""
    start_state(bs, bbs; N=7)

Get starting state for N pieces.
"""
function start_state(bs::Vector{UInt32}, bbs::Vector{UInt32}; N=7)::UInt32
    s = UInt32(0)
    z = UInt32(0)
    o = UInt32(1)
    # Shared unsafe tiles
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
    for i in 26:28
        if (N ÷ 2^(28-i)) % 2 == 1
            s += bs[i]*o
        end
    end
    # black home
    for i in 29:31
        if (N ÷ 2^(31-i)) % 2 == 1
            s += bs[i]*o
        end
    end
    # turn (not needed for self-other representation)
    #s += bs[32]*z
    return s
end

"""
    check_bit(s, n)

Check bit in position n (0 or 1).
"""
function check_bit(s::UInt32, n::Int)::Bool
    return ((s >> (n-1)) & 1) == 1
end

"""
    check_trit(s, n, bs, bbs)

Return trit in position n (0, 1, or 2).
"""
function check_trit(s::UInt32, n::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::Int
    a = s % bs[14]
    return (a ÷ bbs[n]) % bbs[2]
end

"""
    how_many_home(s)

Return decimal of how many pieces are at home for self player.
"""
function how_many_home(s::UInt32)::Int
    return 4*check_bit(s, 26) + 2*check_bit(s, 27) + check_bit(s, 28)
end

"""
    has_home(s)

Return whether self player has any pieces left at home.
"""
function has_home(s::UInt32)::Bool
    return check_bit(s, 26) || check_bit(s, 27) || check_bit(s, 28)
end


"""
    move_out(s, bs)

Remove one piece from self's home.
"""
function move_out(s::UInt32, bs::Vector{UInt32})::UInt32
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

"""
    move_in(s, bs)

Add one piece to other's home.
"""
function move_in(s::UInt32, bs::Vector{UInt32})::UInt32
    # adds 1 to 3 bit encoded home (110 -> 111, 011 -> 100, etc)
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
    return s
end

"""
    flip_turn(s, bs, bbs)

Switches pieces from self and other.
"""
function flip_turn(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::UInt32
    # Shared tiles (turn 2 into 1 and 1 into 2 for each trit)
    for i in 1:8
        ct = check_trit(s, i, bs, bbs)
        if ct == 1
            s += bbs[i]
        elseif ct == 2
            s -= bbs[i]
        end
    end
    # store which bits will be swapped 
    # (remove is where to swap from, add is where to swap to)
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
    # Turn switch
    s += bs[32]
    return s
end

"""
    place_piece(s, to, bs, bbs)

Place piece on tile `to`.
"""
function place_piece(s::UInt32, to::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::UInt32
    if to == 0
        return s
    elseif to < 9
        # capture
        if check_trit(s, to, bs, bbs) == 2
            s = move_in(s, bs)
            s -= 0x2*bbs[to]
        end
        if check_trit(s, to, bs, bbs) == 1
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

"""
    take_piece(s, from, bs, bbs)

Remove piece from tile `from`.
"""
function take_piece(s::UInt32, from::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::UInt32
    if from == 0
        s = move_out(s, bs)
    elseif from < 9
        if check_trit(s, from, bs, bbs) != 1
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

"""
    move_piece(s, from, to, bs, bbs)

Move piece from `from` to `to`. Flip turn if `to` is not a rosette
"""
function move_piece(s::UInt32, from::Int, to::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::UInt32
    s = take_piece(s, from, bs, bbs)
    s = place_piece(s, to, bs, bbs)
    # get another roll (positions of rosette)
    if to == 4 || to == 17 || to == 23 || to == 19 || to == 25
        return s
    else
        return flip_turn(s, bs, bbs)
    end
end

"""
    neighbours(s, roll, bs, bbs)

Get list of possible neighbouring states from state `s` and dice roll `roll`.
"""
function neighbours(s::UInt32, roll::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::Vector{UInt32}
    if roll == 0
        return [flip_turn(s, bs, bbs)]
    end
    neighs = UInt32[]
    if roll < 1 || roll > 4
        throw(ErrorException("Wrong roll"))
    end
    # from home
    if has_home(s)
        if !check_bit(s, 13+roll)
            push!(neighs, move_piece(s, 0, 13+roll, bs, bbs))
        end
    end
    # from start safe
    for i in 1:4
        if check_bit(s, 13+i)
            # Check if goes to start safe
            if i+roll < 5
                if !check_bit(s, 13+i+roll)
                    push!(neighs, move_piece(s, 13+i, 13+i+roll, bs, bbs))
                end
            else # Else goes to unsafe
                if check_trit(s, i+roll-4, bs, bbs) != 1
                    # cant capture central rosette
                    if !(check_trit(s, i+roll-4, bs, bbs) == 2 && i+roll-4 == 4)
                        push!(neighs, move_piece(s, 13+i, i+roll-4, bs, bbs))
                    end
                end
            end
        end
    end
    # from unsafe
    for i in 1:8
        if check_trit(s, i, bs, bbs) == 1
            # Check if goes to unsafe
            if i+roll < 9
                # Central rosette
                if i+roll == 4
                    if check_trit(s, i+roll, bs, bbs) == 0
                        push!(neighs, move_piece(s, i, i+roll, bs, bbs))
                    end
                elseif check_trit(s, i+roll, bs, bbs) != 1
                    push!(neighs, move_piece(s, i, i+roll, bs, bbs))
                end
            elseif i+roll < 11 # Check if goes to end safe
                if !check_bit(s, 18+i+roll-9)
                    push!(neighs, move_piece(s, i, 18+i+roll-9, bs, bbs))
                end
            elseif i+roll == 11 # Check if completes path
                push!(neighs, move_piece(s, i, 0, bs, bbs))
            end
        end
    end
    # from end safe
    for i in 1:2
        if check_bit(s, 17+i)
            if i+roll == 3 # Check if completes path
                push!(neighs, move_piece(s, 17+i, 0, bs, bbs))
            elseif i+roll < 3 # Check if goes to end safe
                if !check_bit(s, 17+i+roll)
                    push!(neighs, move_piece(s, 17+i, 17+i+roll, bs, bbs))
                end
            end
        end
    end
    # No possible moves
    if isempty(neighs)
        return [flip_turn(s, bs, bbs)]
    else
        return neighs
    end
end

"""
    neighbours!(ns, s, roll, bs, bbs)

Get list of possible neighbouring states from state `s` and dice roll `roll`. Operation is in-place on existing list of
seven UInt32 `ns` which will be updated to remove memory allocations.
"""
function neighbours!(ns::Vector{UInt32}, s::UInt32, roll::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::Nothing
    # Maximum 7 actions in a state
    fill!(ns, 0)
    idx = 0
    if roll == 0
        idx += 1
        ns[idx] = flip_turn(s, bs, bbs)
        return nothing
    end
    #neighs = UInt32[]
    if roll < 1 || roll > 4
        throw(ErrorException("Wrong roll"))
    end
    # from home
    if has_home(s)
        if !check_bit(s, 13+roll)
            idx += 1
            ns[idx] = move_piece(s, 0, 13+roll, bs, bbs)
        end
    end
    # from start safe
    for i in 1:4
        if check_bit(s, 13+i)
            # Check if goes to start safe
            if i+roll < 5
                if !check_bit(s, 13+i+roll)
                    idx += 1
                    ns[idx] = move_piece(s, 13+i, 13+i+roll, bs, bbs)
                end
            else # Else goes to unsafe
                if check_trit(s, i+roll-4, bs, bbs) != 1
                    # cant capture central rosette
                    if !(check_trit(s, i+roll-4, bs, bbs) == 2 && i+roll-4 == 4)
                        idx += 1
                        ns[idx] = move_piece(s, 13+i, i+roll-4, bs, bbs)
                    end
                end
            end
        end
    end
    # from unsafe
    for i in 1:8
        if check_trit(s, i, bs, bbs) == 1
            # Check if goes to unsafe
            if i+roll < 9
                # Central rosette
                if i+roll == 4
                    if check_trit(s, i+roll, bs, bbs) == 0
                        idx += 1
                        ns[idx] = move_piece(s, i, i+roll, bs, bbs)
                    end
                elseif check_trit(s, i+roll, bs, bbs) != 1
                    idx += 1
                    ns[idx] = move_piece(s, i, i+roll, bs, bbs)
                end
            elseif i+roll < 11 # Check if goes to end safe
                if !check_bit(s, 18+i+roll-9)
                    idx += 1
                    ns[idx] = move_piece(s, i, 18+i+roll-9, bs, bbs)
                end
            elseif i+roll == 11 # Check if completes path
                idx += 1
                ns[idx] = move_piece(s, i, 0, bs, bbs)
            end
        end
    end
    # from end safe
    for i in 1:2
        if check_bit(s, 17+i)
            if i+roll == 3 # Check if completes path
                idx += 1
                ns[idx] = move_piece(s, 17+i, 0, bs, bbs)
            elseif i+roll < 3 # Check if goes to end safe
                if !check_bit(s, 17+i+roll)
                    idx += 1
                    ns[idx] = move_piece(s, 17+i, 17+i+roll, bs, bbs)
                end
            end
        end
    end
    # No possible moves
    if idx == 0
        idx += 1
        ns[idx] = flip_turn(s, bs, bbs)
    end
    return nothing
end

"""
    has_won(s, bs, bbs)

Return true if other has won, false otherwise.
"""
function has_won(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Bool
    # Check if pieces left at home
    for i in 29:31
        if check_bit(s, i)
            return false
        end
    end
    # Check if pieces on safe tiles
    for i in 20:25
        if check_bit(s, i)
            return false
        end
    end
    # Check if pieces on central column
    for i in 1:8
        if check_trit(s, i, bs, bbs) == 2
            return false
        end
    end
    # If no pieces, has won
    return true
end

"""
    to_move(s, from, to, bs, bbs)

Convert state representation move to board move.
Include info on whether capture was made or move is to rosette.
(from, to, is_capture, is_rosette)
"""
function to_move(s::UInt32, from::Int, to::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::Tuple{Int, Int, Bool, Bool}
    rosette = false
    capture = false
    # check if to has an existing piece
    if 0 < to < 9
        if check_trit(s, to, bs, bbs) > 0
            capture = true
        end
    else
        if check_bit(s, to)
            capture = true
        end
    end
    # check if to lands on rosette
    if to == 4 || to == 17 || to == 23 || to == 19 || to == 25
        rosette = true
    end
    # convert from state notation to board notation
    if from != 0
        if from < 9
            from += 4
        elseif from < 14
            throw(ErrorException("Wrong from"))
        elseif from < 18
            from -= 13
        elseif from < 20
            from -= 5
        else
            throw(ErrorException("Wrong from"))
        end
    end
    # convert from state notation to board notation
    if to != 0
        if to < 9
            to += 4
        elseif to < 14
            throw(ErrorException("Wrong to"))
        elseif to < 18
            to -= 13
        elseif to < 20
            to -= 5
        else
            throw(ErrorException("Wrong to"))
        end
    end
    return (from, to, capture, rosette)
end

"""
    neighbours_moves(s, roll, bs, bbs)

Get list of possible neighbouring states from state `s` and dice roll `roll` as well as associated moves.
"""
function neighbours_moves(s::UInt32, roll::Int, bs::Vector{UInt32}, bbs::Vector{UInt32})::Tuple{Vector{UInt32}, Vector{Tuple{UInt8, UInt8, Bool, Bool}}}
    if roll == 0
        return [flip_turn(s, bs, bbs)], [(UInt8(0), UInt8(0), false, false)]
    end
    neighs = UInt32[]
    moves = Tuple{UInt8, UInt8, Bool, Bool}[]
    if roll < 1 || roll > 4
        throw(ErrorException("Wrong roll"))
    end
    # from home
    if has_home(s)
        if !check_bit(s, 13+roll)
            push!(neighs, move_piece(s, 0, 13+roll, bs, bbs))
            push!(moves, to_move(s, 0, 13+roll, bs, bbs))
        end
    end
    # from start safe
    for i in 1:4
        if check_bit(s, 13+i)
            if i+roll < 5
                if !check_bit(s, 13+i+roll)
                    push!(neighs, move_piece(s, 13+i, 13+i+roll, bs, bbs))
                    push!(moves, to_move(s, 13+i, 13+i+roll, bs, bbs))
                end
            else
                if check_trit(s, i+roll-4, bs, bbs) != 1
                    # cant capture central rosette
                    if !(check_trit(s, i+roll-4, bs, bbs) == 2 && i+roll-4 == 4)
                        push!(neighs, move_piece(s, 13+i, i+roll-4, bs, bbs))
                        push!(moves, to_move(s, 13+i, i+roll-4, bs, bbs))
                    end
                end
            end
        end
    end
    # # from unsafe
    for i in 1:8
        if check_trit(s, i, bs, bbs) == 1
            if i+roll < 9
                # central safe tile
                if i+roll == 4
                    if check_trit(s, i+roll, bs, bbs) == 0
                        push!(neighs, move_piece(s, i, i+roll, bs, bbs))
                        push!(moves, to_move(s, i, i+roll, bs, bbs))
                    end
                elseif check_trit(s, i+roll, bs, bbs) != 1
                    push!(neighs, move_piece(s, i, i+roll, bs, bbs))
                    push!(moves, to_move(s, i, i+roll, bs, bbs))
                end
            elseif i+roll < 11
                if !check_bit(s, 18+i+roll-9)
                    push!(neighs, move_piece(s, i, 18+i+roll-9, bs, bbs))
                    push!(moves, to_move(s, i, 18+i+roll-9, bs, bbs))
                end
            elseif i+roll == 11
                push!(neighs, move_piece(s, i, 0, bs, bbs))
                push!(moves, to_move(s, i, 0, bs, bbs))
            end
        end
    end
    # from end safe
    for i in 1:2
        if check_bit(s, 17+i)
            if i+roll == 3
                push!(neighs, move_piece(s, 17+i, 0, bs, bbs))
                push!(moves, to_move(s, 17+i, 0, bs, bbs))
            elseif i+roll < 3
                if !check_bit(s, 17+i+roll)
                    push!(neighs, move_piece(s, 17+i, 17+i+roll, bs, bbs))
                    push!(moves, to_move(s, 17+i, 17+i+roll, bs, bbs))
                end
            end
        end
    end
    if isempty(neighs)
        return [flip_turn(s, bs, bbs)], [(UInt8(0), UInt8(0), false, false)]
    else
        return neighs, moves
    end
end

"""
    pieces_on_board(s, bs, bbs)

Get number of pieces on the board from each player (m, n), where ``m ≤ n``
"""
function pieces_on_board(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Tuple{Int, Int}
    self_count = 0
    other_count = 0
    # Count safe tiles
    for i in 14:19
        self_count += check_bit(s, i)
    end
    for i in 20:25
        other_count += check_bit(s, i)
    end
    # Count unsafe tiles
    for i in 1:8
        if check_trit(s, i, bs, bbs) == 1
            self_count += 1
        elseif check_trit(s, i, bs, bbs) == 2
            other_count += 1
        end
    end
    # Count pieces at home
    self_count += 4*check_bit(s, 26) + 2*check_bit(s, 27) + check_bit(s, 28)
    other_count += 4*check_bit(s, 29) + 2*check_bit(s, 30) + check_bit(s, 31)
    # Order
    if self_count < other_count
        return (self_count, other_count)
    else
        return (other_count, self_count)
    end
end

"""
    turn_change(s, bs)

Check if there has been a turn change and return new state and -1 if change, and 1 otherwise (s', ±1).
"""
function turn_change(s::UInt32, bs::Vector{UInt32})::Tuple{UInt32, Int}
    # Undo turn change encoded in state
    if check_bit(s, 32)
        s -= bs[32]
        return s, -1
    end
    return s, 1
end

"""
    get_Ps()

Get probabilities of dice rolls.
"""
function get_Ps()
    return [binomial(4, k) for k in 0:4]*(0.5^4) 
end

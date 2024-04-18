function start_state_int(bs, bbs)
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
    for i in 26:28#26:28
        s += bs[i]*o
    end
    # black home
    for i in 29:31#29:31
        s += bs[i]*o
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

function to_move(s, from, to, bs, bbs)
    rosette = false
    capture = false
    if 0 < to < 9
        if check_trit(s, to, bs, bbs) > 0
            capture = true
        end
    else
        if check_bit(s, to)
            capture = true
        end
    end
    if to == 4 || to == 17 || to == 23 || to == 19 || to == 25
        rosette = true
    end
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

function possible_neighs_moves(s, roll, bs, bbs)
    if roll == 0
        return [flip_turn(s, bs, bbs)], [(UInt8(0), UInt8(0), false, false)]
    end
    moves = Tuple{UInt8, UInt8, Bool, Bool}[]
    neighs = UInt32[]
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
                if check_trit(s, i+roll-4, bs, bbs) != 0x1
                    # cant capture central rosette
                    if !(check_trit(s, i+roll-4, bs, bbs) == 0x2 && i+roll-4 == 4)
                        push!(neighs, move_piece(s, 13+i, i+roll-4, bs, bbs))
                        push!(moves, to_move(s, 13+i, i+roll-4, bs, bbs))
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
                        push!(neighs, move_piece(s, i, i+roll, bs, bbs))
                        push!(moves, to_move(s, i, i+roll, bs, bbs))
                    end
                elseif check_trit(s, i+roll, bs, bbs) != 0x1
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
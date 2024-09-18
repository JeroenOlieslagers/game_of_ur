
function advancement(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32}; N::Int=7, score::Bool=false, def::Int=-1)::Tuple{Int, Int}
    self_advancement = 0
    other_advancement = 0
    # Also keep track of how many pieces have finished
    self_score = N
    other_score = N
    # Count safe tiles
    for i in 14:17
        if check_bit(s, i)
            self_advancement += (i-13)
            self_score -= 1
        end
    end
    for i in 18:19
        if check_bit(s, i)
            self_advancement += (i-5)
            self_score -= 1
        end
    end
    for i in 20:23
        if check_bit(s, i)
            other_advancement += (i-19)
            other_score -= 1
        end
    end
    for i in 24:25
        if check_bit(s, i)
            other_advancement += (i-11)
            other_score -= 1
        end
    end
    # Count unsafe tiles
    for i in 1:8
        if check_trit(s, i, bs, bbs) == 1
            self_advancement += i+4
            self_score -= 1
        elseif check_trit(s, i, bs, bbs) == 2
            other_advancement += i+4
            other_score -= 1
        end
    end
    # Subtract pieces at home from score
    self_home, other_home = how_many_home(s)
    self_score -= self_home
    other_score -= other_home
    # A scored piece is on the '15th' tile or it is given a set value
    # default to the sum of 14-N+1:14 (all pieces lining up) plus one
    if score
        if def == -1
            def = Int((14*13/2) - ((14-N)*(14-N-1)/2) + N + 1)
        end
        scored_worth = def
    else
        scored_worth = 15
    end
    self_advancement += self_score*scored_worth
    other_advancement += other_score*scored_worth
    return self_advancement, other_advancement
end

function h_advancement(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32}; N::Int=7)::Float64
    max_advancement = 15*N
    self_advancement, other_advancement = advancement(s, bs, bbs; N=N)

    self_remaining = max_advancement - self_advancement
    other_remaining = max_advancement - other_advancement
    if self_remaining > other_remaining
        return -100 * (1 - other_remaining / self_remaining)
    else
        return 100 * (1 - self_remaining / other_remaining)
    end
end


function h_0(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Float64
    return 0.0
end

function h_randn(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Float64
    return 0.001 * randn()
end

function h_ninf(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Float64
    return -Inf
end



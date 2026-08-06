function advancement(
    s::UInt32,
    bs::Vector{UInt32},
    bbs::Vector{UInt32};
    N::Int=7,
    score::Bool=false,
    def::Int=-1,
)::Tuple{Int, Int}
    self_advancement = 0
    other_advancement = 0
    self_score = N
    other_score = N

    for i in 14:17
        if check_bit(s, i)
            self_advancement += i - 13
            self_score -= 1
        end
    end
    for i in 18:19
        if check_bit(s, i)
            self_advancement += i - 5
            self_score -= 1
        end
    end
    for i in 20:23
        if check_bit(s, i)
            other_advancement += i - 19
            other_score -= 1
        end
    end
    for i in 24:25
        if check_bit(s, i)
            other_advancement += i - 11
            other_score -= 1
        end
    end

    for i in 1:8
        trit = check_trit(s, i, bs, bbs)
        if trit == 1
            self_advancement += i + 4
            self_score -= 1
        elseif trit == 2
            other_advancement += i + 4
            other_score -= 1
        end
    end

    self_home, other_home = how_many_home(s)
    self_score -= self_home
    other_score -= other_home

    scored_worth = if score
        def == -1 ? Int((14 * 13 / 2) - ((14 - N) * (14 - N - 1) / 2) + N + 1) : def
    else
        15
    end

    return self_advancement + self_score * scored_worth,
           other_advancement + other_score * scored_worth
end

function h_advancement(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32}; N::Int=7)::Float64
    max_advancement = 15 * N
    self_advancement, other_advancement = advancement(s, bs, bbs; N=N)
    self_remaining = max_advancement - self_advancement
    other_remaining = max_advancement - other_advancement

    if self_remaining > other_remaining
        return -100 * (1 - other_remaining / self_remaining)
    else
        return 100 * (1 - self_remaining / other_remaining)
    end
end

h_0(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Float64 = 0.0
h_rand(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Float64 = 200 * rand() - 100
h_randn(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Float64 = 0.001 * randn()
h_ninf(s::UInt32, bs::Vector{UInt32}, bbs::Vector{UInt32})::Float64 = -Inf

using ProgressBars
using StatsBase
using BenchmarkTools

function simulate_game(V1::Vector{Float64}, V2::Vector{Float64}, s_start::UInt32, state_to_ind::Dict{UInt32, Int32}, Ps::Vector{Float64}; max_iter::Int=2000)::Bool
    s = s_start
    light_turn = true
    ns = zeros(UInt32, 7)
    for iter in 1:max_iter
        roll = 4
        ran = rand()
        for i in 1:4
            if ran < Ps[i]
                roll = i - 1
                break
            end
        end
        if roll == 0
            s = flip_turn(s, bs, bbs) - bs[32]
            light_turn = ~light_turn
        else
            neighbours!(ns, s, roll, bs, bbs)
            max_V = -Inf
            @inbounds for i in 1:7
                neigh = ns[i]
                if neigh == 0
                    break
                end
                if light_turn
                    if check_bit(neigh, 32)
                        nd = neigh - bs[32]
                        n_ind = state_to_ind[nd]
                        Vn = -V1[n_ind]
                    else
                        Vn = V1[state_to_ind[neigh]]
                    end
                else
                    if check_bit(neigh, 32)
                        nd = neigh - bs[32]
                        n_ind = state_to_ind[nd]
                        Vn = -V2[n_ind]
                    else
                        Vn = V2[state_to_ind[neigh]]
                    end
                end
                if Vn > max_V
                    max_V = Vn
                    s = neigh
                end
            end
            if check_bit(s, 32)
                s -= bs[32]
                light_turn = ~light_turn
            end
            if has_won(s, bs, bbs)
                return ~light_turn
            end
        end
    end
    throw(ErrorException("Iteration limit reached"))
end

s_startt = start_state(bs, bbs; N=N)
Ps = cumsum(get_Ps())

@btime duel(1000, V2, V1, s_startt, state_to_ind)
@btime simulate_game(V2, V1, s_startt, state_to_ind, Ps)
@btime simulate_game(V2, V1, s_start, neigh_tensor, mirror_states, Ps)

#StatsBase.weights(w::Weights) = w
#function simulate_game(V1::Vector{Float64}, V2::Vector{Float64}, s_start::Int32, neigh_tensor::Matrix{Int32}, mirror_states::Vector{Int32}, Ps::Vector{Float64}; max_iter::Int=2000)::Bool
function simulate_game(V1::Vector{Float64}, V2::Vector{Float64}, s_start::Int32, neigh_tensor::Array{Int32, 3}, mirror_states::Vector{Int32}, Ps::Vector{Float64}; max_iter::Int=2000)::Bool
#function simulate_game(V1::AbstractArray, V2::AbstractArray, s_start::Int32, neigh_tensor::AbstractArray, mirror_states::AbstractArray, Ps::AbstractArray; max_iter::Int=1000)::Bool
    s = s_start
    _, _, max_ind = size(neigh_tensor)
    #Ps = Weights(get_Ps())
    light_turn = true
    for iter in 1:max_iter
        roll = 4
        ran = rand()
        for i in 1:4
            if ran < Ps[i]
                roll = i - 1
                break
            end
        end
        #roll = wsample(0:4, Ps)
        if roll == 0
            s = mirror_states[s]
            light_turn = ~light_turn
        else
            #neighs = @view neigh_tensor[:, roll, s]
            max_V = -Inf
            s_old = s
            #for neigh in neighs
            @inbounds for i in 1:7
                neigh = neigh_tensor[i, roll, s_old]
                if neigh == 0
                    break
                end
                if light_turn
                    if neigh < 0
                        Vn = -V1[-neigh]
                    else
                        Vn = V1[neigh]
                    end
                else
                    if neigh < 0
                        Vn = -V2[-neigh]
                    else
                        Vn = V2[neigh]
                    end
                end
                if Vn > max_V
                    max_V = Vn
                    s = neigh
                end
            end
            # if max_V == 0
            #     neighs_ = neighs[neighs .!= 0]
            #     if length(neighs_) > 1
            #         s = rand(neighs_)
            #     end
            # end
            if s < 0
                s = -s
                light_turn = ~light_turn
            end
            if s > max_ind
                return ~light_turn
            end
        end
    end
    throw(ErrorException("Iteration limit reached"))
end

for i in ProgressBar(1:137870097)
    for j in 1:4
        for k in 1:7
            neigh_tensor2[k + (j-1)*7, i] = neigh_tensor[k, j, i]
        end
    end
end

V1 = agents[:, end]
V2 = agents[:, 1]
V3 = agents[:, end]
V1 = initialize_value(h_0, ind_to_state, boundaries, bs, bbs);
V1 = initialize_value(h_randn, ind_to_state, boundaries, bs, bbs);
V2 += randn(length(V2));
V1 += randn(length(V1));

s_start = state_to_ind[start_state(bs, bbs; N=N)]

a = @view aagents[:, 1]
b = @view aagents[:, 2]

@btime simulate_game(V1, V1, s_start, neigh_tensor, mirror_states)
duel(100000, V, V2, s_start, neigh_tensor, mirror_states)
a = [simulate_game(V1, V3, s_start, neigh_tensor, mirror_states) for _ in 1:100000]

rresults2 = tournament(1000, aagents, s_start, neigh_tensor, mirror_states)

agentss = hcat(agents[:, 1:end-1], aagents[:, 2:end])
resultss = tournament(1000, agentss, s_start, neigh_tensor, mirror_states)

#function duel(n_games::Int64, V1::Vector{Float64}, V2::Vector{Float64}, s_start::UInt32, state_to_ind::Dict{UInt32, Int32})::Int64
function duel(n_games::Int64, V1::Vector{Float64}, V2::Vector{Float64}, s_start::Int32, neigh_tensor::Array{Int32, 3}, mirror_states::Vector{Int32})::Int64
#function duel(n_games::Int64, V1::AbstractArray, V2::AbstractArray, s_start::Int32, neigh_tensor::Array{Int32, 3}, mirror_states::Vector{Int32})
    wins = zeros(Bool, n_games)
    Ps = cumsum(get_Ps())
    Threads.@threads for i in 1:n_games
        #wins[i] = simulate_game(V1, V2, s_start, state_to_ind, Ps)
        wins[i] = simulate_game(V1, V2, s_start, neigh_tensor, mirror_states, Ps)
    end
    return sum(wins)
end

function tournament(n_games::Int64, agents::Matrix{Float64}, s_start::Int32, neigh_tensor::Array{Int32, 3}, mirror_states::Vector{Int32})
    N_ag = size(agents)[2]
    results = zeros(Int64, N_ag, N_ag)
    Threads.@threads for i in ProgressBar(1:N_ag)
        for j in 1:N_ag
            if i == j
                continue
            end
            V1 = @view agents[:, i]
            V2 = @view agents[:, j]
            wins = duel(n_games, V1, V2, s_start, neigh_tensor, mirror_states)
            results[j, i] = wins
        end
    end
    return results
end

function tournament_from_files(n_games::Int64, N_ag::Int64, s_start::Int32, neigh_tensor::Array{Int32, 3}, mirror_states::Vector{Int32})
    results = zeros(Int64, N_ag, N_ag)
    for i in 1:N_ag#Threads.@threads 
        V1 = load("jld2_files/agents/agents_$(i).jld2")["V"]
        for j in ProgressBar(i:N_ag)
            if i == j
                continue
            end
            V2 = load("jld2_files/agents/agents_$(j).jld2")["V"]
            wins = duel(n_games, V1, V2, s_start, neigh_tensor, mirror_states)
            results[j, i] = wins
            wins = duel(n_games, V2, V1, s_start, neigh_tensor, mirror_states)
            results[i, j] = wins
        end
    end
    return results
end

s_start = state_to_ind[start_state(bs, bbs; N=N)]
@time results = tournament_from_files(5000, 3, s_start, neigh_tensor, mirror_states)
results = load("jld2_files/results_AB.jld2")["results"]

V1 = load("jld2_files/agents/agents_split/agents_$(1).jld2")["V"]
V2 = load("jld2_files/agents/agents_split/agents_$(41).jld2")["V"]
@btime duel(1000, V2, V1, s_start, neigh_tensor, mirror_states)
@profview duel(1000, V2, V1, s_start, neigh_tensor, mirror_states)
@time simulate_game(V2, V1, s_start, neigh_tensor, mirror_states, Ps)
@btime simulate_game(V2, V2, s_start, neigh_tensor, mirror_states, Ps)
@profview simulate_game(V2, V2, s_start, neigh_tensor, mirror_states, Ps)
save("jld2_files/results.jld2", "results", results)

function tournament_results_to_series(results::Matrix{Int64}, n_games::Int64; base::Int64=0)
    N_ag = size(results)[1]
    N_n = N_ag * (N_ag-1) * n_games
    pairings = zeros(Int64, N_n, 2)
    outcomes = zeros(Bool, N_n)
    counter = 0
    for i in ProgressBar(1:N_ag)
        for j in 1:N_ag
            if i == j
                continue
            end
            win_counter = 0
            for _ in 1:n_games
                counter += 1
                win_counter += 1
                pairings[counter, 1] = i + base
                pairings[counter, 2] = j + base
                outcomes[counter] = win_counter <= results[j, i]
            end
        end
    end
    return pairings, outcomes
end

function write_pgn_file(pairings, outcomes)
    open("elo_games_91_agents.pgn", "w") do file
        for n in eachindex(outcomes)
            i, j = pairings[n, :]
            o = Int(outcomes[n])
            write(file, """[Event "RGU Tournament"]\n""")
            write(file, """[Site "Meyer 5"]\n""")
            write(file, """[Date "2024.09.18"]\n""")
            write(file, """[Round "$(n)"]\n""")
            write(file, """[White "$(i)"]\n""")
            write(file, """[Black "$(j)"]\n""")
            write(file, """[Result "$(o)-$(1-o)"]\n""")
            write(file, "\n$(o)-$(1-o)\n\n")
        end
    end
end

pairings, outcomes = tournament_results_to_series(results, 5000)

write_pgn_file(pairings, outcomes)

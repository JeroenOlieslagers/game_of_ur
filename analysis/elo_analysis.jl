function read_ratings(n)
    f = open("cleaned_code/analysis/BayesElo/elo_games_$(n)_agents.txt", "r")
    bounds = zeros(Int64, 2, n)
    elos = zeros(Int64, n)
    for (n, line) in enumerate(readlines(f))
        if n == 1
            continue
        end
     
        line_ = split(line, " ")
        line_ = line_[line_ .!= ""]
        rank, agent, elo, ub, lb, ngames, win_p, av_oppo, _ = line_
        agent = parse(Int64, agent)
        elos[agent] = parse(Int64, elo)
        bounds[:, agent] = [parse(Int64, lb), parse(Int64, ub)]
    end
    return elos .+ 1500, bounds
end


elos, bounds = read_ratings(91)
#elos_1, bounds_1 = read_ratings(52)
elos_1 = vcat(elos[1:40], elos[end])
bounds_1 = hcat(bounds[:, 1:40], bounds[:, end])
elos_2 = vcat(elos[1], elos[end-50:end])
bounds_2 = hcat(bounds[:, 1], bounds[:, end-50:end])

agent_t = load("jld2_files/agents/agents_split/agent_t.jld2")["agent_t"][1:41]
agent_counter = load("jld2_files/agents/agents_split/agent_counter.jld2")["agent_counter"][1:41]
agent_nm = load("jld2_files/agents/agents_split/agent_nm.jld2")["agent_nm"][1:41]
agent_counter .-= 1
agent_counter[1] = 0

aagent_t = load("jld2_files/agents/agent_t.jld2")["agent_t"]
aagent_counter = load("jld2_files/agents/agent_counter.jld2")["agent_counter"]
aagent_counter .-= 1
aagent_counter[1] = 0

plot(agent_t ./ 1000, elos_1, ribbon=bounds_1, grid=false, label=nothing, size=(400, 400), xlabel="Time (s)", ylabel="Elo")
scatter!(agent_t ./ 1000, elos_1, color=:red, markersize=2, markerstrokewidth=0, grid=false, label=nothing)
nms = get_piece_iterator(N)
ls = zeros(Int, length(nms))
lss = zeros(Int, length(nms))
for i in eachindex(nms)
    ind = findfirst(x->x==nms[i], agent_nm)
    ls[i] = agent_t[ind-1]
    lss[i] = agent_counter[ind-1]
end
vline!(ls ./ 1000, label=nothing, alpha=0.3, linestyle=:dash, color=:black)

plot(agent_t./1000, agent_counter, grid=false, label=nothing, size=(400, 400), ylabel="Iteration", xlabel="Time (s)")


plot(agent_counter, elos_1, ribbon=bounds_1, grid=false, label=nothing, size=(400, 400), xlabel="Iteration", ylabel="Elo")
scatter!(agent_counter, elos_1, color=:red, markersize=2, markerstrokewidth=0, grid=false, label=nothing)
vline!(lss, label=nothing, alpha=0.3, linestyle=:dash, color=:black)

plot(aagent_t ./ 1000, elos_2, ribbon=bounds_2, grid=false, label=nothing, size=(400, 400), xlabel="Time (s)", ylabel="Elo")
scatter!(aagent_t ./ 1000, elos_2, color=:red, markersize=2, markerstrokewidth=0, grid=false, label=nothing)

plot(aagent_counter, elos_2, ribbon=bounds_2, grid=false, label=nothing, size=(400, 400), xlabel="Iteration", ylabel="Elo")
scatter!(aagent_counter, elos_2, color=:red, markersize=2, markerstrokewidth=0, grid=false, label=nothing)

plot(aagent_t./1000, aagent_counter, grid=false, label=nothing, size=(400, 400), ylabel="Iteration", xlabel="Time (s)")
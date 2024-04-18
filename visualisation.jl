using Plots
using LaTeXStrings
using Statistics
using StatsBase
using StatsPlots

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

function draw_board(s, bs, bbs)
    cmap = [
        RGBA(([255, 255, 255]./255)...), 
        RGBA(([147, 190, 53]./255)...), 
        RGBA(([252, 80, 30]./255)...)
        ]
    pretty = pretty_print(s, bs, bbs)
    board = zeros(8, 3)
    board[1:4, 1] = reverse(pretty[3][1:4])
    board[1:4, 3] = reverse(pretty[3][5:8])*2
    board[1:8, 2] = pretty[3][9:16]
    board[7:8, 1] = reverse(pretty[3][17:18])
    board[7:8, 3] = reverse(pretty[3][19:20])*2
    if sum(board .== 2) == 0
        cmap = [
            RGBA(([255, 255, 255]./255)...), 
            RGBA(([147, 190, 53]./255)...), 
            ]
    end
    lh = sum(pretty[1] .* (2 .^(reverse(collect(0:2)))))
    ls = 7 - lh - sum(board .== 1)
    dh = sum(pretty[2] .* (2 .^(reverse(collect(0:2)))))
    ds = 7 - dh - sum(board .== 2)
    heatmap(board, c=cmap, size=(150, 300), colorbar=false, yflip=true, xticks=:none, yticks=:none, legend=false, dpi=300, fontfamily="helvetica")
    vline!(0.5:3.5, c=:black)
    hline!(0.5:4.5, c=:black)
    hline!(6.5:8.5, c=:black)
    annotate!(1, 4.5, text("↑", :green, :center, 12))
    annotate!(3, 4.5, text("↑", :red, :center, 12))
    annotate!(1.5, 1, text("→", :green, :center, 12))
    annotate!(2.5, 1, text("←", :red, :center, 12))
    annotate!(1.9, 1.5, text("↓", :green, :center, 12))
    annotate!(2.1, 1.5, text("↓", :red, :center, 12))
    annotate!(1.5, 8, text("←", :green, :center, 12))
    annotate!(2.5, 8, text("→", :red, :center, 12))
    annotate!(1, 6.5, text("↑", :green, :center, 12))
    annotate!(3, 6.5, text("↑", :red, :center, 12))
    annotate!(1, 1, text("X", :black, :center, 18))
    annotate!(3, 1, text("X", :black, :center, 18))
    annotate!(2, 4, text("X", :black, :center, 18))
    annotate!(1, 7, text("X", :black, :center, 18))
    annotate!(3, 7, text("X", :black, :center, 18))
    annotate!(1, 5, text(lh, :green, :center, 12))
    annotate!(3, 5, text(dh, :red, :center, 12))
    annotate!(1, 6, text(ls, :green, :center, 12))
    annotate!(3, 6, text(ds, :red, :center, 12))
    annotate!(2, 5.2, text("_____", :black, :center, 12))
end

function draw_boards(ss, bs, bbs; titles=nothing)
    titles_ = ["" for _ in eachindex(ss)]
    if titles !== nothing
        titles_ = titles
    end
    cmap = [
        RGBA(([255, 255, 255]./255)...), 
        RGBA(([147, 190, 53]./255)...), 
        RGBA(([252, 80, 30]./255)...)
        ]
    plot(layout=(4, Int(length(ss) / 4)), size=(25*length(ss), 800), dpi=300)
    for (n, s) in enumerate(ss)
        pretty = pretty_print(s, bs, bbs)
        board = zeros(8, 3)
        board[1:4, 1] = reverse(pretty[3][1:4])
        board[1:4, 3] = reverse(pretty[3][5:8])*2
        board[1:8, 2] = pretty[3][9:16]
        board[7:8, 1] = reverse(pretty[3][17:18])
        board[7:8, 3] = reverse(pretty[3][19:20])*2
        lh = sum(pretty[1] .* (2 .^(reverse(collect(0:2)))))
        ls = 7 - lh - sum(board .== 1)
        dh = sum(pretty[2] .* (2 .^(reverse(collect(0:2)))))
        ds = 7 - dh - sum(board .== 2)
        heatmap!(board, sp=n, title=string(titles_[n])*"%", c=cmap, colorbar=false, yflip=true, xticks=:none, yticks=:none, legend=false)
        vline!(0.5:3.5, sp=n, c=:black)
        hline!(0.5:4.5, sp=n, c=:black)
        hline!(6.5:8.5, sp=n, c=:black)
        annotate!(1, 4.5, sp=n, text("↑", :green, :center, 8))
        annotate!(3, 4.5, sp=n, text("↑", :red, :center, 8))
        annotate!(1.5, 1, sp=n, text("→", :green, :center, 8))
        annotate!(2.5, 1, sp=n, text("←", :red, :center, 8))
        annotate!(1.5, 8, sp=n, text("←", :green, :center, 8))
        annotate!(2.5, 8, sp=n, text("→", :red, :center, 8))
        annotate!(1, 6.5, sp=n, text("↑", :green, :center, 8))
        annotate!(3, 6.5, sp=n, text("↑", :red, :center, 8))
        annotate!(1, 1, sp=n, text("X", :black, :center, 14))
        annotate!(3, 1, sp=n, text("X", :black, :center, 14))
        annotate!(2, 4, sp=n, text("X", :black, :center, 14))
        annotate!(1, 7, sp=n, text("X", :black, :center, 14))
        annotate!(3, 7, sp=n, text("X", :black, :center, 14))
        annotate!(1, 5, sp=n, text(lh, :green, :center, 8))
        annotate!(3, 5, sp=n, text(dh, :red, :center, 8))
        annotate!(1, 6, sp=n, text(ls, :green, :center, 8))
        annotate!(3, 6, sp=n, text(ds, :red, :center, 8))
        annotate!(2, 5.2, sp=n, text("____", :black, :center, 8))
    end
    display(plot!())
end

function draw_board_heatmap(p)
    board = zeros(8, 3)
    board[5:6, 1] .= NaN
    board[5:6, 3] .= NaN
    board[1:4, 1] = reverse(p[1:4])
    board[1:4, 3] = reverse(p[1:4])
    board[1:8, 2] = p[5:12]
    board[7:8, 1] = reverse(p[13:14])
    board[7:8, 3] = reverse(p[13:14])
    heatmap(board, c=:thermal, size=(230, 400), colorbar=true, yflip=true, xticks=:none, yticks=:none, legend=false, colorbar_title="", dpi=300)#Occupancy (%)", dpi=300)
    vline!(0.5:3.5, c=:black)
    hline!(0.5:4.5, c=:black)
    hline!(6.5:8.5, c=:black)
end

function value_hist(V; s_start=nothing)
    bins = Float64.(collect(-100:2:100))
    bins[end] += 0.01
    histogram(push!(collect(values(V)), -collect(values(V))...), bins=bins, xlabel=latexstring("V"), ylabel="Frequency (x1M)", grid=false, label=nothing, dpi=300, size=(500,300),
        legendfont=font(10), 
        xtickfont=font(10), 
        ytickfont=font(10), 
        titlefont=font(10), 
        guidefont=font(14), yticks=([0, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000], ["0", "1", "2", "3", "4", "5", "6"]), fontfamily="helvetica")
    vline!([0], label=nothing, c=:gray, linestyle=:dash)
    if s_start !== nothing
        vline!([-V[s_start]], label="Start ("*latexstring("V")*"="*string(round(-V[s_start]*10)/10)*")", linewidth=2, foreground_color_legend=nothing, background_color_legend=nothing)
    end
    #vline!([mean(collect(values(V)))], label="Mean ("*latexstring("V")*"="*string(round(mean(collect(values(V)))*10)/10)*")", linewidth=2)
    #vline!([-13], label="Mode ("*latexstring("V")*"=-13)", linewidth=2)
    display(plot!())
end

function value_hist_empirical(vss; s_start=nothing)
    bins = Float64.(collect(-100:2:100))
    bins[end] += 0.01
    histogram(-reduce(vcat, vss), bins=bins, xlabel=latexstring("V"), ylabel="Frequency (x100k)", grid=false, label=nothing, dpi=300, size=(500,300),
        legendfont=font(10), 
        xtickfont=font(10), 
        ytickfont=font(10), 
        titlefont=font(10), 
        guidefont=font(14), yticks=([0, 2, 4, 6, 8, 10]*100000, ["0", "2", "4", "6", "8", "10"]), fontfamily="helvetica")
    vline!([0], label=nothing, c=:gray, linestyle=:dash)
    if s_start !== nothing
        vline!([V[s_start]], linewidth=2, label=nothing)
    end
    display(plot!())
end
#value_hist_empirical(vss; s_start=start_state_int(bs, bbs; N=7))

function value_hist_subset(V, sp; s_start=nothing)
    bins = Float64.(collect(-100:2:100))
    bins[end] += 0.01
    stephist!(push!(collect(values(V)), -collect(values(V))...), sp=sp, bins=bins, grid=false, label=nothing)
    vline!([0], sp=sp, label=nothing, c=:gray, linestyle=:dash)
    if s_start !== nothing
        vline!([V[s_start]], sp=sp, linewidth=2, label=nothing)#, label="Start ("*latexstring("V")*"="*string(round(V[s_start]*10)/10)*")", foreground_color_legend=nothing, background_color_legend=nothing)
    end
    #vline!([mean(collect(values(V)))], sp=sp, label=nothing, linewidth=2)
    display(plot!())
end

function subset_pieces(visited, leafs, N)
    println(N)
    subset = Dict{UInt32, Float64}()
    for k in keys(visited)
        if k[2] <= N
            for s in visited[k]
                subset[s] = V[s]
            end
        end
    end
    for k in keys(leafs)
        if k[2] <= N
            for s in leafs[k]
                subset[s] = V[s]
            end
        end
    end
    return subset
end

function trajectories_mean_plot(vss)
    maxx = maximum(length.(vss))
    a = zeros(Union{Missing, Float64}, maxx, 100000)
    for i in 1:100000
        g = vss[i]
        a[1:length(g), i] = g
        a[length(g)+1:end, i] .= missing
    end

    m = zeros(maxx)
    v = zeros(maxx)
    idx = 0
    for i in 1:maxx
        if length(collect(skipmissing(a[i, :]))) < 2
            break
        end
        m[i] = mean(skipmissing(a[i, :]))
        v[i] = std(skipmissing(a[i, :]))#/sqrt(length(collect(skipmissing(a[i, :]))))
        idx = i
    end
    m = m[1:250] 
    v = v[1:250]

    plot(m, ribbon=2*v, ylabel=latexstring("V"), xlabel="Move #", grid=false, label=nothing, dpi=300, size=(500,300),
    legendfont=font(10), 
    xtickfont=font(10), 
    ytickfont=font(10), 
    titlefont=font(10), 
    guidefont=font(14), fontfamily="helvetica")
    display(plot!())
end

function skill_plot(diffs)
    move_ns = Int.(collect(2:maximum(diffs[:, 2])))
    plot(layout=grid(3, 1, heights=[0.3, 0.3, 0.4]), xlabel="Available moves", grid=false, label=nothing, dpi=300, size=(300,500),
    legendfont=font(10), 
    xtickfont=font(10), 
    ytickfont=font(10), 
    titlefont=font(10), 
    guidefont=font(12), fontfamily="helvetica", link=:x, bottom_margin=-2Plots.mm)
    for move_n in move_ns
        idxs = diffs[:, 2] .== move_n .&& diffs[:, 1] .!= 0
        accuracy = 1 - sum(idxs) / sum(diffs[:, 2] .== move_n)
        bar!([move_n], [sum(diffs[:, 2] .== move_n)], sp=1, label=nothing, c=palette(:default)[move_n-1], xlabel="", ylabel="Count (x100k)", bar_width=0.5, yticks=([0, 2.5, 5]*100000, ["0", "2.5", "5.0"]), xticks=[])
        bar!([move_n], [accuracy], sp=2, label=nothing, c=palette(:default)[move_n-1], yticks=[0, 0.5], ylabel="Accuracy", ylim=(0, 0.55), bar_width=0.5)
        violin!([move_n], diffs[idxs, 1], sp=3, label=nothing, bar_width=0.5, c=palette(:default)[move_n-1], alpha=0.5, ylabel=latexstring("\\Delta V"))
        #dotplot!([move_n], diffs[idxs, 1], sp=3, label=nothing, bar_width=0.5, c=palette(:default)[move_n-1], markersize=4, alpha=0.5)
        #boxplot!([move_n], diffs[idxs, 1], sp=3, label=nothing, bar_width=0.5, c=palette(:default)[move_n-1], markersize=4)
    end
    plot!(move_ns, 1 ./ move_ns, sp=2, label=nothing, c=:red, markershape=:hline, l=nothing, linewidth=3, markersize=12, xticks=[], xlabel="")
    plot!([], [], sp=2, c=:red, linewidth=3, label="Chance", foreground_color_legend=nothing, background_color_legend=nothing)
    display(plot!())
end
skill_plot(huge_diffs)
huge_diffs = reduce(vcat, diffs)
huge_roll_vs = reduce(vcat, roll_vs)
huge_vss = reduce(vcat, vss)
huge_rolls = reduce(vcat, rolls)
visiteddd = [visitedd[i][1:end-1] for i in eachindex(visitedd)]
huge_visited = reduce(vcat, visiteddd)

function luck_plot(roll_vs, rolls)
    function break_ties_randomly(x, y)
        if x < y
            return true
        elseif x > y
            return false
        else
            # If elements are equal, break ties randomly
            return rand(Bool)
        end
    end
    diffs = (maximum(roll_vs, dims=2) .- [roll_vs[i, rolls[i]+1] for i in eachindex(rolls)])[:]
    move_ranks = mapslices(x -> ordinalrank(x, rev=true, lt=break_ties_randomly), roll_vs, dims=2)
    ranks = [move_ranks[i, rolls[i]+1] for i in eachindex(rolls)]
    plot(layout=grid(3, 1, heights=[0.3, 0.3, 0.4]), grid=false, dpi=300, size=(300,500),
    legendfont=font(10), 
    xtickfont=font(10), 
    ytickfont=font(10), 
    titlefont=font(10), 
    guidefont=font(12), fontfamily="helvetica", bottom_margin=0Plots.mm, link=:x)
    for rank in 1:5
        idxs = ranks .== rank
        bar!([rank], [sum(idxs)], sp=1, bar_width=0.5, label=nothing, ylabel="Count (x100k)", yticks=([0, 2, 4]*100000, ["0", "2", "4"]), xticks=[])
        violin!([rank], rolls[idxs], sp=2, ylabel="Roll", label=nothing, bandwidth=0.05, xticks=[])
        violin!([rank], diffs[idxs], sp=3, ylabel=latexstring("\\Delta V"), label=nothing, xlabel="Rank of roll")
    end
    display(plot!())
end
luck_plot(huge_roll_vs, huge_rolls)

trajectories_mean_plot(vss)
plot(m[1:300], ribbon=2*v[1:300], ylabel=latexstring("V"), xlabel="Move #", grid=false, label=nothing, dpi=300, size=(500,300),
    legendfont=font(10), 
    xtickfont=font(10), 
    ytickfont=font(10), 
    titlefont=font(10), 
    guidefont=font(14), fontfamily="helvetica")


value_hist(V; s_start=start_state_int(bs, bbs; N=7))
subsets = [subset_pieces(visited, leafs, n) for n in 1:6];

plot(size=(600, 600), layout=grid(3, 2),
    legendfont=font(10), 
    xtickfont=font(10), 
    ytickfont=font(10), 
    titlefont=font(10),
    guidefont=font(10), fontfamily="helvetica")
yticks = [
    ([0, 5, 10, 15], ["0", "0.5", "1.0", "1.5"]), 
    ([0, 300, 600], ["0", "3", "6"]), 
    ([0, 4000, 8000, 12000], ["0", "4", "8", "12"]), 
    ([0, 40000, 80000, 120000], ["0", "4", "8", "12"]), 
    ([0, 350000, 700000], ["0", "4", "8"]), 
    ([0, 1000000, 2000000, 3000000], ["0", "1", "2", "3"])]
ylims = [
    (0, 16),
    (0, 650),
    (0, 14000),
    (0, 140000),
    (0, 820000),
    (0, 3200000)
]
ylabels = [
    "Frequency (x10¹)",
    "Frequency (x10²)",
    "Frequency (x10³)",
    "Frequency (x10⁴)",
    "Frequency (x10⁵)",
    "Frequency (x10⁶)"
]
for i in 1:6
    value_hist_subset(subsets[i], i; s_start=start_state_int(bs, bbs; N=i))
    plot!(sp=i, ylim=ylims[i], ylabel=ylabels[i], link=:x , xlabel=latexstring("V")*" ("*string(i)*" pieces)", xticks=i in [5, 6] ? :auto : nothing, yticks=yticks[i], tick_direction=:out)
end
display(plot!())

plot(size=(1000, 1000), layout=grid(7, 7),
    legendfont=font(10), 
    xtickfont=font(10), 
    ytickfont=font(10), 
    titlefont=font(10),
    guidefont=font(10), fontfamily="helvetica", showaxis=false, grid=false, ticks=false)
for k in ProgressBar(keys(visited))
    subset = Dict{UInt32, Float64}()
    for s in visited[k]
        subset[s] = V[s]
    end
    value_hist_subset(subset, k[1] + (k[2]-1)*7)
    plot!(sp=k[1] + (k[2]-1)*7, title=string(k))
end
display(plot!())

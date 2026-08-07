using CairoMakie
using MakieExtra
using Statistics

const RESULTS_DIR = length(ARGS) >= 1 ? ARGS[1] : "ruleset_analysis/results"
const FIGURE_DIR = length(ARGS) >= 2 ? ARGS[2] : "paper/ICGA/appendix_figures"
const MODEL_DIR = length(ARGS) >= 3 ? ARGS[3] : "ruleset_analysis/models"
mkpath(FIGURE_DIR)

# This is the theme used by new_code/solver/plotting.jl for the manuscript figures.
best_theme = Theme(
    figure_padding=6,
    fontsize=9,
    size=(372, 250),
    px_per_unit=2,
    fonts = Attributes(
        :regular => Makie.texfont(:regular)
    ),
    Axis = (
        backgroundcolor = :transparent,
        xgridvisible = false,
        ygridvisible = false,
        leftspinevisible = true,
        rightspinevisible = false,
        bottomspinevisible = true,
        topspinevisible = false,
        xticksize = 2.0,
        yticksize = 2.0,
        xticklabelpad = 1,
        yticklabelpad = 1,
        xlabelpadding = 3,
        ylabelpadding = 3,
        xminorticksize = 1.0,
        yminorticksize = 1.0,
    ),
    Legend = (
        framevisible = false,
        padding = (0, 0, 0, 0)
    ),
    Colorbar = (
        ticksvisible = false,
        spinewidth = 0,
        ticklabelpad = 5,
    )
)
set_theme!(best_theme)

function load_gap_counts(file)
    data = Dict{Int, Int}()
    ties = 0
    open(file, "r") do io
        readline(io)
        for line in eachline(io)
            fields = split(line, ',')
            gap = parse(Float64, fields[7])
            if gap == 0
                ties += 1
            else
                exponent = floor(Int, log10(gap))
                data[exponent] = get(data, exponent, 0) + 1
            end
        end
    end
    return data, ties
end

function load_comparison(file)
    xs = Float64[]
    ys = Float64[]
    open(file, "r") do io
        readline(io)
        for line in eachline(io)
            fields = split(line, ',')
            push!(xs, parse(Float64, fields[3]))
            push!(ys, parse(Float64, fields[6]))
        end
    end
    return xs, ys
end

function load_random_results(file)
    epsilons = Float64[]
    ys = Float64[]
    open(file, "r") do io
        readline(io)
        for line in eachline(io)
            fields = split(line, ',')
            push!(epsilons, parse(Float64, fields[1]))
            push!(ys, parse(Float64, fields[4]))
        end
    end
    return epsilons, ys
end

function training_precision(file)
    open(file, "r") do io
        read(io, 4) == UInt8[0x52, 0x47, 0x55, 0x00] || error("Not an RGU map: $file")
        metadata_length = ntoh(read(io, UInt32))
        metadata = String(read(io, metadata_length))
        matched = match(r"\"training-precision\":([0-9.eE+\-]+)", metadata)
        matched === nothing && error("No training-precision in $file")
        return parse(Float64, matched.captures[1])
    end
end

function fig1(file, precision)
    data, ties = load_gap_counts(file)
    exponents = sort(collect(keys(data)))

    f = Figure(size=(372, 200));
    ax = Axis(f[1, 1], xlabel="Smallest difference in winning probability between two possible moves (%)", ylabel="Number of states", limits=((-15, 2), (1, nothing)), xticks=(collect(-14:2) .- 0.5, [rich("10", superscript("$(i)")) for i in -14:2]), ygridvisible=true, yscale=log10, xreversed=true);
    barplot!(exponents, [data[i] for i in exponents], color=:white, strokecolor=:black, strokewidth=1, width=1.25, fillto=1);
    vlines!([log10(precision) + 0.5], color=:red, linestyle=:dash, linewidth=1, label="Floating point precision");
    axislegend(ax, position=(0.75, 0.85), labelsize=10);
    return f, ties
end

function fig2(file)
    xs, ys = load_comparison(file)

    f = Figure(size=(372, 200));
    ax = Axis(f[1, 1], xlabel="Winning probability, value iteration (%)", ylabel="Winning probability, simulations (%)", limits=((0, 100), (0, 100)));
    scatter!(xs, ys, markersize=8, alpha=0.3, color=:black, strokecolor=:white, strokewidth=1);
    lines!([0, 100], [0, 100], color=:red, linestyle=:dash, linewidth=1, alpha=0.6);
    Label(f[1, 1, TopLeft()], "A", fontsize=14, font=:bold, halign=:left);
    text!(20, 90, text="r²=$(round(cor(xs, ys); digits=7))", fontsize=10);

    ax = Axis(f[1, 2], xlabel="Difference in winning probability (%)", ylabel="Number of states", limits=((nothing, nothing), (0, nothing)));
    hist!(xs .- ys, color=:transparent, strokecolor=:black, strokewidth=1, bins=10);
    Label(f[1, 2, TopLeft()], "B", fontsize=14, font=:bold, halign=:left);

    return f
end

function fig3(file, inset_y_limits)
    epsilons, ys = load_random_results(file)

    f = Figure(size=(372, 200));
    ax = Axis(f[1, 1], xlabel="Probability that opponent makes a random move (%)", ylabel="Winning probability, simulations (%)", limits=((0, 100), (50, 100)), xticks=0:10:100, xgridvisible=true, ygridvisible=true);
    sp = sortperm(epsilons)
    lines!([50, 50], [0, ys[sp][51]], color=:red, linestyle=:dash, linewidth=1);
    lines!([0, 50], [ys[sp][51], ys[sp][51]], color=:red, linestyle=:dash, linewidth=1);
    lines!(epsilons[sp] .* 100, ys[sp], color=:black, linewidth=1);

    inset = Axis(f[1, 1], width=Relative(0.35), height=Relative(0.5), halign=0.95, valign=0.4, limits=((80, 100), inset_y_limits), backgroundcolor=(:white, 1.0));
    translate!(inset.blockscene, 0, 0, 1000);
    lines!(100 .* epsilons[sp][end-20:end], ys[sp][end-20:end], color=:black, linewidth=1);

    zoom_lines!(ax, inset);

    return f
end

# The map is read only for its training-precision metadata, which annotates the
# gap histogram; the figure data itself comes from the CSVs.
# Candidate file names per rule set, tried in order. The map is opened only for
# its training-precision metadata, so either our own solve or the published map
# will do; whichever is present locally is used.
const MODEL_FILES = Dict(
    "blitz"   => ["blitz_f64.rgu"],
    "masters" => ["masters3d_f64.rgu"],
    "finkel"  => ["finkel_f64_ours.rgu", "finkel_f64.rgu"],
)

function find_model(ruleset)
    for name in MODEL_FILES[ruleset]
        path = joinpath(MODEL_DIR, name)
        isfile(path) && return path
    end
    error("no map found for $ruleset; tried " * join(MODEL_FILES[ruleset], ", "))
end
# The inset magnifies the high-epsilon corner, where the optimal agent's win
# rate approaches 100%; each rule set needs its own window.
const INSET_LIMITS = Dict(
    "blitz"   => (98, 100),
    "masters" => (99.8, 100),
    "finkel"  => (99.5, 100),
)

const RULESETS = length(ARGS) >= 4 ? split(ARGS[4], ",") : ["blitz", "finkel", "masters"]

for ruleset in RULESETS
    model_path = find_model(ruleset)
    inset_y_limits = INSET_LIMITS[ruleset]
    precision = training_precision(model_path)
    gap_figure, ties = fig1(joinpath(RESULTS_DIR, "$(ruleset)_gaps.csv"), precision)
    save(joinpath(FIGURE_DIR, "$(ruleset)_difference_hist.pdf"), gap_figure)
    save(joinpath(FIGURE_DIR, "$(ruleset)_simulation_compare.pdf"), fig2(joinpath(RESULTS_DIR, "$(ruleset)_compare.csv")))
    save(joinpath(FIGURE_DIR, "$(ruleset)_random_simulations.pdf"), fig3(joinpath(RESULTS_DIR, "$(ruleset)_epsilon.csv"), inset_y_limits))
    println("$(ruleset): omitted $(ties) zero-gap ties from logarithmic histogram")
end

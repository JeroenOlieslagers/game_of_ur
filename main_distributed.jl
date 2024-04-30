using Distributed

n_threads = Threads.nthreads()
if n_threads - nprocs() > 0
    addprocs(n_threads - nprocs())
end

@everywhere using Dates
@everywhere using SharedArrays
@everywhere using Random
@everywhere include("game_logic.jl")
@everywhere include("search.jl")
@everywhere include("value_iteration_threads.jl")


N = 3;
θ = 0.001;

bs, bbs = get_bases();
s_start = start_state(bs, bbs; N=N);

visited, leafs = bfs(s_start, bs, bbs);

p, V = value_iteration_smart(visited, leafs, θ, bs, bbs; N=N)



#V = get_value_dict(visited, leafs)
p, V = get_shared_array_pointers(visited, leafs)
ppp = get_reduced_pointer_sets(visited, p, bs, bbs)
Ps = get_Ps()

@btime value_iteration!(ppp[(1,1)], V, visited[(1,1)], θ, bs, bbs, Ps; max_iter=100)
subsets = random_split(visited[(3,3)], nprocs()-1);
@btime value_iteration_parallel!(subsets, ppp[(3,3)], V, visited[(3,3)], θ, bs, bbs, Ps; max_iter=100)

ns = zeros(UInt32, 7)
@btime a=get_new_value!(ns, s_start, bs, bbs, p, V, Ps)


V = get_value_dict(visited, leafs)
@btime value_iteration(V, visited[(1,1)], θ, bs, bbs, Ps)
subsets = random_split(visited[(3,3)], Threads.nthreads());
@btime value_iteration_parallel(subsets, V, visited[(3,3)], θ, bs, bbs, Ps)

@allocated value_iteration_smart(visited, leafs, θ, bs, bbs; N=N)

# for pieces in get_piece_iterator(N)
#     println(pieces)
#     no_leafs = visited[pieces]
#     value_iteration_parallel(p, V, no_leafs, θ, bs, bbs, Ps; max_iter=100)
#     #value_iteration_parallel(p, V, no_leafs, θ, bs, bbs, Ps; max_iter=100)
#     value_iteration(p, V, no_leafs, θ, bs, bbs, Ps; max_iter=100)
# end

#pieces_order = get_piece_iterator(N)


#value_iteration_parallel(pp, VV, visited[pieces_order[1]], θ, bs, bbs, Ps; max_iter=100)
#value_iteration_parallel(p, V, visited[pieces_order[1]], θ, bs, bbs, Ps; max_iter=100)
# begin
# value_iteration(p, V, visited[pieces_order[1]], θ, bs, bbs, Ps; max_iter=100)
# value_iteration(p, V, visited[pieces_order[2]], θ, bs, bbs, Ps; max_iter=100)
# value_iteration(p, V, visited[pieces_order[3]], θ, bs, bbs, Ps; max_iter=100)
# end


#V = value_iteration_smart(visited, leafs, θ, bs, bbs; N=N)
#VV = load("V_half_smart.jld2")["V"]
#visited = load("visited_half_smart.jld2")["visited"]
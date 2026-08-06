using Test

include(joinpath(@__DIR__, "..", "src", "GameOfUr.jl"))
import .GameOfUr as UR

# These expectations are pinned from the implementation this package was
# validated against (the earlier `solver/` tree, which it reproduced exactly on
# state helpers, BFS output, transition tensors and Bellman updates for N <= 3).
# They are recorded here so the suite stands alone rather than needing a second
# implementation to diff against.

@testset "dice distribution" begin
    Ps = UR.get_Ps()
    # Four binary dice: the number of 1s is Binomial(4, 1/2).
    @test Ps == [0.0625, 0.25, 0.375, 0.25, 0.0625]
    @test sum(Ps) ≈ 1.0
    @test length(Ps) == 5
end

@testset "start states" begin
    bs, bbs = UR.get_bases()
    expected = Dict(1 => 0x48000000, 2 => 0x24000000, 3 => 0x6c000000, 4 => 0x12000000)
    for N in 1:4
        s = UR.start_state(bs, bbs; N=N)
        @test s == expected[N]
        # Both players start with every piece at home and none on the board.
        @test UR.how_many_home(s) == (N, N)
        @test UR.pieces_left(s, bs, bbs) == (N, N)
        @test UR.pieces_on_board(s, bs, bbs) == (0, 0)
        @test !UR.has_won(s, bs, bbs)
    end
end

@testset "successors of the start state" begin
    bs, bbs = UR.get_bases()
    expected = Dict(
        1 => Dict(0 => [0xc8000000], 1 => [0x88080000], 2 => [0x88100000],
                  3 => [0x88200000], 4 => [0x40010000]),
        3 => Dict(0 => [0xec000000], 1 => [0xac080000], 2 => [0xac100000],
                  3 => [0xac200000], 4 => [0x64010000]),
    )
    for (N, by_roll) in expected
        s = UR.start_state(bs, bbs; N=N)
        for (roll, want) in by_roll
            got = sort(collect(UR.neighbours(s, roll, bs, bbs)))
            @test got == UInt32.(want)
        end
        # A roll of zero only passes the turn, so it has exactly one successor.
        @test length(UR.neighbours(s, 0, bs, bbs)) == 1
    end
end

@testset "state space enumeration" begin
    bs, bbs = UR.get_bases()
    expected_groups = Dict(
        2 => Dict((1, 1) => 217, (1, 2) => 2956, (2, 2) => 9696),
        3 => Dict((1, 1) => 217, (1, 2) => 2956, (1, 3) => 12628,
                  (2, 2) => 9696, (2, 3) => 79760, (3, 3) => 157864),
    )
    expected_leafs = Dict(2 => 121, 3 => 591)
    for (N, groups) in expected_groups
        s = UR.start_state(bs, bbs; N=N)
        visited, leafs = UR.bfs(s, bs, bbs)
        @test Set(keys(visited)) == Set(keys(groups))
        for (key, count) in groups
            @test length(visited[key]) == count
        end
        @test length(leafs) == expected_leafs[N]
        @test sum(length, values(visited)) == sum(values(groups))
        # Groups partition the state space: no state appears in two of them.
        all_states = union(values(visited)...)
        @test length(all_states) == sum(values(groups))
    end
end

@testset "value iteration converges" begin
    N = 3
    θ = 1e-9
    bs, bbs = UR.get_bases()
    s_start = UR.start_state(bs, bbs; N=N)
    visited, leafs = UR.bfs(s_start, bs, bbs)
    ind_to_state, state_to_ind, boundaries = UR.get_conversions(visited, leafs; N=N)
    states = setdiff(union(values(visited)...), leafs)
    neigh, mirror = UR.get_neigh_tensor(states, state_to_ind, bs, bbs)

    V = UR.initialize_value(UR.h_0, ind_to_state, boundaries, bs, bbs; N=N)
    UR.solve_game!(V, boundaries, neigh, mirror; θ=θ, n_epochs=200, n_iters=50)

    # Values are win percentages mapped onto [-100, 100].
    @test all(-100 - 1e-9 .<= V .<= 100 + 1e-9)

    # The residual of the Bellman operator must sit at or below the threshold on
    # the top score layer, which is solved last.
    range = boundaries[(N, N)][1]:boundaries[(N, N)][2]
    residual = UR.calculate_delta(copy(V), range, neigh, mirror, UR.get_Ps())
    @test residual <= θ

    # Going first is an advantage, but not an overwhelming one.
    win_percent = (V[state_to_ind[s_start]] + 100) / 2
    @test 50 < win_percent < 70
end

@testset "policy extraction" begin
    N = 2
    bs, bbs = UR.get_bases()
    s_start = UR.start_state(bs, bbs; N=N)
    visited, leafs = UR.bfs(s_start, bs, bbs)
    ind_to_state, state_to_ind, boundaries = UR.get_conversions(visited, leafs; N=N)
    states = setdiff(union(values(visited)...), leafs)
    neigh, mirror = UR.get_neigh_tensor(states, state_to_ind, bs, bbs)

    V = UR.initialize_value(UR.h_0, ind_to_state, boundaries, bs, bbs; N=N)
    agents = UR.solve_game!(V, boundaries, neigh, mirror; θ=1e-2, n_epochs=2, n_iters=5)
    policy = UR.value_to_policy(V, neigh)

    @test size(agents, 1) == length(V)
    @test size(policy) == (4, size(neigh, 3))
    @test -100 <= V[state_to_ind[s_start]] <= 100
end

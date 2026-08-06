function inferred_piece_count(visited::Dict{Tuple{Int64, Int64}, Set{UInt32}})::Int
    return maximum(last.(keys(visited)))
end

function get_conversions(
    visited::Dict{Tuple{Int64, Int64}, Set{UInt32}},
    leaf_nodes::Set{UInt32};
    N::Int=inferred_piece_count(visited),
)
    state_count = sum(length, values(visited))
    state_to_ind = Dict{UInt32, Int32}()
    ind_to_state = zeros(UInt32, state_count + length(leaf_nodes))
    boundaries = Dict{Tuple{Int64, Int64}, Tuple{Int32, Int32}}()

    counter = 0
    for pieces in get_piece_iterator(N)
        haskey(visited, pieces) || continue
        first_edge = counter + 1
        for s in visited[pieces]
            counter += 1
            ind_to_state[counter] = s
            state_to_ind[s] = Int32(counter)
        end
        boundaries[pieces] = (Int32(first_edge), Int32(counter))
    end

    for s in leaf_nodes
        counter += 1
        ind_to_state[counter] = s
        state_to_ind[s] = Int32(counter)
    end

    return ind_to_state, state_to_ind, boundaries
end

function get_neigh_tensor(
    states::Set{UInt32},
    state_to_ind::Dict{UInt32, Int32},
    bs::Vector{UInt32},
    bbs::Vector{UInt32};
    max_moves::Int=7,
)
    state_count = length(states)
    neigh_tensor = zeros(Int32, max_moves, 4, state_count)
    mirror_states = zeros(Int32, state_count)
    ns = zeros(UInt32, max_moves)

    for s in states
        n = state_to_ind[s]
        mirror_states[n] = state_to_ind[flip_turn(s, bs, bbs) - bs[32]]
        for roll in 1:4
            neighbours!(ns, s, roll, bs, bbs)
            for (i, neigh) in enumerate(ns)
                neigh == 0 && break
                if check_bit(neigh, 32)
                    neigh_tensor[i, roll, n] = -state_to_ind[neigh - bs[32]]
                else
                    neigh_tensor[i, roll, n] = state_to_ind[neigh]
                end
            end
        end
    end

    return neigh_tensor, mirror_states
end

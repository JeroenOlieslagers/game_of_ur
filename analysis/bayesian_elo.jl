using Turing

@model function bayesian_elo(pairings, outcomes, num_agents)    
    # Prior for skills with learned standard deviation
    #σ ~ truncated(Cauchy(0, 100), 0, Inf)  # Standard deviation of the log-normal distribution
    
    skills ~ filldist(truncated(Normal(1000, 100), 0, Inf), num_agents)
    

    # skills = Vector{Int}(undef, num_agents)
    # for i in 1:num_agents
    #     skills[i] ~ truncated(Normal(1000, 100), 0, Inf)
    #     #skills[i] ~ Normal(1000, σ)
    # end

    # Likelihood of match outcomes
    for n in eachindex(outcomes)
        i = pairings[n, 1]
        j = pairings[n, 2]
        prob = 1 / (1 + exp(-(skills[i] - skills[j])/400))
        outcomes[n] ~ Bernoulli(prob)
    end
end

function estimate_skills(pairings, outcomes, num_agents)
    # Run MCMC sampling to estimate posterior distribution
    model = bayesian_elo(pairings, outcomes, num_agents)
    chain = sample(model, NUTS(), MCMCThreads(), 1000, 8)
    #chain = sample(model, NUTS(), 1000)
    
    skills_posterior = [mean(chain[Symbol("skills[$(i)]")]) for i in 1:num_agents]
    #println(mean(chain[:σ]))

    return skills_posterior# ,σ_posterior
end

# Example usage:
# pairings = [(1, 2) for _ in 1:10000]
# outcomes = wsample([0, 1], [0.2, 0.8], 10000)
# num_agents = 2

num_agents = 11
skills_bayesian = estimate_skills(pairings, outcomes, num_agents)


# rA = 1000
# rB = 1000
# K = 1

# for i in outcomes
#     p = 1 / (1 + exp(-(rA - rB)/400))
#     rA = rA + (i - p)
#     rB = rB + (1-i - (1-p))
# end




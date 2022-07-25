using Distributions

function post_stim_lapse_prob(υ, τ, λ, σ, n_lapses)
    v1 = exp((υ.*(λ.^n_lapses))./σ)
    v2 = exp(τ./σ)
    return prod(v2./(v1 + v2 + 1))
end

function prob_attentive_resp(υ, τ, λ, σ, n_lapses)
    v1 = exp((υ.*(λ.^n_lapses))./σ)
    v2 = exp(τ./σ)
    return v1./(v1 + v2)
end

function prob_attend(υ, τ, λ, σ, n_lapses)
    v1 = exp((υ.*(λ.^n_lapses))./σ)
    v2 = exp(τ./σ)
    return v1/(1 + v1 + v2)
end

function prob_nonattentive_resp(υ, τ, λ, σ, n_lapses)
    v1 = exp((υ.*(λ.^n_lapses))./σ)
    v2 = exp(τ./σ)
    return 1/(1 + v1 + v2)
end

function post_stimulus_prob(υ, τ, λ, σ, n_cycles)
    #Non attentive respose
    nonatt_resp = 1.0
    for n_lapses in 0:(n_cycles-2)
        nonatt_resp *= post_stim_lapse_prob(υ, τ, λ, σ, n_lapses)
    end
    nonatt_resp *= prob_nonattentive_resp(υ, τ, λ, σ, n_cycles-1)
    #Attentive Response
    att_resp = 0.0
    for n_lapses = 0:(n_cycles-2)
        temp = 1.0
        for i in 0:(n_lapses-1)
            temp *= post_stim_lapse_prob(υ, τ, λ, σ, i)
        end
        temp *= prob_attend(υ, τ, λ, σ, n_lapses)
        for i in n_lapses:(n_cycles-3)
            temp *= post_stim_lapse_prob(υ, τ, λ, σ, i)
        end
        att_resp += temp
    end
    att_resp = att_resp.*prob_attentive_resp(υ, τ, λ, σ, n_cycles-2)
    return [nonatt_resp att_resp]
end

function show_cycles(n_cycles)
    output = []
    nonAtt =""
    for i in 0:(n_cycles-2)
        nonAtt = nonAtt*string("L(", i, ") ")
    end
    nonAtt = nonAtt*string("R(", n_cycles-1, ")")
    push!(output, nonAtt)
    for n_lapses in 0:(n_cycles-2)
        temp = ""
        for i in 0:(n_lapses-1)
            temp = temp*string("L(", i, ") ")
        end
        temp = temp*string("A(", n_lapses, ") ")
        for i in n_lapses:(n_cycles-3)
            temp = temp*string("L(", i, ") ")
        end
        temp = temp*string("R(", n_cycles-2, ")")
        push!(output, temp)
    end
    return output
end

function mixture_parms(max_cycles; γ, mt=.06, ρ=2/3)
    mu = fill(0.0, max_cycles, 2)
    sigma = similar(mu)
    cycles = [1:max_cycles;]
    @. mu[:,1] = cycles*γ + mt + .5*γ
    @. mu[:,2] = cycles*γ  + .085 + mt + .5*γ
    @. sigma[:,1] = 1.05*sqrt(
    (1/12)*(cycles*(γ*ρ).^2 + (mt*ρ)^2)) + .1*γ
    @. sigma[:,2] = 1.05*sqrt.(
    (1/12)*(cycles*(γ*ρ)^2  + (.085*ρ)^2
        + (mt*ρ)^2)) + .1*γ
    return mu, sigma
end

function compute_mixture_weights(υ, τ, λ, σ, max_cycles)
    probs = fill(0.0, max_cycles, 2)
    for n_cycles in 1:max_cycles
        probs[n_cycles,:] = post_stimulus_prob(υ, τ, λ, σ, n_cycles)
    end
    return probs
end

"""
Computes the log likelihood of a vector of rts for the PVT model. Note that
this does not handle false starts. The parameters are as follows:
υ: utility
τ: threshold
λ: FPdec
γ: conflict resolution time
rts: the data
σ: utility noise default
max_cycles: maximum number of component distributions,  each corresponding
    to a latent number of production cycles. 15 is sufficient for most parameterizations.
"""
LogLikelihood(rt::Float64; υ, τ, λ, γ, σ=.45345, max_cycles=15) = LogLikelihood([rt]; υ, τ, λ, γ, σ, max_cycles)

function LogLikelihood(RTs::Vector{Float64}; υ, τ, λ, γ, σ=.45345, max_cycles=15)
    LL = 0.0
    mus, sigmas = mixture_parms(max_cycles; γ)
    weights = compute_mixture_weights(υ, τ, λ, σ, max_cycles)
    for rt in RTs
        L = likelihood_trial(rt, mus, sigmas, weights, max_cycles)
        LL += log(L)
    end
    return LL
end

function likelihood_trial(rt, mus, sigmas, weights, max_cycles)
    L = 0.0
    for att in 1:2
        for n_cycles in 1:max_cycles
            L += pdf(Normal(mus[n_cycles, att], sigmas[n_cycles, att]), rt)*weights[n_cycles, att]
        end
    end
    return L
end



function simulate(υ, τ, λ, γ, trials)

    egs = .45345 ##utility noise free [.45345]
    n = 3 ##production cycle noise fixed [3]
    # n = x[5]

    a_time = .085 ##time to attend fixed [.085]
    r_time = .06 ##time to respond fixed [.06]

    p_util = ones(1, 3) ##set initial utility to 1.0
    pm = [0 0 1; 0 0 1; 0 0 0] ##mismatch penalty [free]

    ##Upper bound and lower bound times for three types of events:
    ##production cycle attend respond: See ACT-R manual
    t_c = zeros(n, n-1);
    t_c[:, 1] = [γ; a_time; r_time];
    t_c[:, 2] = t_c[:, 1].*[n-1]./n;
    t_c[:, 1] = 2.0*t_c[:, 1]./n;
    durations = round.(8.0*rand(trials, 1).+2) ##duration of delay [uniform over 8-10 seconds]
    performance = zeros(trials, 2) ##300 trials

    for i in 1:trials ##trials [2500]
        ub = durations[i] + 10 ##point at which trial terminates [10 seconds]
        state = 1 ##[1 = wait 2 = attend 3 = respond]
        cnt = 0.0 ## model run time
        fp_percent = 1.0 ##micro-lapse attenuation [starts at 1]
        while (state == 1) && (cnt < ub) ##waiting for stimulus
            if cnt < durations[i] ##signal absent
                utility = fp_percent * υ * [1.0,  0.0] .+ rand(Logistic(0.0, egs), 2)#s_noise[pd, [1; 3]]
                cnt = cnt + t_c[1, 1] + t_c[1, 2].*rand()#.*t_noise[pd, 1] ##duration of production cycle
                ##conflict resolution
                if (utility[1]  < τ) && (utility[2]  < τ) ##no production exceeds threshold
                    fp_percent = fp_percent * λ ##decrease G
                elseif utility[2] > utility[1] ##false start
                    state = 4 ##respond
                    cnt = cnt + t_c[3, 1] + t_c[3, 2] * rand()#*t_noise[pd, 3]
                end
            else ##signal present
                state = 2 ##prepare to encode
            end
        end
        fp_percent = 1.0
        while (state == 2) && (cnt < ub) ##signal present
            utility = fp_percent * υ * [1.0,  0.0] .+ rand(Logistic(0.0, egs), 2)#s_noise[pd, [2; 3]]
            cnt = cnt + t_c[1, 1] + t_c[1, 2] * rand()#*t_noise[pd, 1] ##duration of production cycle
            max_util = findmax(utility) ##conflict resolution
            if max_util[1] < τ ##no production exceeds threshold
                fp_percent = fp_percent * λ ##decrease G
            elseif max_util[2] == 1 ##production exceeds threshold
                state = 3 ##prepare to respond
                cnt = cnt + t_c[2, 1] + t_c[2, 2] * rand()#*t_noise[pd, 2] ##time to attend
            elseif max_util[2] == 2
                state = 4 ##respond
                cnt = cnt + t_c[3, 1] + t_c[3, 2] * rand()#*t_noise[pd, 3]
            end
        end

        while (state == 3) && (cnt < ub) ##encoding complete
            utility = fp_percent*υ .+ rand(Logistic(0.0, egs), 2)#s_noise[pd, 3]
            cnt = cnt + t_c[1, 1] + t_c[1, 2].*rand()#*t_noise[pd, 1] ##duration of production cycle
            #max_util = findmax(utility) ##conflict resolution
            if utility[1] .< τ ##no production exceeds threshold
                fp_percent = fp_percent * λ ##decrease G
            else ##production exceeds threshold
                state = 4 ##respond
                cnt = cnt + t_c[3, 1] + t_c[3, 2] * rand()#*t_noise[pd, 3]
            end
        end
        performance[i, :] = [durations[i] cnt]
    end
    rts = (performance[:, 2] - performance[:, 1]) ##RT - ISI
    isi = performance[:, 1]
    return rts, isi
end

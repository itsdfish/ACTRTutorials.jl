using Parameters, KernelDensity, Distributions, Interpolations
import Distributions: logpdf
import KernelDensity: kernel_dist

kernel_dist(::Type{Epanechnikov}, w::Float64) = Epanechnikov(0.0, w)
kernel(data) = kde(data; kernel=Epanechnikov)

function simulate(parms; blc, τ)
    # Create chunk
    chunks = [Chunk()]
    # add chunk to declarative memory
    memory = Declarative(;memory=chunks)
    # create ACTR object and pass parameters
    actr = ACTR(;declarative=memory, parms..., blc, τ)
    # retrieve chunk
    chunk = retrieve(actr)
    # 2 if empty, 1 otherwise
    resp = isempty(chunk) ? resp = 2 : 1
    # compute reaction time 
    rt = compute_RT(actr, chunk) + actr.parms.ter
    return (resp = resp,rt = rt)
end

function loglike(data, blc, τ; parms, n_sim=2_000)
    # generate simulated data
    sim_data = map(_->simulate(parms; blc, τ), 1:n_sim)
    # get correct rts
    correct_rts = get_rts(sim_data, 1)
    # get incorrect rts
    incorrect_rts = get_rts(sim_data, 2)
    # this will return loglike = -Inf if empty
    incorrect_rts = isempty(incorrect_rts) ? [-100.0] : incorrect_rts
    # probability correct
    p_correct = length(correct_rts)/n_sim
    # kernel density for correct rts
    kd_correct = kernel(correct_rts)
    # kernel density distribution object for correct
    dist_correct = InterpKDE(kd_correct)
    # kernel density for incorrect rts
    kd_incorrect = kernel(incorrect_rts)
    # kernel density distribution object for correct
    dist_incorrect = InterpKDE(kd_incorrect)
    LL = 0.0
    # compute log likelihood for each trial
    for d in data
        if d.resp == 1
            LL += logpdf(dist_correct, d.rt) + log(p_correct)
        else
            LL += logpdf(dist_incorrect, d.rt) + log(1 - p_correct)
        end
    end
    return LL
end

logpdf(dist::InterpKDE, y) = log(abs(pdf(dist, y))) 

get_rts(data, resp) = map(x->x.rt, filter(x->x.resp==resp, data))
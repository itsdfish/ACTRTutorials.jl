using Parameters
import Distributions: logpdf, rand, loglikelihood
import VisualSearchACTR:
    initialize_model, compute_angular_size!, update_finst!, attend_object!
import VisualSearchACTR: update_threshold!, get_iconic_memory, relevant_object

struct VisualSearch{T1, T2} <: ContinuousUnivariateDistribution
    topdown_weight::T1
    stimuli::T2
end

VisualSearch(; topdown_weight, stimuli) = VisualSearch(topdown_weight, stimuli)

loglikelihood(d::VisualSearch, data::Vector{Vector{Fixation}}) = logpdf(d, data)

function logpdf(d::VisualSearch, data::Vector{Vector{Fixation}})
    LL = computeLL(d.stimuli, data; topdown_weight = d.topdown_weight)
    return LL
end

function simulate(experiment; parms...)
    # generate stimuli, consisting of visual array and target for each trial
    stimuli = map(_ -> generate_stimuli(experiment), 1:(experiment.n_trials))
    # generate data for each trial
    run_condition!(experiment, stimuli; parms...)
    # return stimuli and fixation data
    return stimuli, experiment.fixations
end

function reset!(visual_objects)
    for vo in visual_objects
        vo.attended = false
        vo.visible = false
        vo.attend_time = 0.0
    end
    return nothing
end

function loglikelihood_trial(ex, target, visicon, fixations; parms...)
    # reset attend time, visibility etc. 
    reset!(visicon)
    # create a model based on target, visual objects and parameters
    actr = initialize_model(ex, target, visicon; noise = false, parms...)
    # compute the angular size of objects in visicon
    compute_angular_size!(actr, ex.ppi)
    # initialize visual attention in center of screen
    orient!(actr, ex)
    LL = 0.0
    for fixation in fixations
        # increment model time 
        actr.time += 0.05
        # compute log likelihood of fixation 
        LL += loglikelihood_fixation(ex, actr, visicon, fixation)
        # if the model terminates search, break. Otherwise update attention and threshold
        fixation.stop ? (break) : nothing
        vo = visicon[fixation.idx]
        actr.time = fixation.attend_time
        # update model fixation to new visual object
        attend_object!(actr, ex, vo)
        # update termination threshold
        update_threshold!(actr)
    end
    return LL
end

function loglikelihood_fixation(ex, actr, visicon, fixation)
    # before computing fixation probability, update decay in iconic memory, finst, visibliity and activaiton values
    update_decay!(actr)
    update_finst!(actr)
    update_visibility!(actr, ex.ppi)
    compute_activations!(actr)
    # get all visible objects, which factor in the the fixation probability
    visible_objects = filter(x -> relevant_object(actr, x), get_iconic_memory(actr))
    # get activation values
    act = map(x -> x.activation, visible_objects)
    # add termination threshold to activation values
    push!(act, actr.parms.τₐ)
    # compute the probability of the fixation
    p = fixation_prob(actr, visicon, visible_objects, fixation)
    return log(p)
end

function computeLL(stimuli, all_fixations; topdown_weight, parms...)
    ex = Experiment()
    LL = 0.0
    # copy and reset fields in visual array
    _stimuli = set_stimuli(stimuli, topdown_weight)
    for i = 1:length(all_fixations)
        LL += loglikelihood_trial(
            ex,
            _stimuli[i][1],
            _stimuli[i][3],
            all_fixations[i];
            topdown_weight,
            parms...
        )
    end
    return LL
end

function set_stimuli(stimuli, parm)
    return [copy_data(s, parm) for s in stimuli]
end

function copy_vo(vo, parm)
    VisualObject(; features = vo.features, location = vo.location,
        activation = zero(parm))
end

function copy_data(s, parm)
    target = s[1]
    present = s[2]
    vos = [copy_vo(vo, parm) for vo in s[3]]
    return (target, present, vos)
end

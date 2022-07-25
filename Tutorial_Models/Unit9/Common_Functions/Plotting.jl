function get_title()
  title = [string("place: ",i," ","person: ",j) for i in 1:3 for j in 1:3]
  return reshape(title, 1, 9)
end

function plot_chains(chain; options...)
    p1 = plot(chain, xaxis=font(5), yaxis=font(5), seriestype=(:traceplot),
      grid=false, size=(275,125), titlefont=font(5); options...)
    p2 = plot(chain, xaxis=font(5), yaxis=font(5), seriestype=(:autocorplot),
      grid=false, size=(275,125), titlefont=font(5); options...)
    p3 = plot(chain, xaxis=font(5), yaxis=font(5), seriestype=(:mixeddensity),
      grid=false, size=(275,125), titlefont=font(5); options...)
    p = plot(p1, p2, p3, layout=(3,1), size=(300,300); options...)
    return p
end

function plot_fan(preds, trial, resp; options...)
  fanEffect = filter(x->(x.trial == trial) && (x.resp == resp), preds)
  title = get_title()
  p = @df fanEffect histogram(:MeanRT, group=(:fanPlace,:fanPerson), ylabel="Density",
      xaxis=font(6), yaxis=font(6), grid=false, norm=true, color=:grey, leg=false, size=(350,300),
      titlefont=font(6), title=title, layout=9, xlims=(1,3), ylims=(0,7), bins=15; options...)
  return p
end

plot_fan!(p1, preds, trial::String, resp; options...) = plot_fan!(p1, preds, Symbol(trial), Symbol(resp); options...)

function plot_fan!(p1, preds, trial, resp; options...)
  fanEffect = filter(x->(x.trial == trial) && (x.resp == resp), preds)
  isempty(fanEffect) ? (return nothing) : nothing
  add_missing_factors!(fanEffect)
  sort!(fanEffect, [:fanPlace,:fanPerson])
  vline!(p1, fanEffect[:,:MeanRT]', color=:darkred)
  return nothing
end

function scatter_fan(preds, parm, trial, resp; options...)
  fanEffect = filter(x->(x.trial == trial) && (x.resp == resp), preds)
  title = get_title()
  p = @df fanEffect scatter(cols(parm), :MeanRT, group=(:fanPlace,:fanPerson), xlabel=string(parm),
    ylabel="Mean RT", xaxis=font(6), yaxis=font(6), grid=false, color=:grey, leg=false, size=(800,600),
      titlefont=font(6), title=title, layout=9; options...)
  return p
end

function scatter_accuracy(preds, parm, trial; options...)
  fanEffect = filter(x->x.trial == trial, preds)
  title = get_title()
  p = @df fanEffect scatter(cols(parm), :accuracy, group=(:fanPlace,:fanPerson), ylabel="Accuracy",
       xlabel=string(parm), xaxis=font(6), yaxis=font(6), grid=false, color=:grey, leg=false,
       size=(800,600), titlefont=font(6), layout=9, ylims=(.5, 1), title=title; options...)
  return p
end

"""
add missing factors for accurate plotting. If a factor is missing, it is added
and the RT is set to -Inf so that it is not plotted.
"""
function add_missing_factors!(df)
  levels = 1:3
  for i in levels, j in levels
      temp = filter(x->(x.fanPlace==i) && (x.fanPerson==j), df)
      if isempty(temp)
        new_row = DataFrame(df[1,:])
        new_row[1,:fanPlace] = i
        new_row[1,:fanPerson] = j
        new_row[1,:MeanRT] = -Inf
        append!(df, new_row)
      end
  end
end

function plot_accuracy(preds, trial; options...)
  fanEffect = filter(x->x.trial == trial, preds)
  title = get_title()
  p = @df fanEffect histogram(:accuracy, group=(:fanPlace,:fanPerson), ylabel="Density",
      xaxis=font(6), yaxis=font(6), grid=false, norm=true, color=:grey, leg=false, size=(350,300),
      titlefont=font(6), layout=9, xlims=(.5, 1), ylims=(0,8), bins=10, title=title; options...)
  return p
end

function plot_accuracy!(p1, preds, trial; options...)
  fanEffect = filter(x->x.trial == trial, preds)
  sort!(fanEffect, [:fanPlace,:fanPerson])
  vline!(p1, fanEffect[:,:accuracy]', color=:darkred)
end

fan_contrast(data) = fan_contrast(DataFrame(data))

function fan_contrast(df::DataFrame)
  weights = df[:,:fanPerson] .+ df[:, :fanPlace] .- 4
  return df[:,:mean_zrt]'*weights
end

"""
Plot and save all results for parameter estimates using empirical data.
"""
function plot_all_post_preds(chain, df_data, parms, base_path, subj_idx; options...)
  println("plotting subject ", subj_idx)
  #######################################################################################
  #                                  Posterior Predictive RTs
  #######################################################################################
  path = string(base_path, "/subject_", subj_idx, "/")
  isdir(path) ? nothing : mkpath(path)
  slots = get_memory_slots(df_data, :person, :place)
  data = Tables.rowtable(df_data)
  stimuli = [(trial=d.trial,person=d.person,place=d.place) for d in data]
  sim(p) = simulate(stimuli, slots, parms, 1; p...)
  preds_rt = posteriorPredictive(x->sim(x), chain, 1000, summarize)
  preds_rt = vcat(preds_rt...)
  df_subj_rt = summarize(df_data)
  # hits
  fan_rt_hits = plot_fan(preds_rt, :target, :yes; xlims=(0,2.5), options...)
  plot_fan!(fan_rt_hits, df_subj_rt, :target, :yes)
  file_name = string(path, "rt_hits.eps")
  savefig(fan_rt_hits, file_name)
  # misses
  fan_rt_misses = plot_fan(preds_rt, :target, :no; xlims=(0,2.5), options...)
  plot_fan!(fan_rt_misses, df_subj_rt, :target, :no)
  file_name = string(path, "rt_misses.eps")
  savefig(fan_rt_misses, file_name)
  # correct rejections
  fan_rt_cr = plot_fan(preds_rt, :foil, :no; xlims=(0,2.5), options...)
  plot_fan!(fan_rt_cr, df_subj_rt, :foil, :no)
  file_name = string(path, "rt_cr.eps")
  savefig(fan_rt_cr, file_name)

  # false alarms
  fan_rt_fa = plot_fan(preds_rt, :foil, :no; xlims=(0,2.5), options...)
  plot_fan!(fan_rt_fa, df_subj_rt, :foil, :yes)
  file_name = string(path, "rt_false_alarms.eps")
  savefig(fan_rt_fa, file_name)
  #######################################################################################
  #                                  Posterior Predictive Accuracies
  #######################################################################################
  stimuli = [(trial=d.trial,person=d.person,place=d.place) for d in data]
  sim(p) = simulate(stimuli, slots, parms, 1; p...)
  preds_accuracy = posteriorPredictive(x->sim(x), chain, 1000, accuracy)
  preds_accuracy = vcat(preds_accuracy...)
  df_subj_accuracy = accuracy(df_data)
  # target trials
  accuracy_target = plot_accuracy(preds_accuracy, :target)
  plot_accuracy!(accuracy_target, df_subj_accuracy, :target)
  file_name = string(path, "accuracy_target.eps")
  savefig(accuracy_target, file_name)
  # foil trials
  accuracy_foil = plot_accuracy(preds_accuracy, :foil)
  plot_accuracy!(accuracy_foil, df_subj_accuracy, :foil)
  file_name = string(path, "accuracy_foil.eps")
  savefig(accuracy_foil, file_name)
end

using Parameters, Distributions, StatsBase
import Distributions: logpdf, rand, loglikelihood

struct Markov{T1, T2, T3} <: ContinuousUnivariateDistribution
    υ::T1
    τ::T2
    σ::T3
end

Markov(; υ, τ, σ) = Markov(υ, τ, σ)

function logpdf(d::Markov, rts::Array{Float64, 1})
    LL = computeLL(rts, τ = d.τ, σ = d.σ, υ = d.υ)
    return LL
end

loglikelihood(d::Markov, data::Array{Float64, 1}) = logpdf(d, data)

function simulate(parms, n_trials)
    Q = QMatrix(; parms...)
    R = RMatrix()
    QR = QRMatrix(Q, R)
    N = size(Q, 1)
    rts = fill(0.0, n_trials)
    for trial = 1:n_trials
        s0 = 1
        s1 = 1
        t = 0.0
        while s0 != N
            w = weights(Q[s0, :])
            s1 = sample(1:N, w)
            t += rand(Exponential(1 / R[s0, s1]))
            s0 = s1
        end
        rts[trial] = t
    end
    return rts
end

function QMatrix(; υ, τ, σ)
    N = 5
    Q = zeros(typeof(υ), N, N)
    for c = 1:N
        for r = 1:(N - 1)
            if isodd(r) && iseven(c)
                flag = ((c - 1) == r) * 1
                Q[r, c] = Prob(υ, τ, σ, flag)
            elseif iseven(r) && isodd(c) && (c > r) && (c < r + 2)
                Q[r, c] = 1.0
            elseif isodd(r) && (c == 1)
                Q[r, c] = Prob(υ, τ, σ, 2)
            end
        end
    end
    return Q
end

function RMatrix(N = 5)
    R = fill(0.0, N, N)
    λ = [1.0 / 0.085; 1.0 / 0.06]
    cnt = 0
    for c = 1:N
        for r = 1:(N - 1)
            if isodd(r) && iseven(c)
                R[r, c] = 20.0
            elseif iseven(r) && isodd(c) && (c > r) && (c < r + 2)
                cnt += 1
                R[r, c] = λ[cnt]
            elseif isodd(r) && (c == 1)
                R[r, c] = 20.0
            end
        end
    end
    return R
end

function Prob(υ, τ, σ, flag)
    N = 5
    u = zeros(typeof(υ), N - 2)
    u[1] = υ
    u[end] = τ
    vals = exp.(u / σ)
    probs = vals ./ sum(vals)
    if flag == 1
        return maximum(probs[1:(end - 1)])
    elseif flag == 0
        return minimum(probs[1:(end - 1)])
    else
        return probs[end]
    end
end

function QRMatrix(Q, R)
    QR = R .* Q
    diag!(QR)
    return QR
end

function diag!(QR)
    N = size(QR, 1)
    for r = 1:N
        QR[r, r] = 0.0
    end
    d = sum(QR, dims = 2)
    for r = 1:N
        QR[r, r] = -d[r]
    end
end

function computeLL(rts; υ, τ, σ)
    Q = QMatrix(; υ, τ, σ)
    R = RMatrix()
    QR = QRMatrix(Q, R)
    LL = 0.0
    for rt in rts
        LL += Loglikelihood(QR, rt)
    end
    return LL
end

function likelihood(QR, rt)
    k = 500
    N = size(QR, 1)
    val = (one(QR) + QR * rt / k)^k
    val = QR * val
    return val[1, end]
end

Loglikelihood(QR, rt) = log(likelihood(QR, rt))

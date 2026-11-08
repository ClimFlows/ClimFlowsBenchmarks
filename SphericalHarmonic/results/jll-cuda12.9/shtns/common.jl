using Random, LinearAlgebra, Statistics, TOML, Dates, SHA

const ROOT = normpath(joinpath(@__DIR__, ".."))

"""Orthonormal associated Legendre function and colatitude derivative.
Independent Float64 recurrence, including Condon–Shortley phase.
"""
function legendre_mode(l, m, theta)
    x, s = cos(theta), sin(theta)
    p = inv(sqrt(4pi))
    for k in 1:m
        p *= -sqrt((2k + 1) / (2k)) * s
    end
    prev = 0.0
    if l > m
        prev, p = p, sqrt(2m + 3) * x * p
        for k in (m + 2):l
            a = sqrt((4k^2 - 1) / (k^2 - m^2))
            b = sqrt(((2k + 1) * ((k - 1)^2 - m^2)) / ((2k - 3) * (k^2 - m^2)))
            prev, p = p, a * x * p - b * prev
        end
    end
    c = l == m ? 0.0 : sqrt((2l + 1) / (2l - 1) * (l^2 - m^2))
    return p, (l * x * p - c * prev) / s
end

"""Shared analytic vector fixture: grad(chi) + r × grad(psi), unit sphere.
Both potentials contain low and near-cutoff modes, different in every batch member.
Arrays use (latitude, longitude, batch); evaluated in Float64 then rounded once.
"""
function fixture(theta, phi, L, B, ::Type{T}, seed) where T
    modes = unique([(1, 0), (2, 1), (3, 2), (L ÷ 2, L ÷ 3), (L - 1, 1), (L, L)])
    rng = MersenneTwister(seed)
    coeff = randn(rng, length(modes), B, 4)
    shape = (length(theta), length(phi), B)
    vt, vp, div, vor = (zeros(Float64, shape) for _ in 1:4)
    for (q, (l, m)) in enumerate(modes)
        ll = l * (l + 1)
        p_and_d = [legendre_mode(l, m, t) for t in theta]
        for b in 1:B, j in eachindex(phi)
            c, s = cos(m * phi[j]), sin(m * phi[j])
            a, ai, t, ti = (coeff[q, b, k] / ll for k in 1:4)
            chi, psi = a*c + ai*s, t*c + ti*s
            dchi, dpsi = m*(-a*s + ai*c), m*(-t*s + ti*c)
            for i in eachindex(theta)
                p, dp = p_and_d[i]
                invsin = inv(sin(theta[i]))
                vt[i,j,b] += dp*chi - p*dpsi*invsin
                vp[i,j,b] += p*dchi*invsin + dp*psi
                div[i,j,b] -= ll*p*chi
                vor[i,j,b] -= ll*p*psi
            end
        end
    end
    return (; vt=T.(vt), vp=T.(vp), div=T.(div), vor=T.(vor), modes)
end

function relative_error(a, b)
    F = eltype(a) <: Complex || eltype(b) <: Complex ? ComplexF64 : Float64
    x, y = F.(a), F.(b)
    return norm(x - y) / max(norm(y), eps(Float64))
end

function write_toml(path, data)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, data; sorted=true)
    end
end

function options(args)
    o = Dict{String,String}()
    for a in args
        startswith(a, "--") || error("Expected --key=value, got $a")
        pair = split(a[3:end], '='; limit=2)
        o[pair[1]] = length(pair) == 2 ? pair[2] : "true"
    end
    allowed = Set(["backend", "out", "lmax", "precision", "samples", "warmup", "batch", "validate-only", "mode", "seconds-total", "min-samples"])
    isempty(setdiff(keys(o), allowed)) || error("Unknown options: $(setdiff(keys(o), allowed))")
    return o
end

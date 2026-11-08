using SHTnsKit, GPUArrays, GPUArraysCore, KernelAbstractions

function make_case(L, nlat, nlon, B, T, seed)
    cfg = create_gauss_config(L, nlat; nlon)
    ref = fixture(cfg.θ, cfg.φ, L, B, T, seed)
    input = (CuArray(ref.vt), CuArray(ref.vp))
    eigen = CuArray(reshape(T[-l*(l+1) for l in 0:L], L+1, 1, 1))
    inverse = CuArray(reshape(T[l == 0 ? 0 : -1/(l*(l+1)) for l in 0:L], L+1, 1, 1))
    function analysis(v)
        s, t = analysis_sphtor_batch(cfg, v...)
        return (s .* eigen, t .* eigen)
    end
    synthesis(dz) = synthesis_sphtor_batch(cfg, dz[1] .* inverse, dz[2] .* inverse)
    scalar(c) = synthesis_batch(cfg, c; real_output=true, use_rfft=true)
    return (; input, analysis, synthesis, scalar, ref,
        unpack = x -> Array(x), nlat_actual=cfg.nlat,
        notes="Public allocating vector batch API; includes potential/divergence-vorticity conversions; native internal synchronizations retained")
end

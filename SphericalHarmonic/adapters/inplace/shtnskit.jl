using SHTnsKit, GPUArrays, GPUArraysCore, KernelAbstractions

# Upstream has no in-place vector batch API. Reuse its exact batch kernels
# with fixed workspaces and FFT plans; do not replace the mathematical algorithm.
function make_case(L, nlat, nlon, B, T, seed)
    cfg = create_gauss_config(L, nlat; nlon)
    ref = fixture(cfg.θ, cfg.φ, L, B, T, seed)
    input = (CuArray(ref.vt), CuArray(ref.vp))
    eigen = CuArray(reshape(T[-l*(l+1) for l in 0:L], L+1, 1, 1))
    inverse = CuArray(reshape(T[l == 0 ? 0 : -1/(l*(l+1)) for l in 0:L], L+1, 1, 1))
    ext = Base.get_extension(SHTnsKit, :SHTnsKitGPUExt)
    tables = ext._cuda_vector_tables(cfg, T)
    ft, fp = (CUDA.zeros(Complex{T}, nlat, nlon, B) for _ in 1:2)
    s, t, d, z = (CUDA.zeros(Complex{T}, L+1, L+1, B) for _ in 1:4)
    vt, vp = map(similar, input)
    forward = SHTnsKit.FFTW.plan_fft!(ft, 2)
    backward = SHTnsKit.FFTW.plan_bfft!(ft, 2)
    ak = ext.vector_batch_analysis_kernel!(CUDA.CUDABackend(), 256)
    sk = ext.vector_batch_synthesis_kernel!(CUDA.CUDABackend(), 256)
    cphi = T(SHTnsKit._analysis_phi_scale(cfg))
    invscale = T(SHTnsKit.phi_inv_scale(cfg))
    function analysis(v)
        ft .= v[1]; fp .= v[2]
        mul!(ft, forward, ft); mul!(fp, forward, fp)
        ak(s, t, ft, fp, tables.dtheta, tables.over_sin, tables.weights,
           tables.scales, tables.sint, cphi, L, L, cfg.mres, cfg.robert_form;
           ndrange=size(s))
        d .= s .* eigen; z .= t .* eigen
        return (d,z)
    end
    function synthesis(dz)
        s .= dz[1] .* inverse; t .= dz[2] .* inverse
        fill!(ft,0); fill!(fp,0)
        sk(ft, fp, s, t, tables.dtheta, tables.over_sin, tables.scales,
           tables.sint, invscale, nlon, L, L, cfg.mres, true, cfg.robert_form;
           ndrange=(nlat,L+1,B))
        mul!(ft, backward, ft); mul!(fp, backward, fp)
        vt .= real.(ft) ./ nlon; vp .= real.(fp) ./ nlon
        return (vt,vp)
    end
    scalar(c) = synthesis_batch(cfg, c; real_output=true, use_rfft=true)
    return (; input, analysis, synthesis, scalar, ref,
        unpack=x->Array(x), nlat_actual=nlat,
        notes="Adapted preallocated batch pipeline: unchanged upstream SHTnsKit batch kernels, reused complex FFT plans, no per-call arrays; not a public upstream in-place batch API")
end

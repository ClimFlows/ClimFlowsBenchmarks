using Libdl
# Set before importing SHTns: its library path is evaluated at precompile time.
const CUSTOM_SHTNS = get(ENV, "SHTNS_PATH", joinpath(ROOT, "build", "shtns-c", "libshtns_cuda.so"))
isfile(CUSTOM_SHTNS) || error("CUDA SHTns library missing: $CUSTOM_SHTNS. Run scripts/build_shtns.sh first.")
ENV["SHTNS_PATH"] = abspath(CUSTOM_SHTNS)
using SHTns
abspath(SHTns.libshtns) == abspath(CUSTOM_SHTNS) || error("SHTns cached another library. Rebuild the SHTns compile cache with SHTNS_PATH set; see README.md.")
Libdl.dlsym(Libdl.dlopen(SHTns.libshtns), :cu_spat_to_SHsphtor)

function make_case(L, nlat, nlon, B, T, seed)
    T === Float64 || error("Pinned SHTns.jl CUDA methods support Float64 only")
    cfg = SHTns.SHTnsCfg(L, L, 1, nlat, nlon;
        shtype=SHTns.Gauss(; gpu=true), howmany=B)
    cfg.nlat == nlat && cfg.nphi == nlon || error("SHTns changed the requested grid")
    theta, phi = SHTns.grid(cfg)
    ref = fixture(theta, phi, L, B, T, seed)
    function padded(x)
        a = zeros(T, cfg.nlat_padded, nlon, B)
        a[1:nlat,:,:] .= x
        return CuArray(a)
    end
    input = (padded(ref.vt), padded(ref.vp))
    ll = Float64[-l*(l+1) for m in 0:L for l in m:L]
    eigen = CuArray(reshape(ll, :, 1))
    inverse = CuArray(reshape([x == 0 ? 0.0 : inv(x) for x in ll], :, 1))
    # SHTns toroidal potential has the opposite sign to r × grad(psi).
    function analysis(v)
        s, t = SHTns.analys(cfg, v...)
        return (s .* eigen, .-t .* eigen)
    end
    synthesis(dz) = SHTns.synth(cfg, dz[1] .* inverse, .-dz[2] .* inverse)
    scalar(c) = SHTns.synth(cfg, c)
    unpack(x) = Array(x)[1:nlat,:,:]
    return (; input, analysis, synthesis, scalar, unpack, ref,
        nlat_actual=cfg.nlat_padded,
        notes="Public allocating vector batch API; analysis copies destructive inputs; includes potential/divergence-vorticity conversions")
end

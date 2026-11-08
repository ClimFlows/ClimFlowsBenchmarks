using Libdl
if get(ENV, "SHTNS_BINARY", "local") == "jll"
    pop!(ENV, "SHTNS_PATH", nothing)
    @eval using SHTns
    SHTns.libshtns == SHTns.SHTns_jll.LibSHTns_path || error("SHTns cache references a custom library; rebuild its cache without SHTNS_PATH")
else
# Set before importing SHTns: its library path is evaluated at precompile time.
const CUSTOM_SHTNS = get(ENV, "SHTNS_PATH", joinpath(ROOT, "build", "shtns-c", "libshtns_cuda.so"))
isfile(CUSTOM_SHTNS) || error("CUDA SHTns library missing: $CUSTOM_SHTNS. Run scripts/build_shtns.sh first.")
ENV["SHTNS_PATH"] = abspath(CUSTOM_SHTNS)
using SHTns
abspath(SHTns.libshtns) == abspath(CUSTOM_SHTNS) || error("SHTns cached another library. Rebuild the SHTns compile cache with SHTNS_PATH set; see README.md.")
end
Libdl.dlsym(Libdl.dlopen(SHTns.libshtns), :cu_spat_to_SHsphtor)

function make_case(L, nlat, nlon, B, T, seed)
    T === Float64 || error("Pinned SHTns.jl CUDA methods support Float64 only")
    # Set up the C on-device API on Julia's actual stream before GPU init.
    # Using CUDA's special legacy-stream object breaks its async sync worker.
    cfg = SHTns.SHTnsCfg(L, L, 1, nlat, nlon;
        shtype=SHTns.Gauss(), howmany=B)
    stream = CUDA.stream()
    ccall((:cushtns_set_streams, SHTns.libshtns), Cvoid,
        (SHTns.shtns_cfg, CUDA.CUstream, CUDA.CUstream), cfg.cfg, stream, C_NULL)
    device = ccall((:cushtns_init_gpu, SHTns.libshtns), Cint,
        (SHTns.shtns_cfg,), cfg.cfg)
    device >= 0 || error("SHTns could not initialize the on-device CUDA configuration")
    cfg.shtype = SHTns.Gauss(; gpu=true)
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
    scratch = map(similar, input)
    s, t, d, z = (CUDA.zeros(ComplexF64, length(ll), B) for _ in 1:4)
    output = map(similar, input)
    function analysis(v)
        copyto!(scratch[1], v[1]); copyto!(scratch[2], v[2])
        SHTns.analys!(cfg, scratch[1], scratch[2], s, t)
        d .= s .* eigen; z .= .-t .* eigen
        return (d,z)
    end
    function synthesis(dz)
        s .= dz[1] .* inverse; t .= .-dz[2] .* inverse
        SHTns.synth!(cfg, s, t, output[1], output[2])
        return output
    end
    scalar(c) = SHTns.synth(cfg, c)
    unpack(x) = Array(x)[1:nlat,:,:]
    return (; input, analysis, synthesis, scalar, unpack, ref,
        nlat_actual=cfg.nlat_padded,
        notes="Preallocated public vector batch API; protective input copies and spectral conversions included; C initialized on Julia stream")
end

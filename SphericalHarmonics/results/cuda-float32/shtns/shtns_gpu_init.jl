# SHTns 3.7 uses atomic Float32 additions when analysis spans latitude blocks.
# For the T255 / 64-field case, one 192-thread latitude block removes their
# run-to-run noise, which otherwise exceeds the graph/direct check (1e-5).
# LSPAN_A=12 satisfies the C kernel's divisibility constraints at 192 threads.
const SHTNS_T255_FP32_CONFIG = "4\n4\n2\n6\n4\n12\n"

function initialize_shtns_gpu(cfg, T, L, nlat, B)
    init() = ccall((:cushtns_init_gpu, SHTns.libshtns), Cint,
        (SHTns.shtns_cfg,), cfg.cfg)
    profile = get(ENV, "SHTNS_T255_FP32_TUNING", "single-block")
    profile in ("single-block", "default") || error("Unknown SHTns Float32 tuning profile: $profile")
    tuned = profile == "single-block" && T === Float32 && L == 255 && nlat == 384 && B == 64
    if tuned
        device = mktempdir() do dir
            write(joinpath(dir,"sht_gpu.conf"), SHTNS_T255_FP32_CONFIG)
            cd(dir) do
                withenv("SHTNS_GPU_CONF" => "1") do
                    init()
                end
            end
        end
        return device, "T255 Float32: SHTns GPU configuration [4,4,2,6,4,12], one latitude block for deterministic analysis"
    end
    return init(), "SHTns default GPU configuration"
end

include("common.jl")
using CUDA, Libdl
CUDA.allowscalar(false)
include(joinpath(ROOT,"adapters","shtns_float32_prototype.jl"))
"--default" in ARGS && (ENV["SHTNS_T255_FP32_TUNING"] = "default")
c = make_case(255,384,768,64,Float32,20261107)
f() = c.analysis(c.input)
for _ in 1:3; f(); CUDA.device_synchronize(); end
reference = map(Array, f()); CUDA.device_synchronize()
velocity = c.synthesis(f()); CUDA.device_synchronize()
println("analytic velocity error: ", max(relative_error(c.unpack(velocity[1]),c.ref.vt), relative_error(c.unpack(velocity[2]),c.ref.vp)))
for i in 1:10
    actual = f(); CUDA.device_synchronize()
    println("direct repeat ",i,": ",maximum(relative_error(Array(x),y) for (x,y) in zip(actual,reference)))
end
targets = f(); CUDA.device_synchronize()
graph = CUDA.capture(f; throw_error=true)
exe = CUDA.instantiate(graph)
for i in 1:10
    CUDA.launch(exe,CUDA.stream()); CUDA.device_synchronize()
    println("graph repeat ",i,": ",maximum(relative_error(Array(x),y) for (x,y) in zip(targets,reference)))
end

if "--full-spectrum" in ARGS
    cfg = c.analysis.cfg
    cpu = SHTns.SHTnsCfg(255,255,1,384,768; howmany=64, shtype=SHTns.QuickInit())
    rng = MersenneTwister(20261110)
    coeff = ntuple(2) do _
        x = randn(rng,ComplexF64,SHTns.nlm(cfg),64)
        x[cfg.mi .== 0,:] .= real.(x[cfg.mi .== 0,:])
        x[1,:] .= 0
        x
    end
    spectra = map(x->CuArray(ComplexF32.(x)),coeff)
    spatial = map(similar,c.input)
    prototype_synthesis!(cfg,spectra...,spatial...)
    CUDA.device_synchronize()
    expected = SHTns.synth(cpu,coeff...)
    errors = [relative_error(Array(x),y) for (x,y) in zip(spatial,expected)]
    println("full-spectrum CPU reference errors: ", errors)
    @assert maximum(errors) < 5e-4
    recovered = map(similar,spectra)
    prototype_analysis!(cfg,spatial...,recovered...)
    CUDA.device_synchronize()
    errors = [relative_error(Array(x),y) for (x,y) in zip(recovered,coeff)]
    println("full-spectrum spectral roundtrip errors: ", errors)
    @assert maximum(errors) < 5e-4
end

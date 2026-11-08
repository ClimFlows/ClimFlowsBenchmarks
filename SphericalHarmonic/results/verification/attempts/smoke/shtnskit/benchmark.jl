include("common.jl")
using CUDA, Pkg

function measure(f, warmup, samples)
    for _ in 1:warmup
        f(); CUDA.synchronize()
    end
    times = Float64[]
    for _ in 1:samples
        GC.gc(); CUDA.synchronize()
        t = time_ns()
        output = f()
        CUDA.synchronize()
        push!(times, (time_ns()-t)/1e9)
        output = nothing
    end
    return Dict("seconds"=>times, "minimum_seconds"=>minimum(times),
        "median_seconds"=>median(times), "p95_seconds"=>quantile(times,0.95))
end

function run_case(case, B, T, config, opts)
    L,nlat,nlon = (case[k] for k in ("lmax","nlat","nlon"))
    CUDA.reclaim(); CUDA.synchronize()
    free_before = Int(CUDA.available_memory())
    start = time_ns()
    c = make_case(L,nlat,nlon,B,T,config["seed"])
    CUDA.synchronize()
    setup_seconds = (time_ns()-start)/1e9
    dz = c.analysis(c.input)
    reconstructed = c.synthesis(dz)
    CUDA.synchronize()
    recovered = hasproperty(c,:unpack_vector) ? c.unpack_vector(reconstructed) : map(c.unpack,reconstructed)
    de,ze = c.unpack(c.scalar(dz[1])), c.unpack(c.scalar(dz[2]))
    errors = Dict("velocity_theta_relative_l2"=>relative_error(recovered[1],c.ref.vt),
        "velocity_phi_relative_l2"=>relative_error(recovered[2],c.ref.vp),
        "divergence_relative_l2"=>relative_error(de,c.ref.div),
        "vorticity_relative_l2"=>relative_error(ze,c.ref.vor))
    tolerance = T === Float32 ? 5e-4 : 5e-10
    all(isfinite, values(errors)) || error("Nonfinite validation errors: $errors")
    valid = maximum(values(errors)) <= tolerance
    result = Dict{String,Any}("status"=>valid ? "validated" : "validation_failed",
        "backend"=>opts["backend"], "lmax"=>L, "precision"=>string(T), "batch"=>B,
        "nlat"=>nlat, "nlon"=>nlon, "nlat_storage"=>c.nlat_actual,
        "setup_and_fixture_seconds"=>setup_seconds, "errors"=>errors,
        "tolerance"=>tolerance, "notes"=>c.notes,
        "memory_free_before_bytes"=>free_before,
        "memory_free_after_validation_bytes"=>Int(CUDA.available_memory()),
        "analytic_modes"=>[collect(m) for m in c.ref.modes])
    valid || return result  # Never publish timings for an invalid transform.
    if !haskey(opts,"validate-only")
        for (name,f) in (("analysis",()->c.analysis(c.input)),
                         ("synthesis",()->c.synthesis(dz)),
                         ("roundtrip",()->c.synthesis(c.analysis(c.input))))
            stats = measure(f, parse(Int,get(opts,"warmup",string(config["warmup"]))),
                parse(Int,get(opts,"samples",string(config["samples"]))))
            stats["vector_fields_per_second"] = B/stats["median_seconds"]
            result[name] = stats
        end
        result["status"] = "ok"
    end
    return result
end

function main()
    opts=options(ARGS)
    backend=get(opts,"backend", "")
    backend in ("shtns","shtnskit","speedy") || error("Specify --backend=shtns|shtnskit|speedy")
    config=TOML.parsefile(joinpath(ROOT,"config.toml"))
    B=parse(Int,get(opts,"batch",string(config["batch"]))); B>0 || error("batch must be positive")
    parse(Int,get(opts,"samples",string(config["samples"])))>0 || error("samples must be positive")
    out=haskey(opts,"out") ? joinpath(abspath(opts["out"]),backend) :
        joinpath(ROOT,"results",Dates.format(now(),"yyyymmdd-HHMMSS")*"-"*backend)
    mkpath(out)
    cp(joinpath(ROOT,"config.toml"),joinpath(out,"config.toml"); force=true)
    manifest=joinpath(dirname(Base.active_project()),"Manifest.toml")
    cp(manifest,joinpath(out,"Manifest.toml"); force=true)
    cp(joinpath(ROOT,"sources","revisions.toml"),joinpath(out,"sources.toml"); force=true)
    for file in ("scripts/common.jl","scripts/benchmark.jl","adapters/$backend.jl")
        cp(joinpath(ROOT,file),joinpath(out,basename(file));force=true)
    end
    open(joinpath(out,"nvidia-smi.txt"),"w") do io
        try
            run(pipeline(`nvidia-smi`,stdout=io,stderr=io))
        catch e
            println(io,"nvidia-smi unavailable: ",e)
        end
    end
    open(joinpath(out,"cuda-versioninfo.txt"),"w") do io; CUDA.versioninfo(io); end
    metadata=Dict("julia_version"=>string(VERSION), "backend"=>backend,
        "utc_started"=>string(now(UTC)), "gpu"=>CUDA.name(CUDA.device()),
        "compute_capability"=>string(CUDA.capability(CUDA.device())),
        "total_memory_bytes"=>Int(CUDA.total_memory()), "julia_threads"=>Threads.nthreads(),
        "manifest_sha256"=>bytes2hex(sha256(read(manifest))), "arguments"=>ARGS,
        "measurement"=>"Synchronized wall time, allocating public pipelines, plans reused; no host/device transfer in timed region")
    write_toml(joinpath(out,"metadata.toml"),metadata)
    open(joinpath(out,"packages.txt"),"w") do io; Pkg.status(;io, mode=Pkg.PKGMODE_MANIFEST); end
    CUDA.functional() || error("CUDA is not functional; see cuda-versioninfo.txt")
    CUDA.allowscalar(false)
    include(joinpath(ROOT,"adapters",backend*".jl"))
    cases=filter(c->!haskey(opts,"lmax") || c["lmax"] in parse.(Int,split(opts["lmax"],',')), config["cases"])
    isempty(cases) && error("No requested lmax exists in config.toml")
    precisions=split(get(opts,"precision",join(config["precisions"],',')),',')
    all(p->p in ("Float32","Float64"),precisions) || error("precision must be Float32 or Float64")
    failed=false
    for case in cases, precision in precisions
        L=case["lmax"]; filename=joinpath(out,"T$L-$precision.toml")
        if backend=="shtns" && precision=="Float32"
            write_toml(filename,Dict("status"=>"unsupported", "reason"=>"Pinned SHTns.jl CUDA API uses Float64", "backend"=>backend,"lmax"=>L,"precision"=>precision,"batch"=>B))
            continue
        end
        println("$backend T$L $precision batch=$B"); flush(stdout)
        try
            r=Base.invokelatest(run_case,case,B,precision=="Float32" ? Float32 : Float64,config,opts)
            write_toml(filename,r)
            println("  ",r["status"]," errors=",r["errors"])
            failed |= r["status"]=="validation_failed"
        catch e
            failed=true
            trace=sprint(showerror,e,catch_backtrace())
            write_toml(filename,Dict("status"=>"error","backend"=>backend,"lmax"=>L,"precision"=>precision,"batch"=>B,"error"=>trace))
            println(stderr,trace)
        end
        GC.gc(); CUDA.reclaim()
    end
    failed && exit(1)
end
main()

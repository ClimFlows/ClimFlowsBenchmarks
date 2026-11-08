include("common.jl")
using CUDA, Pkg

function measure_allocating(f, warmup, samples)
    for _ in 1:warmup
        f(); CUDA.device_synchronize()
    end
    times = Float64[]
    for _ in 1:samples
        GC.gc(); CUDA.device_synchronize()
        t = time_ns()
        output = f()
        CUDA.device_synchronize()
        push!(times, (time_ns()-t)/1e9)
        output = nothing
    end
    return Dict("seconds"=>times, "minimum_seconds"=>minimum(times),
        "median_seconds"=>median(times), "p95_seconds"=>quantile(times,0.95))
end

function measure_preallocated(f, warmup, budget, min_samples)
    for _ in 1:max(warmup,3)
        f(); CUDA.device_synchronize()
    end
    rawdata(x) = x isa CUDA.CuArray ? x : x.data
    targets = f(); CUDA.device_synchronize()
    expected = map(x->Array(rawdata(x)), targets)
    direct_host_bytes = @allocated f()
    CUDA.device_synchronize()
    direct_gpu_bytes = CUDA.@allocated f()
    CUDA.device_synchronize()
    direct_gpu_bytes == 0 || error("Direct preallocated pipeline still allocates $direct_gpu_bytes GPU bytes")
    stream = CUDA.stream()
    graph_error = ""
    exec = nothing
    try
        graph = CUDA.capture(f; throw_error=true)
        exec = CUDA.instantiate(graph)
    catch e
        graph_error = sprint(showerror,e)
    end
    replay = exec === nothing ? f : let exe=exec; ()->CUDA.launch(exe,stream); end
    # Poison every output before replay so a missing graph node cannot pass
    # just because a previous direct call left a correct result in the buffers.
    for x in targets; fill!(rawdata(x),convert(eltype(rawdata(x)),NaN)); end
    replay(); CUDA.device_synchronize()
    replay_error = maximum(relative_error(Array(rawdata(x)), y) for (x,y) in zip(targets,expected))
    isfinite(replay_error) && replay_error < 1e-5 || error("Graph output differs from direct execution: $replay_error")
    expected = nothing
    for _ in 1:3; replay(); CUDA.device_synchronize(); end
    host_bytes = minimum([@allocated(replay()) for _ in 1:3])
    CUDA.device_synchronize()
    gpu_bytes = minimum([CUDA.@allocated(replay()) for _ in 1:3])
    CUDA.device_synchronize()
    gpu_bytes == 0 || error("Preallocated pipeline still allocates $gpu_bytes GPU bytes per call")
    host_bytes == 0 || error("Selected execution path still allocates $host_bytes CPU bytes per call: $graph_error")
    first, last = CUDA.CuEvent(), CUDA.CuEvent()
    CUDA.record(first,stream); replay(); CUDA.record(last,stream)
    CUDA.device_synchronize()
    estimate = max(Float64(CUDA.elapsed(first,last)),1e-7)
    repeats = clamp(ceil(Int,0.002/estimate),1,1000)
    times, wall = Float64[], Float64[]
    spent = 0.0
    GC.gc(); CUDA.device_synchronize()
    while spent < budget || length(times) < min_samples
        start = time_ns()
        CUDA.record(first,stream)
        for _ in 1:repeats; replay(); end
        CUDA.record(last,stream)
        CUDA.device_synchronize()
        host_time = (time_ns()-start)/1e9
        gpu_time = Float64(CUDA.elapsed(first,last))
        push!(times,gpu_time/repeats); push!(wall,host_time/repeats)
        spent += gpu_time
    end
    return Dict{String,Any}("seconds"=>times,"wall_seconds"=>wall,
        "minimum_seconds"=>minimum(times),"median_seconds"=>median(times),
        "p95_seconds"=>quantile(times,0.95),"wall_median_seconds"=>median(wall),
        "gpu_measured_seconds"=>spent,"target_gpu_seconds"=>budget,
        "samples"=>length(times),"repeats_per_sample"=>repeats,
        "executions"=>length(times)*repeats,
        "gpu_allocated_bytes_per_call"=>gpu_bytes,"host_allocated_bytes_per_call"=>host_bytes,
        "direct_gpu_allocated_bytes_per_call"=>direct_gpu_bytes,
        "direct_host_allocated_bytes_per_call"=>direct_host_bytes,
        "execution_path"=>exec===nothing ? "direct" : "CUDA graph replay",
        "graph_capture_error"=>graph_error,"graph_output_relative_error"=>replay_error)
end

const GPU_BUDGET = Ref(0.0)
const OPS_LEFT = Ref(0)

function run_case(case, B, T, config, opts)
    L,nlat,nlon = (case[k] for k in ("lmax","nlat","nlon"))
    CUDA.reclaim(); CUDA.device_synchronize()
    free_before = Int(CUDA.free_memory())
    start = time_ns()
    c = make_case(L,nlat,nlon,B,T,config["seed"])
    CUDA.device_synchronize()
    setup_seconds = (time_ns()-start)/1e9
    dz = c.analysis(c.input)
    reconstructed = c.synthesis(dz)
    CUDA.device_synchronize()
    recovered = hasproperty(c,:unpack_vector) ? c.unpack_vector(reconstructed) : map(c.unpack,reconstructed)
    reconstructed = nothing
    GC.gc(); CUDA.reclaim()
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
        "memory_free_after_validation_bytes"=>Int(CUDA.free_memory()),
        "analytic_modes"=>[collect(m) for m in c.ref.modes], "mode"=>get(opts,"mode","inplace"))
    valid || return result  # Never publish timings for an invalid transform.
    if !haskey(opts,"validate-only")
        recovered = nothing; de = nothing; ze = nothing
        GC.gc(); CUDA.reclaim()
        for (name,f) in (("analysis",()->c.analysis(c.input)),
                         ("synthesis",()->c.synthesis(dz)),
                         ("roundtrip",()->c.synthesis(c.analysis(c.input))))
            warmup = parse(Int,get(opts,"warmup",string(config["warmup"])))
            if get(opts,"mode","inplace") == "inplace"
                target = max(0.0,GPU_BUDGET[]/max(1,OPS_LEFT[]))
                stats = measure_preallocated(f,warmup,target,parse(Int,get(opts,"min-samples",string(get(config,"min_samples",10)))))
                GPU_BUDGET[] -= stats["gpu_measured_seconds"]
                OPS_LEFT[] -= 1
            else
                stats = measure_allocating(f,warmup,parse(Int,get(opts,"samples",string(config["samples"]))))
            end
            stats["vector_fields_per_second"] = B/stats["median_seconds"]
            result[name] = stats
        end
        # Validate the final state after repeated graph replays, not only setup.
        repeated = c.synthesis(dz); CUDA.device_synchronize()
        final_velocity = hasproperty(c,:unpack_vector) ? c.unpack_vector(repeated) : map(c.unpack,repeated)
        post = max(relative_error(final_velocity[1],c.ref.vt),relative_error(final_velocity[2],c.ref.vp))
        result["post_measurement_velocity_error"] = post
        post <= tolerance || error("Repeated execution corrupted the preallocated buffers: $post")
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
    preferences=joinpath(dirname(manifest),"LocalPreferences.toml")
    isfile(preferences) && cp(preferences,joinpath(out,"LocalPreferences.toml");force=true)
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
        "measurement"=>get(opts,"mode","inplace")=="inplace" ? "Preallocated pipelines, warmed CUDA graphs where supported, CUDA event timings with reusable events; host allocations reported separately" : "Synchronized wall time, allocating public pipelines")
    write_toml(joinpath(out,"metadata.toml"),metadata)
    open(joinpath(out,"packages.txt"),"w") do io; Pkg.status(;io, mode=Pkg.PKGMODE_MANIFEST); end
    CUDA.functional() || error("CUDA is not functional; see cuda-versioninfo.txt")
    CUDA.allowscalar(false)
    mode=get(opts,"mode","inplace")
    mode in ("inplace","allocating") || error("Unknown mode $mode")
    adapter=mode=="inplace" ? joinpath(ROOT,"adapters","inplace",backend*".jl") : joinpath(ROOT,"adapters",backend*".jl")
    cp(adapter,joinpath(out,backend*".jl");force=true)
    include(adapter)
    if backend=="shtns"
        metadata["shtns_binary_origin"] = get(ENV,"SHTNS_BINARY","local")
        metadata["shtns_library"] = SHTns.libshtns
        metadata["shtns_library_sha256"] = bytes2hex(sha256(read(SHTns.libshtns)))
        for file in ("configure.log","make.log","nvcc-version.txt","cc-version.txt")
            source=joinpath(dirname(SHTns.libshtns),file)
            isfile(source) && cp(source,joinpath(out,file);force=true)
        end
        write_toml(joinpath(out,"metadata.toml"),metadata)
    end
    cases=filter(c->!haskey(opts,"lmax") || c["lmax"] in parse.(Int,split(opts["lmax"],',')), config["cases"])
    isempty(cases) && error("No requested lmax exists in config.toml")
    precisions=split(get(opts,"precision",join(config["precisions"],',')),',')
    all(p->p in ("Float32","Float64"),precisions) || error("precision must be Float32 or Float64")
    if mode=="inplace"
        sort!(cases;by=c->c["lmax"],rev=true)
        sort!(precisions;rev=true)
    end
    supported=length(cases)*count(p->!(backend=="shtns" && p=="Float32"),precisions)
    OPS_LEFT[]=3supported
    full_combinations=length(config["cases"])*(3length(config["precisions"])-count(==("Float32"),config["precisions"]))
    default_budget=get(config,"gpu_seconds_total",60.0)*OPS_LEFT[]/(3full_combinations)
    GPU_BUDGET[]=parse(Float64,get(opts,"seconds-total",string(default_budget)))
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

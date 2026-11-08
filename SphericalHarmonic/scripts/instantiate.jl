using Pkg
VERSION >= v"1.11" || error("Use Julia 1.11 or later (reference: Julia 1.11.9)")
backend = only(ARGS)
if !isfile("Manifest.toml")
    paths = backend == "speedy" ? ["SpeedyWeatherInternals", "LowerTriangularArrays", "RingGrids", "SpeedyTransforms"] : String[]
    specs = backend == "speedy" ? [PackageSpec(path="../../vendor/speedy/$p") for p in paths] :
        [PackageSpec(path="../../vendor/" * (startswith(backend,"shtns") && backend != "shtnskit" ? "shtns-julia" : "shtnskit"))]
    Pkg.develop(specs)
end
Pkg.instantiate()
Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)

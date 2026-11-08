using Pkg
VERSION >= v"1.11" || error("Use Julia 1.11 or later (reference: Julia 1.11.9)")
isfile("Manifest.toml") || error("The pinned Manifest.toml is required")
Pkg.instantiate()
Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)

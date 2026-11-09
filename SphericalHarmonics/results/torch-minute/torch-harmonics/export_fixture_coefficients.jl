include("common.jl")

length(ARGS) == 4 || error("Usage: export_fixture_coefficients.jl L B SEED OUTPUT")
L, B, seed = parse.(Int, ARGS[1:3])
destination = ARGS[4]
modes, coeff = fixture_coefficients(L, B, seed)
mkpath(dirname(destination))
open(destination, "w") do io
    write(io, vec(coeff))
end
println("Exported $(size(coeff)) coefficients for $(length(modes)) modes to $destination")

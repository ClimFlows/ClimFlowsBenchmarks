using CairoMakie, TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEGREES = (21, 42, 63, 127, 255)
const SERIES = (
    ("SHTns local", "inplace-minute", "shtns", "#14804a", :dash, ("Float64",)),
    ("SHTns JLL", "cuda-float32", "shtns", "#2563eb", :solid, ("Float32", "Float64")),
    ("SHTnsKit", "inplace-minute", "shtnskit", "#b45309", :solid, ("Float32", "Float64")),
    ("SpeedyTransforms", "inplace-minute", "speedy", "#9333ea", :solid, ("Float32", "Float64")),
    ("torch-harmonics", "torch-minute", "torch-harmonics", "#dc2626", :solid, ("Float32", "Float64")),
    ("S2FFT (JAX)", "jax-minute", "s2fft-jax", "#0891b2", :dot, ("Float32", "Float64")),
    ("Dinosaur", "dinosaur-minute", "dinosaur", "#0f766e", :dash, ("Float32",)),
    ("legoESM", "legoesm-minute", "legoesm", "#d97706", :dashdot, ("Float32", "Float64")),
)

fig = Figure(size=(950, 460), fontsize=13)
for (column, precision) in enumerate(("Float32", "Float64"))
    ax = Axis(fig[1, column]; title=precision, yscale=log10,
        ylabel=column == 1 ? "Round trip per batch (ms)" : "",
        xticks=(1:5, ["T$d" for d in DEGREES]),
        yticks=([0.01, 0.1, 1, 10, 100, 1000, 10000], ["0.01", "0.1", "1", "10", "100", "1000", "10000"]),
        xgridvisible=false)
    for (label, run, backend, color, style, supported) in SERIES
        precision in supported || continue
        times = map(DEGREES) do degree
            file = joinpath(ROOT, "results", run, backend, "T$degree-$precision.toml")
            if !isfile(file)
                return NaN
            end
            result = TOML.parsefile(file)
            @assert result["status"] == "ok" && result["batch"] == 64
            1000 * result["roundtrip"]["median_seconds"]
        end
        scatterlines!(ax, 1:5, collect(times); label, color, linestyle=style, markersize=7)
    end
    ylims!(ax, 0.01, 10000)
    hidespines!(ax, :t, :r)
end
Legend(fig[2, :], [LineElement(color=s[4], linestyle=s[5]) for s in SERIES],
    [s[1] for s in SERIES]; orientation=:horizontal, nbanks=2, labelsize=11, framevisible=false)
Label(fig[0, :], "GPU vector transforms · batch 64 · lower is better"; fontsize=16)
destination = joinpath(ROOT, "results", "roundtrip-jax.svg")
save(destination, fig)
isempty(ARGS) || save(only(ARGS), fig) # Optional PNG preview.
println(destination)

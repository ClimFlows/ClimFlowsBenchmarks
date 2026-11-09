using CairoMakie, TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEGREES = (21, 42, 63, 127, 255)
const SERIES = (
    ("SHTns local", "inplace-minute", "shtns", "#14804a", :dash, ("Float64",)),
    ("SHTns JLL", "jll-cuda12.9", "shtns", "#2563eb", :solid, ("Float64",)),
    ("SHTnsKit", "inplace-minute", "shtnskit", "#b45309", :solid, ("Float32", "Float64")),
    ("SpeedyTransforms", "inplace-minute", "speedy", "#9333ea", :solid, ("Float32", "Float64")),
    ("torch-harmonics", "torch-minute", "torch-harmonics", "#dc2626", :solid, ("Float32", "Float64")),
)

fig = Figure(size=(950, 390), fontsize=13)
for (column, precision) in enumerate(("Float32", "Float64"))
    ax = Axis(fig[1, column]; title=precision, yscale=log10,
        ylabel=column == 1 ? "Round trip per batch (ms)" : "",
        xticks=(1:5, ["T$d" for d in DEGREES]),
        yticks=([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"]),
        xgridvisible=false)
    for (label, run, backend, color, style, supported) in SERIES
        precision in supported || continue
        times = map(DEGREES) do degree
            file = joinpath(ROOT, "results", run, backend, "T$degree-$precision.toml")
            result = TOML.parsefile(file)
            @assert result["status"] == "ok" && result["batch"] == 64
            1000 * result["roundtrip"]["median_seconds"]
        end
        scatterlines!(ax, 1:5, collect(times); label, color, linestyle=style, markersize=7)
    end
    ylims!(ax, 0.12, 1000)
    hidespines!(ax, :t, :r)
    axislegend(ax; position=:lt, labelsize=11, framevisible=false)
end
Label(fig[0, :], "CUDA vector transforms · batch 64 · lower is better"; fontsize=16)
destination = joinpath(ROOT, "results", "roundtrip-torch.svg")
save(destination, fig)
isempty(ARGS) || save(only(ARGS), fig) # Optional PNG preview.
println(destination)

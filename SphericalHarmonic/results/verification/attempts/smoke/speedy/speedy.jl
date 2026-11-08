using SpeedyTransforms, RingGrids, LowerTriangularArrays
using SpeedyWeatherInternals.Architectures: GPU
using cuFFT

function make_case(L, nlat, nlon, B, T, seed)
    arch = GPU()
    grid = FullGaussianGrid(nlat ÷ 2, arch)
    RingGrids.get_nlon_max(grid) == nlon || error("Unexpected Speedy longitude count")
    # Extra degree is workspace for vector derivatives, not an extra retained physical mode.
    spectrum = Spectrum(L+1; one_degree_more=true, architecture=arch)
    trans = SpectralTransform(spectrum, grid; NF=T, nlayers=B,
        transform_batch=[1,B], gpu_graphs=true)
    theta, phi = RingGrids.get_colat(grid), RingGrids.get_lon(grid)
    ref = fixture(theta, phi, L, B, T, seed)
    field() = zeros(T, grid, B)
    spectral() = zeros(LowerTriangularArray{Complex{T}}, spectrum, B)
    pack(x) = reshape(permutedims(x, (2,1,3)), nlat*nlon, B)
    u, v = field(), field()
    copyto!(u.data, pack(ref.vp)); copyto!(v.data, pack(-ref.vt))
    coslat = CuArray(reshape(repeat(T.(sin.(theta)); inner=nlon), :, 1))
    # Spectrum indices are one-based; discard the workspace degree in delta/zeta.
    keep = CuArray(reshape(T.(Array(spectrum.l_indices) .<= L+1), :, 1))
    function analysis(uv)
        a, b = field(), field()
        a.data .= uv[1].data ./ coslat
        b.data .= uv[2].data ./ coslat
        us, vs = spectral(), spectral()
        transform!(us, a, trans); transform!(vs, b, trans)
        d, z = spectral(), spectral()
        divergence!(d, us, vs, trans; radius=1)
        curl!(z, us, vs, trans; radius=1)
        d.data .*= keep; z.data .*= keep
        return (d,z)
    end
    function synthesis(dz)
        us, vs = spectral(), spectral()
        UV_from_vordiv!(us, vs, dz[2], dz[1], trans; radius=1)
        a, b = field(), field()
        transform!(a, us, trans); transform!(b, vs, trans)
        a.data ./= coslat; b.data ./= coslat
        return (a,b)
    end
    function scalar(c)
        a = field(); transform!(a, c, trans); return a
    end
    unpack(x) = permutedims(reshape(Array(x.data), nlon, nlat, B), (2,1,3))
    # Native vector order is (east,north); common reference is (south,east).
    unpack_vector(uv) = (-unpack(uv[2]), unpack(uv[1]))
    return (; input=(u,v), analysis, synthesis, scalar, unpack, unpack_vector, ref,
        nlat_actual=nlat,
        notes="Allocating divergence/vorticity pipeline with reused SpectralTransform; metric scaling included; CUDA graphs enabled; one extra workspace degree")
end

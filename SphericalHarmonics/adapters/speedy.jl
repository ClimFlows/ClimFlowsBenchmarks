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
    a, b = field(), field()
    us, vs = spectral(), spectral()
    d, z = spectral(), spectral()
    outu, outv = field(), field()
    # Compose upstream fused FFT loops directly so the complete pipeline can
    # be captured once, without trying to capture nested graph launches.
    cache = SpeedyTransforms.get_cache(CUDA.CuGraphExec, trans, B)
    scratch = trans.scratch_memory
    function to_spectral!(coeffs, f)
        SpeedyTransforms.forward_loop!(cache,scratch.north,scratch.south,f,trans)
        SpeedyTransforms._legendre!(coeffs,scratch.north,scratch.south,scratch.column,trans)
        return coeffs
    end
    function to_grid!(f, coeffs)
        SpeedyTransforms._legendre!(scratch.north,scratch.south,coeffs,scratch.column,trans; unscale_coslat=false)
        SpeedyTransforms.inverse_loop!(cache,f,scratch.north,scratch.south,trans)
        return f
    end
    function analysis(uv)
        a.data .= uv[1].data ./ coslat
        b.data .= uv[2].data ./ coslat
        to_spectral!(us, a); to_spectral!(vs, b)
        divergence!(d, us, vs, trans; radius=1)
        curl!(z, us, vs, trans; radius=1)
        d.data .*= keep; z.data .*= keep
        return (d,z)
    end
    function synthesis(dz)
        UV_from_vordiv!(us, vs, dz[2], dz[1], trans; radius=1)
        to_grid!(outu, us); to_grid!(outv, vs)
        outu.data ./= coslat; outv.data ./= coslat
        return (outu,outv)
    end
    function scalar(c)
        scalarfield = field(); transform!(scalarfield, c, trans); return scalarfield
    end
    unpack(x) = permutedims(reshape(Array(x.data), nlon, nlat, B), (2,1,3))
    # Native vector order is (east,north); common reference is (south,east).
    unpack_vector(uv) = (-unpack(uv[2]), unpack(uv[1]))
    return (; input=(u,v), analysis, synthesis, scalar, unpack, unpack_vector, ref,
        nlat_actual=nlat,
        notes="Preallocated pipeline composed of native fused FFT loops, Legendre kernels and vector operators; full-pipeline graph capture replaces nested Fourier graphs; one extra workspace degree")
end

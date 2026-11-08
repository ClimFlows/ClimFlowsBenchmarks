include("common.jl")
using Test
@testset "Independent vector fixture" begin
    for theta in (0.2, 0.8, 1.5, 2.5)
        p, dp = legendre_mode(1,0,theta)
        @test p ≈ sqrt(3/(4pi))*cos(theta)
        @test dp ≈ -sqrt(3/(4pi))*sin(theta)
        p, dp = legendre_mode(1,1,theta)
        @test p ≈ -sqrt(3/(8pi))*sin(theta)
        @test dp ≈ -sqrt(3/(8pi))*cos(theta)
    end
    # Finite differences verify metric factors and curl orientation independently.
    function potentials(theta,phi,L,B,seed)
        f=fixture([theta],[phi],L,B,Float64,seed)
        return f.vt[1,1,1],f.vp[1,1,1],f.div[1,1,1],f.vor[1,1,1]
    end
    h=1e-6
    for (theta,phi) in ((0.4,0.3),(1.2,2.0),(2.7,5.0))
        vt,vp,d,z=potentials(theta,phi,21,2,42)
        tm=potentials(theta-h,phi,21,2,42); tp=potentials(theta+h,phi,21,2,42)
        pm=potentials(theta,phi-h,21,2,42); pp=potentials(theta,phi+h,21,2,42)
        div=(sin(theta+h)*tp[1]-sin(theta-h)*tm[1])/(2h*sin(theta))+(pp[2]-pm[2])/(2h*sin(theta))
        vor=(sin(theta+h)*tp[2]-sin(theta-h)*tm[2])/(2h*sin(theta))-(pp[1]-pm[1])/(2h*sin(theta))
        @test div ≈ d rtol=1e-7 atol=1e-8
        @test vor ≈ z rtol=1e-7 atol=1e-8
    end
    c=TOML.parsefile(joinpath(ROOT,"config.toml"))
    @test c["batch"]==64
    @test [x["lmax"] for x in c["cases"]]==[21,42,63,127,255]
end

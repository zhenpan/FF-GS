
using PyCall
using PyPlot
using Dierckx

@pyimport numpy as np

immutable Grid
    a::Array{Float64,2}
    b::Array{Float64,2}
    c::Array{Float64,2}
    d::Array{Float64,2}
    ee::Array{Float64,2}
    S::Array{Float64,2}
    len::Int64
    R::Array{Float64,2}
    μ::Array{Float64,2}
    rmin::Float64
    rmax::Float64
end

function Grid(grid_size, rmin, rmax,Θmin, Θmax)
    Rmin = r2R(rmin)
    Rmax = r2R(rmax)
    μmin = cos(Θmax)
    μmax = cos(Θmin)

    R = linspace(Rmin, Rmax, grid_size)
    μ = linspace(μmin, μmax, grid_size)
    dR  = (Rmax - Rmin)/grid_size
    dμ  = (μmax - μmin)/grid_size
    R,μ = np.meshgrid(R, μ)
    r   = R2r(R)
    S   = -exp(-0.5 .* r) .*(1 - μ.^2).*dμ^2
    len = grid_size

#=#############################################################
  Be careful about index order U[i,j]:
  j is the col index -> r index,  i is the row index -> Θ index
=###############################################################
    Crr = ones(grid_size, grid_size)
    Cr  = 1. ./r

    CRR = Crr .* (1 - R).^4
    CR  = (1 - R).^2 .* (Cr - 2 .* Crr .* (1 - R))
    Cμμ = (1 - R).^2 ./ R.^2 .* (1 - μ.^2)
    Cμ  = (1 - R).^2 ./ R.^2 .* (- μ)

    a = ones(grid_size, grid_size) .* (Cμμ + 0.5 * dμ .*Cμ)
    b = ones(grid_size, grid_size) .* (Cμμ - 0.5 * dμ .*Cμ)
    c = ones(grid_size, grid_size) .* (dμ^2 / dR^2 .* CRR + 0.5*dμ^2/dR .* CR)
    d = ones(grid_size, grid_size) .* (dμ^2 / dR^2 .* CRR - 0.5*dμ^2/dR .* CR)
    ee= ones(grid_size, grid_size) .* (-2.) .* (Cμμ + dμ^2/dR^2 .* CRR)

    grd = Grid(a, b, c, d, ee, S, len, R, μ, rmin, rmax)
    return grd
end

function solver(U, grid; maxitr = 100, omega = 1.0, ϵ = 1.0e-3)
    a = grid.a
    b = grid.b
    c = grid.c
    d = grid.d
    ee= grid.ee
    S = grid.S

    anm_ini= sum(abs(grid.S))
    ρ2_jcb = cos(pi/grid.len)^2
    res = zeros(grid.S)

    for n = 1:maxitr
        for j = 2:grid.len-1            #even run for even j , odd run for odd j
            for l = 2+mod(j, 2):2:grid.len-1
              res[j,l] = a[j,l]*U[j+1,l] + b[j,l]*U[j-1,l] + c[j,l]*U[j,l+1] + d[j,l]*U[j,l-1] + ee[j,l]*U[j,l] - S[j,l]
              U[j,l]  -= omega*res[j,l]/ee[j,l]
            end
        end
        omega = (n==1)? 1./(1.- ρ2_jcb/2.) : 1./(1.-omega*ρ2_jcb/4.)

        #for j = 2:grid.len-1
        for j = grid.len-1:-1:2          #odd run for even j, even run for odd j
            for l = 3-mod(j,2):2:grid.len-1
              res[j,l] = a[j,l]*U[j+1,l] + b[j,l]*U[j-1,l] + c[j,l]*U[j,l+1] + d[j,l]*U[j,l-1] + ee[j,l]*U[j,l] - S[j,l]
              U[j,l]  -= omega*res[j,l]/ee[j,l]
            end
        end
        omega = 1./(1.-omega*ρ2_jcb/4.)

        #update boundary: adiabatic boundary at Θ = 0 and pi , and r=rmin

        #U[1,:] = U[2,:]
        #U[end,:] = U[end-1,:]
        #U[:,1] = U[:,2]


        anm = sum(abs(res))
        println("iter= $n, omega = $omega, anm= $anm")
        if anm < ϵ * anm_ini
          return U, anm, res
        end
    end

    return U, anm, res
end

function r2R(r::Array{Float64,2})
    return r ./(1. + r)
end

function r2R(r::Real)
    return r /(1. + r)
end

function R2r(R::Array{Float64,2})
    return R ./(1. - R)
end

function R2r(R::Real)
    return R /(1. - R)
end

function Rμ2xy(grd, U)
    spl = Spline2D( grd.μ[:,1], grd.R[1,:], U)

    x = linspace(- grd.rmax, grd.rmax, grd.len)
    y = linspace(        0., grd.rmax, grd.len)
    x, y = np.meshgrid(x,y)
    r = sqrt(x .^2 + y .^2)
    Θ = angle(x + y .*im)

    Uxy = zeros(size(r))
    for i = 1:grd.len
      for j = 1:grd.len
        if r[i,j] > grd.rmin
          Uxy[i,j] = evaluate(spl, cos(Θ[i,j]), r2R(r[i,j]))
        end
      end
    end

    return Uxy
end






len = 256
rmin= 2.
rmax= 50.
Θmin= 0.
Θmax= pi

grd = Grid(len, rmin, rmax, Θmin, Θmax)
U   = zeros(len, len)
@time U, anm, res = solver(U, grd, maxitr = 3000, ϵ = 1.0e-5)

subplot(2,2,1)
imshow(U, origin = "lower") #, extent = [r2R(rmin), r2R(rmax), Θmin, Θmax])
title("U")
colorbar()
subplot(2,2,2)
imshow(res, origin = "lower") #, extent = [r2R(rmin), r2R(rmax), Θmin, Θmax])
title("Residual")
colorbar()
subplot(2,2,3)
imshow(-grd.S, origin = "lower") #, extent = [r2R(rmin), r2R(rmax), Θmin, Θmax])
title("Source")
colorbar()

Uxy = Rμ2xy(grd, U)
Resxy = Rμ2xy(grd, res)

figure()
subplot(2,1,1)
imshow(Uxy, origin = "lower", extent = [-rmax, rmax, 0., rmax])
title("U")
colorbar()
subplot(2,1,2)
imshow(Resxy, origin = "lower", extent = [-rmax, rmax, 0., rmax])
title("Residual")
colorbar()

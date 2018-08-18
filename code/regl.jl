
using PyCall, Dierckx

@pyimport numpy as np

immutable Cord
    R::Array{Float64,2}
    μ::Array{Float64,2}
    r::Array{Float64,2}
    Rcol::Array{Float64,1}
    μcol::Array{Float64,1}
    rcol::Array{Float64,1}
    Rlen::Int
    μlen::Int
    δR::Float64
    δμ::Float64
    a::Float64
    Ω_H::Float64
    rmin::Float64
    idx_r2::Int
    idx_xbd::Array{Int64,1}
end

immutable Geom
    r_Σ::Array
    ∂1r_Σ::Array
    ∂2r_Σ::Array
    β_Σ::Array
    ∂1β_Σ::Array
    ∂2β_Σ::Array
    Σ_Δ::Array
    ∂1Σ_Δ::Array
    ∂2Σ_Δ::Array
    s_Δ::Array
    ∂1s_Δ::Array
    ∂2s_Δ::Array
end

immutable Grid
    aa::Array{Float64,2}
    bb::Array{Float64,2}
    cc::Array{Float64,2}
    dd::Array{Float64,2}
    ee::Array{Float64,2}
    ff::Array{Float64,2}
    κ::Array{Float64,2}
end

immutable U_partials
    ∂1U::Array{Float64,2}       #∂1 = ∂r, ∂2 = ∂μ
    ∂2U::Array{Float64,2}
    ∂11U::Array{Float64,2}
    ∂12U::Array{Float64,2}
    ∂22U::Array{Float64,2}
end

immutable Ω_and_I
    Ω::Array{Float64,2}
    ∂1Ω::Array{Float64,2}       #∂1 = ∂r, ∂2 = ∂μ
    ∂2Ω::Array{Float64,2}
    ∂11Ω::Array{Float64,2}
    ∂12Ω::Array{Float64,2}
    ∂22Ω::Array{Float64,2}
    IIp::Array{Float64,2}       #II'
    ∂1IIp::Array{Float64,2}
    ∂2IIp::Array{Float64,2}
    Ωspl::Dierckx.Spline1D
    IIpspl::Dierckx.Spline1D
end


function Cord(; Rlen = 256, μlen = 64, a = 1., rmax = 100., xbd = 4.0)
    rmin = (1. + sqrt(1.-a^2)) * (1-1.e-8)  #slightly inside horizon, avoiding 1/Δ singularity
    Ω_H  = a/(2*(1. + sqrt(1.-a^2)))

    Rmin = r2R(rmin)
    Rmax = r2R(rmax)
    μmin = 0.
    μmax = 1.

    Rcol = collect(linspace(Rmin, Rmax, Rlen))
    μcol = collect(linspace(μmin, μmax, μlen))
    rcol = R2r(Rcol)
    δR  = (Rmax - Rmin)/(Rlen-1)
    δμ  = (μmax - μmin)/(μlen-1)

    idx_r2  = Int( floor((r2R(2.)-Rmin)/δR) + 1)   # index r= 2_mns
    r_xbd   = xbd ./ sqrt(1-μcol.^2 + 1.e-8)
    R_xbd   = r2R(r_xbd)
    idx_xbd = ones(Int, μlen)
    for j = 1:μlen
        Ridx       = Int(floor( (R_xbd[j] - Rmin)/δR + 1. ))
        idx_xbd[j] = min(Ridx, Rlen)                        # points not evolving
    end

    R,μ = np.meshgrid(Rcol, μcol)
    r   = R2r(R)

    crd = Cord(R, μ, r, Rcol, μcol, rcol, Rlen, μlen, δR, δμ, a, Ω_H, rmin, idx_r2, idx_xbd)
    return crd
end

function Geom(r::Array, μ::Array, a::Float64)
    Σ = r.^2 + a^2 * μ.^2
    Δ = r.^2 - 2r + a^2
    β = Δ .* Σ + 2r .*(r.^2+a^2)   # also (r.^2+a^2).*Σ + 2r.* (a^2* sst)

    r_Σ   = r ./Σ
    ∂1r_Σ = 1 ./Σ - 2r.^2 ./ Σ.^2
    ∂2r_Σ = r .* (- 2a^2 * μ ./ Σ.^2)

    sst   = 1 - μ.^2
    β_Σ   = β ./Σ
    ∂1β_Σ = 2r + 2a^2 * sst .* ∂1r_Σ
    ∂2β_Σ = 2r .*(r.^2+a^2) .* (- 2a^2 * μ ./ Σ.^2)

    Σ_Δ   = Σ ./ Δ
    ∂1Σ_Δ = 2r ./ Δ - Σ .* (2r -2) ./ Δ.^2
    ∂2Σ_Δ = 2a^2 .* μ ./ Δ

    s_Δ   =  sst ./ Δ
    ∂1s_Δ = -sst.*(2r-2)./ Δ.^2
    ∂2s_Δ = -2μ ./ Δ

    mtr   = Geom(r_Σ, ∂1r_Σ, ∂2r_Σ, β_Σ, ∂1β_Σ, ∂2β_Σ, Σ_Δ, ∂1Σ_Δ, ∂2Σ_Δ, s_Δ, ∂1s_Δ, ∂2s_Δ)
    return mtr
end


function U_partials(U::Array{Float64,2}, crd::Cord)
    ∂1U = zeros(U)
    ∂1U[:, 2:end-1] = (U[:, 3:end] - U[:, 1:end-2])./ (2 * crd.δR)
    ∂1U[:, end]     = (U[:, end]   - U[:, end-1])  ./ crd.δR
    ∂1U[:, 1]       = (U[:, 2]     - U[:, 1])      ./ crd.δR

    ∂2U = zeros(U)
    ∂2U[2:end-1, :] = (U[3:end, :] - U[1:end-2, :])./ (2 * crd.δμ)
    ∂2U[end, :]     = (U[end, :]   - U[end-1, :])  ./ crd.δμ
    ∂2U[1, :]       = (U[2, :]     - U[1, :])      ./ crd.δμ

    ∂12U = zeros(U)
    ∂12U[2:end-1, :] = (∂1U[3:end, :] - ∂1U[1:end-2, :])./ (2 * crd.δμ)
    ∂12U[end, :]     = (∂1U[end, :]   - ∂1U[end-1, :])  ./ crd.δμ
    ∂12U[1, :]       = (∂1U[2, :]     - ∂1U[1, :])      ./ crd.δμ

    ∂11U = zeros(U)
    ∂11U[:, 2:end-1] = (U[:, 3:end] + U[:, 1:end-2] - 2U[:, 2:end-1])./ (2 * crd.δR)
    ∂11U[:, end]     = (∂1U[:, end]   - ∂1U[:, end-1])  ./ crd.δR
    ∂11U[:, 1]       = (∂1U[:, 2]     - ∂1U[:, 1])      ./ crd.δR

    ∂22U = zeros(U)
    ∂22U[2:end-1, :] = (U[3:end, :] + U[1:end-2, :] - 2U[2:end-1, :])./ (2 * crd.δμ)
    ∂22U[end, :]     = (∂2U[end, :]   - ∂2U[end-1, :])  ./ crd.δμ
    ∂22U[1, :]       = (∂2U[2, :]     - ∂2U[1, :])      ./ crd.δμ

    R    = crd.R
    ∂1U  = ∂1U  .* (1-R).^2                         # ∂1 = ∂r,  ∂2 = ∂μ
    ∂12U = ∂12U .* (1-R).^2
    ∂11U = ∂11U .* (1-R).^4 - 2 * ∂1U .*(1-R).^3

    Upr = U_partials(∂1U, ∂2U, ∂11U, ∂12U, ∂22U)
    return Upr
end

function Ω_and_I(U::Array{Float64,2}, crd::Cord, Ωspl::Dierckx.Spline1D, IIpspl::Dierckx.Spline1D)
    Ω   = evaluate(Ωspl, reshape(U,  length(U)) )
    dΩ  = derivative(Ωspl, reshape(U, length(U)), nu = 1 )
    ddΩ = derivative(Ωspl, reshape(U, length(U)), nu = 2 )

    IIp = evaluate(IIpspl, reshape(U, length(U) ))
    dIIp= derivative(IIpspl, reshape(U, length(U) ))

    Ω   = reshape(Ω, size(U))
    dΩ  = reshape(dΩ, size(U))
    ddΩ = reshape(ddΩ, size(U))
    IIp = reshape(IIp, size(U))
    dIIp= reshape(dIIp, size(U))

    Upr  = U_partials(U, crd)
    ∂1Ω  = dΩ .* Upr.∂1U
    ∂2Ω  = dΩ .* Upr.∂2U
    ∂11Ω = ddΩ .* Upr.∂1U.^2         + dΩ .* Upr.∂11U
    ∂12Ω = ddΩ .* Upr.∂1U .* Upr.∂2U + dΩ .* Upr.∂12U
    ∂22Ω = ddΩ .* Upr.∂2U.^2         + dΩ .* Upr.∂22U

    ∂1IIp = dIIp .* Upr.∂1U
    ∂2IIp = dIIp .* Upr.∂2U
    return Ω_and_I(Ω, ∂1Ω, ∂2Ω, ∂11Ω, ∂12Ω, ∂22Ω, IIp, ∂1IIp, ∂2IIp, Ωspl, IIpspl)
end


function Ω_and_I!(U::Array{Float64,2}, crd::Cord, Ω_I::Ω_and_I, IIpspl::Dierckx.Spline1D)
    Ω   = Ω_I.Ω
    ∂1Ω = Ω_I.∂1Ω
    ∂2Ω = Ω_I.∂2Ω
    ∂11Ω = Ω_I.∂11Ω
    ∂12Ω = Ω_I.∂12Ω
    ∂22Ω = Ω_I.∂22Ω
    Ωspl = Ω_I.Ωspl

    IIp  = evaluate(IIpspl, reshape(U, length(U) ))
    dIIp = derivative(IIpspl, reshape(U, length(U) ))
    IIp  = reshape(IIp, size(U))
    dIIp = reshape(dIIp, size(U))

    Upr   = U_partials(U, crd)
    ∂1IIp = dIIp .* Upr.∂1U
    ∂2IIp = dIIp .* Upr.∂2U

    return Ω_and_I(Ω, ∂1Ω, ∂2Ω, ∂11Ω, ∂12Ω, ∂22Ω, IIp, ∂1IIp, ∂2IIp, Ωspl, IIpspl)
end

function Grid(crd::Cord, mtr::Geom, Ω_I::Ω_and_I)
    R, μ,  δR, δμ, r, a  = crd.R, crd.μ, crd.δR, crd.δμ, crd.r, crd.a
    r_Σ,  ∂1r_Σ,  ∂2r_Σ  = mtr.r_Σ, mtr.∂1r_Σ,  mtr.∂2r_Σ
    β_Σ,  ∂1β_Σ,  ∂2β_Σ  = mtr.β_Σ, mtr.∂1β_Σ,  mtr.∂2β_Σ
    Σ_Δ,  s_Δ            = mtr.Σ_Δ, mtr.s_Δ

    Ω, ∂1Ω, ∂2Ω, IIp = Ω_I.Ω, Ω_I.∂1Ω, Ω_I.∂2Ω, Ω_I.IIp

    sst = 1-μ.^2
    κ   = (β_Σ   .* Ω.^2 - 4a .*   r_Σ  .* Ω) .* sst - (1- 2*r_Σ)
    ∂1κ = (∂1β_Σ .* Ω.^2 - 4a .* ∂1r_Σ  .* Ω) .* sst +  2*∂1r_Σ
    ∂2κ = (∂2β_Σ .* Ω.^2 - 4a .* ∂2r_Σ  .* Ω) .* sst +  2*∂2r_Σ +(β_Σ .* Ω.^2 - 4a .* r_Σ .* Ω) .*(-2μ)
    ∂Ωκ = 2*(β_Σ .* Ω - 2a .* r_Σ) .* sst

    Crr = κ
    Cμμ = κ .* s_Δ
  	Cr  =  ∂1κ + 0.5*∂Ωκ .* ∂1Ω
    Cμ  = (∂2κ + 0.5*∂Ωκ .* ∂2Ω) .* s_Δ
    CRR = Crr .* (1 - R).^4
    CR  = Cr  .* (1 - R).^2 - 2Crr .* (1 - R).^3

    δ   = min(δR, δμ)
    S   = Σ_Δ .* IIp

    #=#############################################################
      Be careful about index order U[i,j]:
      j is the col index -> r index,  i is the row index -> μ index
    =###############################################################
    aa = (Cμμ ./ δμ^2 + 0.5 *Cμ ./ δμ) .* δ^2
    bb = (Cμμ ./ δμ^2 - 0.5 *Cμ ./ δμ) .* δ^2
    cc = (CRR ./ δR^2 + 0.5 *CR ./ δR) .* δ^2
    dd = (CRR ./ δR^2 - 0.5 *CR ./ δR) .* δ^2
    ee = -2*(Cμμ ./ δμ^2 + CRR ./ δR^2) .* δ^2
    ff =  S .* δ^2

    # ee_rgl = ee + sign(ee) * (2.e-4) .* tanh(8./(crd.r - 0.999)).^4

    grd = Grid(aa, bb, cc, dd, ee, ff, κ)
    return grd
end

function Grid!(grd::Grid, crd::Cord, mtr::Geom, Ω_I::Ω_and_I)
    aa = grd.aa
    bb = grd.bb
    cc = grd.cc
    dd = grd.dd
    ee = grd.ee
    κ  = grd.κ

    S  = (mtr.Σ_Δ) .* (Ω_I.IIp)
    δ  = min(crd.δR, crd.δμ)
    ff =  S .* δ^2

    grd = Grid(aa, bb, cc, dd, ee, ff, κ)
    return grd
end

function r2R(r::Array)
    return r ./(1. + r)
end

function r2R(r::Real)
    return r /(1. + r)
end

function R2r(R::Array)
    return R ./(1. - R)
end

function R2r(R::Real)
    return R /(1. - R)
end

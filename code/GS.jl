

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
    r_Σ::Array{Float64,2}
    ∂1r_Σ::Array{Float64,2}
    ∂2r_Σ::Array{Float64,2}
    β_Σ::Array{Float64,2}
    ∂1β_Σ::Array{Float64,2}
    ∂2β_Σ::Array{Float64,2}
    Σ_Δ::Array{Float64,2}
end

immutable Grid
    aa::Array{Float64,2}
    bb::Array{Float64,2}
    cc::Array{Float64,2}
    dd::Array{Float64,2}
    ee::Array{Float64,2}
    ee_rgl::Array{Float64,2}
    ff::Array{Float64,2}
    S::Array{Float64,2}
    κ::Array{Float64,2}
    ∂1κ::Array{Float64,2}
    ∂2κ::Array{Float64,2}
    Cr::Array{Float64,2}
    Cμ::Array{Float64,2}
    Cμ_esn::Array{Float64,2}
end

immutable Ω_and_I
    Ω::Array{Float64,2}
    ∂1Ω::Array{Float64,2}       #∂1 = ∂r, ∂2 = ∂μ
    ∂2Ω::Array{Float64,2}
    IIp::Array{Float64,2}       #II'
    Ωspl::Dierckx.Spline1D
    IIpspl::Dierckx.Spline1D
end


immutable LS_neighbors
    lsn_idx::Array{Int, 2}     # (Ridx_lhs, μidx)
    lsn_map::Array{Int, 2}
    ILS_lt::Array{Float64, 2}  # (Rlt, μlt, xlt)
    ILS_rt::Array{Float64, 2}  # (Rrt, μrt, xrt)
end

immutable LS
    Loc::Array{Float64, 2}
    Σ_Δ::Array{Float64, 1}
    ULS::Array{Float64, 1}
    IIp::Array{Float64, 1}
    S::Array{Float64, 1}
    Cr::Array{Float64, 1}
    Cμ::Array{Float64, 1}
    Cμ_esn::Array{Float64, 1}
end

function Cord(; Rlen = 512, μlen = 64, a = 1., rmax = 100.,  xbd = 4.0)
    rmin = (1. + sqrt(1.-a^2)) * (1-1.e-8)  #slightly inside horizon, avoiding 1/Δ singularity
    Ω_H  = a/(2*rmin)

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
        idx_xbd[j] =  min( Ridx, Rlen)                        # points not evolving
    end

    R,μ = np.meshgrid(Rcol, μcol)
    r   = R2r(R)

    crd = Cord(R, μ, r, Rcol, μcol, rcol, Rlen, μlen, δR, δμ, a, Ω_H, rmin, idx_r2, idx_xbd)
    return crd
end

function Geom(crd::Cord)
    r, a, μ = crd.r, crd.a, crd.μ

    Σ = r.^2 + a^2 .* μ.^2
    Δ = r.^2 - 2r + a^2
    β = Δ .* Σ + 2r .*(r.^2+a^2)   # also (r^2+a^2)Σ + 2r a^2 sst

    r_Σ   = r ./Σ
    ∂1r_Σ = 1 ./Σ - 2r.^2 ./ Σ.^2
    ∂2r_Σ = r .* (- 2a^2 .* μ ./ Σ.^2)

    sst   = 1 - μ.^2
    β_Σ   = β ./Σ
    ∂1β_Σ = 2r + 2a^2 .*sst .* ∂1r_Σ
    ∂2β_Σ = 2r .*(r.^2+a^2) .* (- 2a^2 .* μ ./ Σ.^2)

    Σ_Δ   = Σ ./ Δ

    mtr   = Geom(r_Σ, ∂1r_Σ, ∂2r_Σ, β_Σ, ∂1β_Σ, ∂2β_Σ, Σ_Δ)
    return mtr
end

function Ω_and_I(U::Array{Float64,2}, crd::Cord, Ωspl::Dierckx.Spline1D, IIpspl::Dierckx.Spline1D)
    Ω   = evaluate(Ωspl, reshape(U,  length(U)) )
    IIp = evaluate(IIpspl, reshape(U, length(U) ))
    Ω   = reshape(Ω, size(U))
    IIp = reshape(IIp, size(U))

    ∂1Ω = zeros(Ω)
    ∂1Ω[:, 2:end-1] = (Ω[:, 3:end] - Ω[:, 1:end-2])./ (2 * crd.δR)
    ∂1Ω[:, end]     = (Ω[:, end]   - Ω[:, end-1])  ./ crd.δR
    ∂1Ω[:, 1]       = (Ω[:, 2]     - Ω[:, 1])      ./ crd.δR

    ∂2Ω = zeros(Ω)
    ∂2Ω[2:end-1, :] = (Ω[3:end, :] - Ω[1:end-2, :])./ (2 * crd.δμ)
    ∂2Ω[end, :]     = (Ω[end, :]   - Ω[end-1, :])  ./ crd.δμ
    ∂2Ω[1, :]       = (Ω[2, :]     - Ω[1, :])      ./ crd.δμ

    t1 = deepcopy(∂1Ω)
    t2 = deepcopy(∂2Ω)

    for j = 2:crd.μlen - 1
        for l = 2:crd.Rlen - 1
            ∂1Ω[j,l] = 0.5*t1[j,l] + 0.5*(t1[j+1,l] + t1[j-1, l] + t1[j,l+1] + t1[j,l-1])/4
            ∂2Ω[j,l] = 0.5*t2[j,l] + 0.5*(t2[j+1,l] + t2[j-1, l] + t2[j,l+1] + t2[j,l-1])/4
        end
    end

    ∂1Ω  = ∂1Ω  .* (1-crd.R).^2                         # ∂1 = ∂r,  ∂2 = ∂μ
    return Ω_and_I(Ω, ∂1Ω, ∂2Ω, IIp, Ωspl, IIpspl)
end


function Ω_and_I!(U::Array{Float64,2}, Ω_I::Ω_and_I, IIpspl::Dierckx.Spline1D)
    Ω   = Ω_I.Ω
    ∂1Ω = Ω_I.∂1Ω
    ∂2Ω = Ω_I.∂2Ω
    Ωspl= Ω_I.Ωspl
    IIp = evaluate(IIpspl, reshape(U, length(U) ))
    IIp = reshape(IIp, size(U))
    return Ω_and_I(Ω, ∂1Ω, ∂2Ω, IIp, Ωspl, IIpspl)
end

function Grid(crd::Cord, mtr::Geom, Ω_I::Ω_and_I)
    R, μ,  δR, δμ,  r, a     = crd.R, crd.μ, crd.δR, crd.δμ, crd.r, crd.a
    r_Σ,  ∂1r_Σ,  ∂2r_Σ      = mtr.r_Σ,    mtr.∂1r_Σ,  mtr.∂2r_Σ
    β_Σ,  ∂1β_Σ,  ∂2β_Σ, Σ_Δ = mtr.β_Σ,    mtr.∂1β_Σ,  mtr.∂2β_Σ, mtr.Σ_Δ

    Ω, ∂1Ω, ∂2Ω, IIp = Ω_I.Ω, Ω_I.∂1Ω, Ω_I.∂2Ω, Ω_I.IIp

    sst = 1-μ.^2
    Δ   = r.^2 -2r + a^2

    κ   = (β_Σ   .* Ω.^2 - 4a .*   r_Σ  .* Ω) .* sst - (1- 2*r_Σ)
    ∂1κ = (∂1β_Σ .* Ω.^2 - 4a .* ∂1r_Σ  .* Ω) .* sst +  2*∂1r_Σ
    ∂2κ = (∂2β_Σ .* Ω.^2 - 4a .* ∂2r_Σ  .* Ω) .* sst +  2*∂2r_Σ +(β_Σ   .* Ω.^2 - 4a .*   r_Σ  .* Ω) .*(-2μ)

    ∂Ωκ_hf = (β_Σ .* Ω - 2a .* r_Σ) .* sst

    Crr = κ
  	Cr  = ∂1κ + ∂Ωκ_hf .* ∂1Ω
    Cμμ = κ .* (sst./Δ)
    Cμ  = (∂2κ + ∂Ωκ_hf .* ∂2Ω) .* (sst./Δ);     Cμ_esn = ∂2κ + ∂Ωκ_hf .* ∂2Ω

    δ   = min(δR, δμ)
    S   = Σ_Δ .*  IIp

    #=#############################################################
      Be careful about index order U[i,j]:
      j is the col index -> r index,  i is the row index -> μ index
    =###############################################################

    CRR = Crr .* (1 - R).^4
    CR  = (1 - R).^2 .* (Cr - 2 .* Crr .* (1 - R))

    aa = (Cμμ ./ δμ^2 + 0.5 .*Cμ ./ δμ) .* δ^2
    bb = (Cμμ ./ δμ^2 - 0.5 .*Cμ ./ δμ) .* δ^2
    cc = (CRR ./ δR^2 + 0.5 .*CR ./ δR) .* δ^2
    dd = (CRR ./ δR^2 - 0.5 .*CR ./ δR) .* δ^2
    ee = -2*(Cμμ ./ δμ^2 + CRR ./ δR^2) .* δ^2
    ff =  S .* δ^2

    ee_rgl = ee + sign(ee) * (1.e-3) .* tanh(8./(crd.r - 0.999)).^4
    # lft = κ .> 0.
    # rgt = κ .< 0.
    #
    # ee_rgl = zeros(ee)
    # ee_rgl[lft] = ee[lft] + sign(ee[lft]) * (1.e-3)
    # ee_rgl[rgt] = ee[rgt] + sign(ee[rgt]) * (2.e-4) .* tanh(8./(crd.r[rgt] - 0.999)).^4


    grd = Grid(aa, bb, cc, dd, ee, ee_rgl, ff, S, κ, ∂1κ, ∂2κ, Cr, Cμ, Cμ_esn)
    return grd
end

function Grid!(grd::Grid, crd::Cord, mtr::Geom, Ω_I::Ω_and_I)
    aa = grd.aa
    bb = grd.bb
    cc = grd.cc
    dd = grd.dd
    ee = grd.ee
    ee_rgl = grd.ee_rgl

    κ   = grd.κ
    ∂1κ = grd.∂1κ
    ∂2κ = grd.∂2κ
    Cr  = grd.Cr
    Cμ  = grd.Cμ; Cμ_esn = grd.Cμ_esn

    S  = (mtr.Σ_Δ) .* (Ω_I.IIp)
    δ  = min(crd.δR, crd.δμ)
    ff =  S .* δ^2

    grd = Grid(aa, bb, cc, dd, ee, ee_rgl, ff, S, κ, ∂1κ, ∂2κ, Cr, Cμ, Cμ_esn)
    return grd
end


function LS(U::Array{Float64,2}, grd::Grid, crd::Cord, Ω_I::Ω_and_I)
    idx_r2 = crd.idx_r2 + 1
    RILS   = zeros(crd.μlen)
    μILS   = crd.μcol
    Cr     = zeros(crd.μlen)
    Cμ     = zeros(crd.μlen)
    Cμ_esn = zeros(crd.μlen)

    for μidx = 1: length(RILS)
        κcol = reshape(-grd.κ[μidx, 1:idx_r2], idx_r2)
        Rcol = reshape(crd.R[μidx, 1:idx_r2], idx_r2)
        spl_in  = Spline1D(κcol, Rcol, k = 2)
        RILS[μidx] = spl_in(0.)

        Crcol = reshape(grd.Cr[μidx, 1:idx_r2], idx_r2)
        Cμcol = reshape(grd.Cμ_esn[μidx, 1:idx_r2], idx_r2)
        spl_Cr      = Spline1D(Rcol, Crcol, k = 2)
        spl_Cμesn   = Spline1D(Rcol, Cμcol, k = 2)
        Cr[μidx]     = spl_Cr(RILS[μidx])
        Cμ_esn[μidx] = spl_Cμesn(RILS[μidx])
    end

    Uspl = Spline2D( crd.μcol, crd.Rcol, U, kx =1, ky=1 )
    UILS = evaluate( Uspl, μILS, RILS )
    rILS = R2r(RILS)
    Σ_Δ  = (rILS.^2 + crd.a^2 * μILS.^2)./(rILS.^2 - 2rILS + crd.a^2)
    IIp  = Ω_I.IIpspl(UILS)
    S    = Σ_Δ .* IIp
    Cμ   = Cμ_esn .* (1.-μILS.^2)./(rILS.^2 - 2rILS + crd.a^2)

    ILS_loc = hcat(RILS, μILS)
    return LS(ILS_loc, Σ_Δ, UILS, IIp, S, Cr, Cμ, Cμ_esn)
end

#=#########################################
        update IIp related part only
=##########################################

function LS!(ils::LS, Uils::Array{Float64,1}, IIpspl::Dierckx.Spline1D)
    Loc   = ils.Loc
    Σ_Δ   = ils.Σ_Δ
    UILS  = Uils
    IIp   = IIpspl(UILS)
    S     = Σ_Δ .* IIp
    Cr    = ils.Cr
    Cμ    = ils.Cμ
    Cμ_esn= ils.Cμ_esn
    return LS(Loc, Σ_Δ, UILS, IIp, S, Cr, Cμ, Cμ_esn)
end

function LS_neighbors(U::Array{Float64,2}, ils::LS, grd::Grid, crd::Cord)
        μlen = crd.μlen
        Rlen = crd.Rlen
        RILS = ils.Loc[:,1]

        lsn_idx = zeros(Int, μlen, 2)
        lsn_map = zeros(Int, μlen, Rlen)
        ILS_lt  = zeros(μlen, 2)          # R and μ
        ILS_rt  = zeros(μlen, 2)

    for μidx = 1:μlen
        Ridx = Int( floor( (RILS[μidx] - crd.R[1,1])/crd.δR + 1) )
        lsn_idx[μidx,:] = [Ridx, μidx]

        Rlow = max(Ridx-1, 1)
        Rhgh = min(Ridx+2, Rlen)
        lsn_map[μidx, Rlow:Rhgh] = 1
    end

    ILS_lt, ILS_rt = Sngl_helper(grd, crd, ils)
    lsn = LS_neighbors(lsn_idx, lsn_map, ILS_lt, ILS_rt)
    return lsn
end

function Sngl_helper(grd::Grid, crd::Cord, ils::LS; ϵ = 2.)
    RILS  = ils.Loc[:,1]
    μILS  = ils.Loc[:,2]
    norm1 = ils.Cr.* (1 - RILS).^2
    norm2 = ils.Cμ

    hp1 = norm1 ./ crd.δR
    hp2 = norm2 ./ crd.δμ; hp2[end] = hp2[end-1]
    hp3 = sqrt(hp1.^2 + hp2.^2)

    ϵmax1 = (μILS-1.)./(crd.δμ * hp2 ./hp3)           #avoid μrt > 1 or μlt < 0
    ϵmax2 = (-μILS)./(crd.δμ * hp2 ./hp3)
    ϵsaf  = min(ϵmax1, ϵmax2, ϵ)

    dR  = ϵsaf.*crd.δR .* hp1 ./hp3                   #dR negative
    dμ  = ϵsaf.*crd.δμ .* hp2 ./hp3                   #dμ negative

    Rlt = RILS + dR           #iw point wants smaller R and smaller μ
    μlt = μILS + dμ
    xlt = -ϵsaf ./hp3         #Umns = Ult+ xlt * S

    Rrt = RILS - dR           #iw point wants larger R and larger μ
    μrt = μILS - dμ
    xrt = ϵsaf ./hp3          #Upls = Urt+ xrt * S  

    return hcat(Rlt, μlt, xlt), hcat(Rrt, μrt, xrt)
end



#=#########################################################################

find the right pair of points perpendicularly crossing LS

=##########################################################################

function Sngl_helper1(grd::Grid, crd::Cord, ils::LS, lsn_Ridx, lsn_μidx; opt = :ILS_lt, ϵ = 1.)
    R     = zeros(crd.μlen)
    μ     = zeros(crd.μlen)
    norm1 = zeros(crd.μlen)
    norm2 = zeros(crd.μlen)

    for i = 1:crd.μlen
        Ridx = lsn_Ridx[i]
        μidx = lsn_μidx[i]
        R[i] = crd.R[μidx, Ridx]
        μ[i] = crd.μ[μidx, Ridx]

        norm1[i] = grd.Cr[μidx, Ridx] .* (1 - R[i]).^2
        norm2[i] = grd.Cμ[μidx, Ridx]
    end

    hp1 = norm1 ./ crd.δR
    hp2 = norm2 ./ crd.δμ; hp2[end] = hp2[end-1]
    hp3 = sqrt(hp1.^2 + hp2.^2)
    dR  = ϵ*crd.δR * hp1 ./hp3                      #dR negative
    dμ  = ϵ*crd.δμ * hp2 ./hp3                      #dμ negative
    Ωspl= Spline2D(crd.μcol, crd.Rcol, Ω_I.Ω)

    if opt == :ILS_lt
        Ron, μon, ϵon = Perp_solver(R, μ, -dR, -dμ, crd, ils, Ωspl)    # pair points on LS, wants larger R and μ
        Rintr = R + dR                                              #iw wants smaller R and smaller μ
        μintr = μ + dμ
        xintr = - ϵ ./hp3                                 #Uproj = Uintr + xintr * S
        xon   = - (ϵ + ϵon) ./ hp3                        #Uon   = Uintr + xon * S
    elseif opt == :ILS_rt
        Ron, μon, ϵon  = Perp_solver(R, μ, dR, dμ, crd, ils, Ωspl)
        Rintr = R - dR
        μintr = μ - dμ
        xintr = ϵ ./hp3
        xon   = (ϵ + ϵon) ./ hp3
    else
        println("Wrong Option: Sngl_helper1")
        return
    end
    return hcat(Rintr, μintr, xintr), hcat(Ron, μon, xon)
end

function Perp_solver(R::Array, μ::Array, dR::Array, dμ::Array, crd::Cord, ils::LS, Ωspl::Dierckx.Spline2D)
    ϵon  = zeros(μ)
    ϵ    = collect(linspace(0., 1., 16))
    for i = 1:length(μ)
        Rsamp = R[i] + ϵ.*dR[i]
        μsamp = μ[i] + ϵ.*dμ[i]

        κ  = κ_solver(Rsamp, μsamp, crd, Ωspl)
        if κ[end] > κ[1]
            spl = Spline1D(κ,  ϵ)
        else
            spl = Spline1D(-κ, ϵ)
        end
        ϵon[i]  = spl(0.)
    end

    Ron = R + ϵon.*dR
    μon = μ + ϵon.*dμ
    return Ron, μon, ϵon
end



function κ_solver(R::Array, μ::Array, crd::Cord, Ωspl::Dierckx.Spline2D)
    r = R2r(R)
    a = crd.a

    Σ = r.^2 + a^2 .* μ.^2
    Δ = r.^2 - 2r + a^2
    β = Δ .* Σ + 2r .* (r.^2+a^2)

    β_Σ = β ./ Σ
    r_Σ = r ./ Σ
    sst = 1-μ.^2

    Ω   = evaluate(Ωspl, μ, R)
    κ   = (β_Σ .* Ω.^2 - 4a .* r_Σ .* Ω) .* sst - (1- 2*r_Σ)
    return κ
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

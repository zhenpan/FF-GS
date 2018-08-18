using Dierckx

immutable LS
    μILS::Array{Float64, 1}
    Rspl::Dierckx.Spline1D
    μspl::Dierckx.Spline1D
    Uspl::Dierckx.Spline1D
    IIpspl::Dierckx.Spline1D
    CRspl::Dierckx.Spline1D
    Crspl::Dierckx.Spline1D
    Cμspl::Dierckx.Spline1D
end

immutable Ω_I_LS
    Ω::Array{Float64,1}
    ∂1Ω::Array{Float64,1}       #∂1 = ∂r, ∂2 = ∂μ
    ∂2Ω::Array{Float64,1}
    ∂11Ω::Array{Float64,1}
    ∂12Ω::Array{Float64,1}
    ∂22Ω::Array{Float64,1}
    IIp::Array{Float64,1}       #II'
    ∂1IIp::Array{Float64,1}
    ∂2IIp::Array{Float64,1}
end

immutable LS_neighbors
    lsn_idx::Array{Int, 2}     # (Ridx_lhs, μidx)
    lsn_map::Array{Int, 2}
    ILS_lt::Array{Float64, 2}  # (R, μ)
    ILS_rt::Array{Float64, 2}
end

immutable LS_climate
    Rlsc::Array{Float64,3}
    μlsc::Array{Float64,3}
    κsign::Array{Int,3}
    Rintr::Array{Float64,3}
    μintr::Array{Float64,3}
    xintr::Array{Float64,3}
    μon::Array{Float64,3}
end


immutable κ_partials
    κ::Array{Float64,1}
    ∂1κ::Array{Float64,1}
    ∂2κ::Array{Float64,1}
    ∂Ωκ::Array{Float64,1}
    ∂11κ::Array{Float64,1}
    ∂12κ::Array{Float64,1}
    ∂22κ::Array{Float64,1}
    ∂1Ωκ::Array{Float64,1}
    ∂2Ωκ::Array{Float64,1}
end

immutable Grid_LS
    aa::Array{Float64,1}
    bb::Array{Float64,1}
    cc::Array{Float64,1}
    dd::Array{Float64,1}
    ee::Array{Float64,1}
    ff::Array{Float64,1}
    ww::Array{Float64,1}
end


function LS(U::Array{Float64,2}, grd::Grid, crd::Cord, Ω_I::Ω_and_I)
    idx_r2 = crd.idx_r2 + 1
    RILS   = zeros(crd.μlen)
    μILS   = crd.μcol

    for μidx = 1: length(RILS)
        κcol       = reshape(grd.κ[μidx, 1:idx_r2], idx_r2)
        Rcol       = crd.Rcol[1:idx_r2]
        spl_in     = Spline1D(-κcol, Rcol, k = 2)
        RILS[μidx] = spl_in(0.)
    end

    Uspl = Spline2D( crd.μcol, crd.Rcol, U, kx =1, ky=1 )
    UILS = evaluate( Uspl, μILS, RILS )
    IIp  = Ω_I.IIpspl(UILS)
    Cr, Cμ = Crμ_ls(RILS, μILS, crd, Ω_I)

    Rspl   = Spline1D(μILS, RILS, bc = "extrapolate")
    μspl   = Spline1D(reverse(RILS), reverse(μILS), bc = "extrapolate")
    Uspl   = Spline1D(μILS, UILS)
    IIpspl = Spline1D(μILS, IIp)
    CRspl  = Spline1D(μILS, Cr.*(1-RILS).^2, bc = "extrapolate")
    Crspl  = Spline1D(μILS, Cr, bc = "extrapolate")
    Cμspl  = Spline1D(μILS, Cμ, bc = "extrapolate")

    return LS(μILS, Rspl, μspl, Uspl, IIpspl, CRspl, Crspl, Cμspl)
end

function LS!(ils::LS, Uils, IIpspl)
    μILS = ils.μILS
    Rspl = ils.Rspl
    μspl = ils.μspl
    Uspl = Spline1D(μILS, Uils)
    CRspl= ils.CRspl
    Crspl= ils.Crspl
    Cμspl= ils.Cμspl
    return LS(μILS, Rspl, μspl, Uspl, IIpspl, CRspl, Crspl, Cμspl)
end

function LS_neighbors(U::Array{Float64,2}, ils::LS, grd::Grid, crd::Cord)
        μlen = crd.μlen
        Rlen = crd.Rlen
        RILS = ils.Rspl(ils.μILS)

        lsn_idx = zeros(Int, μlen, 2)
        lsn_map = zeros(Int, μlen, Rlen)
        ILS_lt  = zeros(μlen, 2)          # R and μ
        ILS_rt  = zeros(μlen, 2)

    for μidx = 1:μlen
        Ridx = Int( floor( (RILS[μidx] - crd.R[1,1])/crd.δR + 1) )
        lsn_idx[μidx,    :]   = [Ridx, μidx]
        lsn_map[μidx, Ridx]   = 1
        lsn_map[μidx, Ridx+1] = -1

        ILS_lt[μidx, 1] = crd.R[μidx, Ridx]
        ILS_lt[μidx, 2] = crd.μ[μidx, Ridx]
        ILS_rt[μidx, 1] = crd.R[μidx, Ridx+1]
        ILS_rt[μidx, 2] = crd.μ[μidx, Ridx+1]
    end

    lsn = LS_neighbors(lsn_idx, lsn_map, ILS_lt, ILS_rt)
    return lsn
end

function LS_climate(U::Array{Float64,2}, crd::Cord, ils::LS)
    μILS = ils.μILS
    RILS = ils.Rspl(μILS)

    Rlsc = zeros(3,3, crd.μlen)
    μlsc = zeros(3,3, crd.μlen)

    Rlsc[1,1,:] = RILS-crd.δR; Rlsc[1,2,:] = RILS;  Rlsc[1, 3,:] = RILS+crd.δR
    Rlsc[2,1,:] = RILS-crd.δR; Rlsc[2,2,:] = RILS;  Rlsc[2, 3,:] = RILS+crd.δR
    Rlsc[3,1,:] = RILS-crd.δR; Rlsc[3,2,:] = RILS;  Rlsc[3, 3,:] = RILS+crd.δR

    μlsc[1,1,:] = μILS-crd.δμ; μlsc[2,1,:] = μILS;  μlsc[3, 1,:] = μILS+crd.δμ
    μlsc[1,2,:] = μILS-crd.δμ; μlsc[2,2,:] = μILS;  μlsc[3, 2,:] = μILS+crd.δμ
    μlsc[1,3,:] = μILS-crd.δμ; μlsc[2,3,:] = μILS;  μlsc[3, 3,:] = μILS+crd.δμ

    Ulsc = zeros(3,3, crd.μlen)
    Uspl = Spline2D(crd.μcol, crd.Rcol, U, kx = 1, ky = 1)
    for j = 1:3
        for l = 1:3
            μcol = reshape(μlsc[j,l,:], crd.μlen)
            Rcol = reshape(Rlsc[j,l,:], crd.μlen)
            Ulsc[j,l, :] = Uspl(μcol, Rcol)
        end
    end

    κsign = zeros(Int, (3,3, crd.μlen))
    κsign[1,1,:] = 1; κsign[1,2,:] = 1
    κsign[2,1,:] = 1; κsign[2,2,:] = 0; κsign[2,3,:] = -1
                      κsign[3,2,:] =-1; κsign[3,3,:] = -1

    for l = 2:crd.μlen
        κsign[1,3,l] = (Rlsc[1,3,l] > RILS[l-1])? -1:1
    end
    κsign[1,3,1] = -1

    for l = 1:crd.μlen-1
        κsign[3,1,l] = (Rlsc[3,1,l] > RILS[l+1])? -1:1
    end
    κsign[3,1,end] = -1

    Rintr = zeros(3,3, crd.μlen)
    μintr = zeros(3,3, crd.μlen)
    xintr = zeros(3,3, crd.μlen)
    μon   = zeros(3,3, crd.μlen)

    Rintr[2, 2,:] = ils.Rspl(ils.μILS)
    μintr[2, 2,:] = ils.μILS
    xintr[2, 2,:] = 0.
    μon[2, 2,:]   = ils.μILS

    for j = 1:3
        for l = 1:3
            if !((j == 2) & (l == 2))
                 Rintr[j,l,:], μintr[j,l,:], xintr[j,l,:], μon[j,l,:] = Pair_solver(Rlsc[j,l,:], μlsc[j,l,:], crd, ils)
            end
        end
    end

    return LS_climate(Rlsc, μlsc, κsign, Rintr, μintr, xintr, μon)
end


function Crμ_ls(R::Array, μ::Array, crd::Cord, Ω_I::Ω_and_I)
    r = R2r(R)
    a = crd.a

    mtr_ls = Geom(r, μ, a)
    r_Σ    = mtr_ls.r_Σ
    ∂1r_Σ  = mtr_ls.∂1r_Σ
    ∂2r_Σ  = mtr_ls.∂2r_Σ

    β_Σ    = mtr_ls.β_Σ
    ∂1β_Σ  = mtr_ls.∂1β_Σ
    ∂2β_Σ  = mtr_ls.∂2β_Σ

    sst = 1-μ.^2
    s_Δ = mtr_ls.s_Δ
    s_Δ[end] = Lmt(crd)

    Ωspl   = Spline2D(crd.μcol, crd.Rcol, Ω_I.Ω,   kx = 1, ky = 1)
    ∂1Ωspl = Spline2D(crd.μcol, crd.Rcol, Ω_I.∂1Ω, kx = 1, ky = 1)
    ∂2Ωspl = Spline2D(crd.μcol, crd.Rcol, Ω_I.∂2Ω, kx = 1, ky = 1)
    Ω      = evaluate(  Ωspl, μ, R)
    ∂1Ω    = evaluate(∂1Ωspl, μ, R)
    ∂2Ω    = evaluate(∂1Ωspl, μ, R)

    κ   = (β_Σ   .* Ω.^2 - 4a .*   r_Σ .* Ω) .* sst - (1- 2*r_Σ)
    ∂1κ = (∂1β_Σ .* Ω.^2 - 4a .* ∂1r_Σ .* Ω) .* sst +  2*∂1r_Σ
    ∂2κ = (∂2β_Σ .* Ω.^2 - 4a .* ∂2r_Σ .* Ω) .* sst +  2*∂2r_Σ +(β_Σ .* Ω.^2 - 4a .*r_Σ .* Ω) .*(-2μ)
    ∂Ωκ = 2*(β_Σ .* Ω - 2a .* r_Σ) .* sst

    Cr =  ∂1κ +  0.5*∂Ωκ.* ∂1Ω
    Cμ = (∂2κ +  0.5*∂Ωκ.* ∂2Ω) .* s_Δ
    return Cr, Cμ
end

function Lmt(crd::Cord)
    sst = 1.e-4; μ = sqrt(1-sst)

    a = crd.a; rmin = crd.rmin; rmax = 1.+ sqrt(1-a^2 * μ^2)
    r = linspace(rmin, rmax, 16)

    Σ = r.^2 + a^2 .* μ.^2
    Δ = r.^2 - 2r + a^2
    β = Δ .* Σ + 2r .* (r.^2+a^2)

    β_Σ = β ./ Σ
    r_Σ = r ./ Σ
    Ω   = 0.5*crd.Ω_H
    κ   = (β_Σ .* Ω.^2 - 4a .* r_Σ .* Ω) .* sst - (1- 2*r_Σ)
    κspl = Spline1D(-κ, Δ)
    Δon  = κspl(0.)
    return sst/Δon
end

# plot(ils.Rspl(ils.μILS), ils.μILS)
# plot(reshape(Rintr[j,l,:],64), reshape(μintr[j,l,:],64), ".")
# plot(reshape(Rlsc[j,l,:],64), reshape(μlsc[j,l,:],64), ".")

function Pair_solver(R::Array, μ::Array, crd::Cord, ils::LS)
    R = reshape(R, length(R))
    μ = reshape(μ, length(μ))
    μon = zeros(μ)

    for idx in eachindex(μ)
        μon[idx] = Perp_solver(R[idx], μ[idx], crd, ils)
    end

    Ron  = ils.Rspl(μon)
    CRon = ils.CRspl(μon)
    Cμon = ils.Cμspl(μon)

    Rintr = 2Ron - R
    μintr = 2μon - μ
    weigh = (μon + crd.δμ)./(1+2crd.δμ)
    xintr = (Rintr - R)./CRon .*(1-weigh) + (μintr - μ)./Cμon .*weigh
    return Rintr, μintr, xintr, μon
end

function Perp_solver(R::Float64, μ::Float64, crd::Cord, ils::LS)
    if μ < 0.
        μon = μ
    else
        if μ > 1.
            μbd1 = ils.μspl(R)
            μbd2 = ils.μspl(r2R(crd.rmin)-crd.δR)
            μmin = min(μbd1, μbd2)
            μmax = max(μbd1, μbd2)
        else
            μmin = max(-crd.δμ, μ - 4crd.δμ)
            μmax = max(1., μ + 4crd.δμ)
        end
        μsamp   = linspace(μmin, μmax, 128)
        Rsamp   = ils.Rspl(μsamp)
        CR_samp = ils.CRspl(μsamp)
        Cμ_samp = ils.Cμspl(μsamp)

        dR = Rsamp - R
        dμ = μsamp - μ

        orth = dR .* Cμ_samp - dμ .* CR_samp

        spl = (orth[end] > orth[1]) ? Spline1D(orth, μsamp) : Spline1D(-orth, μsamp)
        μon = spl(0.)
    end
    return μon
end

function Ω_I_LS(crd::Cord, ils::LS, Ω_I::Ω_and_I)
    μILS = ils.μILS
    RILS = ils.Rspl(μILS)
    UILS = ils.Uspl(μILS)

    ∂1Ωspl = Spline2D(crd.μcol, crd.Rcol, Ω_I.∂1Ω, kx = 1, ky = 1)
    ∂2Ωspl = Spline2D(crd.μcol, crd.Rcol, Ω_I.∂2Ω, kx = 1, ky = 1)

    ∂11Ωspl = Spline2D(crd.μcol, crd.Rcol, Ω_I.∂11Ω, kx = 1, ky = 1)
    ∂12Ωspl = Spline2D(crd.μcol, crd.Rcol, Ω_I.∂12Ω, kx = 1, ky = 1)
    ∂22Ωspl = Spline2D(crd.μcol, crd.Rcol, Ω_I.∂22Ω, kx = 1, ky = 1)

    ∂1IIpspl = Spline2D(crd.μcol, crd.Rcol, Ω_I.∂1IIp, kx = 1, ky = 1)
    ∂2IIpspl = Spline2D(crd.μcol, crd.Rcol, Ω_I.∂2IIp, kx = 1, ky = 1)

    Ω   = Ω_I.Ωspl(UILS)
    IIp = Ω_I.IIpspl(UILS)

    ∂1Ω  =  ∂1Ωspl(μILS, RILS)
    ∂2Ω  =  ∂2Ωspl(μILS, RILS)
    ∂11Ω =  ∂11Ωspl(μILS, RILS)
    ∂12Ω =  ∂12Ωspl(μILS, RILS)
    ∂22Ω =  ∂22Ωspl(μILS, RILS)
    ∂1IIp=  ∂1IIpspl(μILS, RILS)
    ∂2IIp=  ∂2IIpspl(μILS, RILS)
    return Ω_I_LS(Ω, ∂1Ω, ∂2Ω, ∂11Ω, ∂12Ω, ∂22Ω, IIp,  ∂1IIp, ∂2IIp)
end

function Ω_I_LS!(crd::Cord, ils::LS, Ω_I::Ω_and_I, Ω_I_ls::Ω_I_LS)
    Ω    =  Ω_I_ls.Ω
    ∂1Ω  =  Ω_I_ls.∂1Ω
    ∂2Ω  =  Ω_I_ls.∂2Ω
    ∂11Ω =  Ω_I_ls.∂11Ω
    ∂12Ω =  Ω_I_ls.∂12Ω
    ∂22Ω =  Ω_I_ls.∂22Ω

    μILS = ils.μILS
    RILS = ils.Rspl(μILS)
    UILS = ils.Uspl(μILS)

    ∂1IIpspl = Spline2D(crd.μcol, crd.Rcol, Ω_I.∂1IIp, kx = 1, ky = 1)
    ∂2IIpspl = Spline2D(crd.μcol, crd.Rcol, Ω_I.∂2IIp, kx = 1, ky = 1)
    IIp  =  Ω_I.IIpspl(UILS)
    ∂1IIp=  ∂1IIpspl(μILS, RILS)
    ∂2IIp=  ∂2IIpspl(μILS, RILS)
    return Ω_I_LS(Ω, ∂1Ω, ∂2Ω, ∂11Ω, ∂12Ω, ∂22Ω, IIp,  ∂1IIp, ∂2IIp)
end


function Grid_LS(crd::Cord, ils::LS, Ω_I_ls::Ω_I_LS)
    ∂1Ω   = Ω_I_ls.∂1Ω
    ∂2Ω   = Ω_I_ls.∂2Ω
    ∂11Ω  = Ω_I_ls.∂11Ω
    ∂12Ω  = Ω_I_ls.∂12Ω
    ∂22Ω  = Ω_I_ls.∂22Ω
    IIp   = Ω_I_ls.IIp
    ∂1IIp = Ω_I_ls.∂1IIp
    ∂2IIp = Ω_I_ls.∂2IIp

    κpr  = κ_partials(crd, ils, Ω_I_ls)
    ∂1κ  = κpr.∂1κ
    ∂2κ  = κpr.∂2κ
    ∂Ωκ  = κpr.∂Ωκ
    ∂11κ = κpr.∂11κ
    ∂12κ = κpr.∂12κ
    ∂22κ = κpr.∂22κ
    ∂1Ωκ = κpr.∂1Ωκ
    ∂2Ωκ = κpr.∂2Ωκ

    μ = ils.μILS;    a = crd.a
    R = ils.Rspl(μ); r = R2r(R)

    mtr_ls = Geom(r, μ, a)
    β_Σ    = mtr_ls.β_Σ
    Σ_Δ    = mtr_ls.Σ_Δ
    ∂1Σ_Δ  = mtr_ls.∂1Σ_Δ
    ∂2Σ_Δ  = mtr_ls.∂2Σ_Δ
    s_Δ    = mtr_ls.s_Δ;    s_Δ[end] = Lmt(crd)
    ∂1s_Δ  = mtr_ls.∂1s_Δ
    ∂2s_Δ  = mtr_ls.∂1s_Δ
    sst    = 1-μ.^2

    ∂rκ  =  ∂1κ +  ∂Ωκ.* ∂1Ω
    ∂μκ  =  ∂2κ +  ∂Ωκ.* ∂2Ω
    Cr   =  ∂1κ +  0.5*∂Ωκ.* ∂1Ω
    Cμ   = (∂2κ +  0.5*∂Ωκ.* ∂2Ω).*s_Δ

    ∂rCr = ∂11κ +  1.5*∂1Ω.*∂1Ωκ              + 0.5*∂Ωκ.*∂11Ω + β_Σ.*sst.*(∂1Ω).^2
    ∂μCr = ∂12κ + (0.5*∂1Ω.*∂2Ωκ + ∂2Ω.*∂1Ωκ) + 0.5*∂Ωκ.*∂12Ω + β_Σ.*sst.*(∂1Ω.*∂2Ω)
    tp_μ = ∂22κ +  1.5*∂2Ω.*∂2Ωκ              + 0.5*∂Ωκ.*∂22Ω + β_Σ.*sst.*(∂2Ω).^2
    tp_r = ∂12κ + (0.5*∂2Ω.*∂1Ωκ + ∂1Ω.*∂2Ωκ) + 0.5*∂Ωκ.*∂12Ω + β_Σ.*sst.*(∂1Ω.*∂2Ω)
    ∂μCμ = s_Δ.*tp_μ + ∂2s_Δ .* (∂2κ +  0.5*∂Ωκ.* ∂2Ω)
    ∂rCμ = s_Δ.*tp_r + ∂1s_Δ .* (∂2κ +  0.5*∂Ωκ.* ∂2Ω)

    Crr_lpt =  ∂rκ.*Cr + Cr.^2 + ∂μκ.*Cμ
    Cμμ_lpt = (∂rκ.*Cr +∂μκ.*Cμ).*s_Δ + Cμ.^2
    Crμ_lpt = 2Cr.*Cμ
    Cr_lpt  = Cr.*∂rCr + Cμ.*∂μCr
    Cμ_lpt  = Cr.*∂rCμ + Cμ.*∂μCμ

    ∂1S   = ∂1Σ_Δ .* IIp + Σ_Δ .* ∂1IIp
    ∂2S   = ∂2Σ_Δ .* IIp + Σ_Δ .* ∂2IIp
    S_lpt = Cr.*∂1S  + Cμ.*∂2S

    CRR_lpt = Crr_lpt .* (1-R).^4
    CRμ_lpt = Crμ_lpt .* (1-R).^2
    CR_lpt  = Cr_lpt  .* (1-R).^2 - 2*Crr_lpt.*(1-R).^3

    δμ = crd.δμ
    δR = crd.δR
    δ  = min(δμ, δR)

    aa = (Cμμ_lpt ./ δμ^2 + 0.5*Cμ_lpt ./ δμ) .* δ^2
    bb = (Cμμ_lpt ./ δμ^2 - 0.5*Cμ_lpt ./ δμ) .* δ^2
    cc = (CRR_lpt ./ δR^2 + 0.5*CR_lpt ./ δR) .* δ^2
    dd = (CRR_lpt ./ δR^2 - 0.5*CR_lpt ./ δR) .* δ^2
    ee = -2*(Cμμ_lpt ./ δμ^2 + CRR_lpt ./ δR^2) .* δ^2
    ff =  S_lpt .* δ^2
    ww = CRμ_lpt /(4δR*δμ) *δ^2
    return Grid_LS(aa, bb, cc, dd, ee, ff, ww)
end

function Grid_LS!(crd::Cord, ils::LS, grd_ls::Grid_LS, Ω_I_ls::Ω_I_LS)
    aa = grd_ls.aa
    bb = grd_ls.bb
    cc = grd_ls.cc
    dd = grd_ls.dd
    ee = grd_ls.ee
    ww = grd_ls.ww

    μ = ils.μILS; a = crd.a
    R = ils.Rspl(μ); r = R2r(R)

    mtr_ls = Geom(r, μ, a)
    Σ_Δ   = mtr_ls.Σ_Δ
    ∂1Σ_Δ = mtr_ls.∂1Σ_Δ
    ∂2Σ_Δ = mtr_ls.∂2Σ_Δ

    μILS = ils.μILS
    RILS = ils.Rspl(μILS)
    Cr   = ils.Crspl(μILS)
    Cμ   = ils.Cμspl(μILS)

    IIp   = Ω_I_ls.IIp
    ∂1IIp = Ω_I_ls.∂1IIp
    ∂2IIp = Ω_I_ls.∂2IIp

    ∂1S   = ∂1Σ_Δ .* IIp + Σ_Δ .* ∂1IIp
    ∂2S   = ∂2Σ_Δ .* IIp + Σ_Δ .* ∂2IIp
    S_lpt = Cr.*∂1S  + Cμ.*∂2S

    δμ = crd.δμ
    δR = crd.δR
    δ  = min(δμ, δR)
    ff =  S_lpt .* δ^2
    return Grid_LS(aa, bb, cc, dd, ee, ff, ww)
end


function κ_partials(crd::Cord, ils::LS, Ω_I_ls::Ω_I_LS)
    μ = ils.μILS
    R = ils.Rspl(μ)
    r = R2r(R)
    a = crd.a

    Σ = r.^2 + a^2 .* μ.^2
    Δ = r.^2 - 2r + a^2
    β = Δ .* Σ + 2r .* (r.^2+a^2)   # also (r^2+a^2)Σ + 2r a^2 sst

    r_Σ   = r ./Σ
    ∂1r_Σ = 1 ./Σ - 2r.^2 ./ Σ.^2
    ∂2r_Σ = r .* (- 2a^2 * μ ./ Σ.^2)

    ∂11r_Σ = - 6r ./ Σ.^2 + 8r.^3 ./ Σ.^3
    ∂12r_Σ = - 2a^2 .* μ ./ Σ.^2  + 4r.^2 .* (2a^2 .*μ) ./ Σ.^3
    ∂22r_Σ = - 2a^2 .* r .* (r.^2 -3a^2 .* μ.^2) ./ Σ.^3

    sst   = 1 - μ.^2
    β_Σ   = β ./Σ
    ∂1β_Σ = 2r + 2a^2 .*sst .* ∂1r_Σ
    ∂2β_Σ = 2r .*(r.^2+a^2) .* (- 2a^2 .* μ ./ Σ.^2)

    ∂11β_Σ = 2 + 2a^2 .*sst .* ∂11r_Σ
    ∂12β_Σ = 2a^2 .* (-2μ) .* ∂1r_Σ + 2a^2 .*sst .* ∂12r_Σ
    ∂22β_Σ = 2r .*(r.^2+a^2) .* (- 2a^2) .* (r.^2 -3a^2 .* μ.^2) ./ Σ.^3

    Ω   = Ω_I_ls.Ω
    κ   = (β_Σ   .* Ω.^2 - 4a .*   r_Σ .* Ω) .* sst - (1- 2*r_Σ)
    ∂1κ = (∂1β_Σ .* Ω.^2 - 4a .* ∂1r_Σ .* Ω) .* sst +  2*∂1r_Σ
    ∂2κ = (∂2β_Σ .* Ω.^2 - 4a .* ∂2r_Σ .* Ω) .* sst +  2*∂2r_Σ +(β_Σ .* Ω.^2 - 4a .*r_Σ .* Ω) .*(-2μ)
    ∂Ωκ = 2*(β_Σ .* Ω - 2a .* r_Σ) .* sst

    ∂11κ = (∂11β_Σ .* Ω.^2 - 4a .* ∂11r_Σ .* Ω) .* sst +  2*∂11r_Σ
    ∂12κ = (∂12β_Σ .* Ω.^2 - 4a .* ∂12r_Σ .* Ω) .* sst +  2*∂12r_Σ
    ∂22κ = (∂22β_Σ .* Ω.^2 - 4a .* ∂22r_Σ .* Ω) .* sst +  2*∂22r_Σ

    ∂1Ωκ = 2*(∂1β_Σ .* Ω - 2a .* ∂1r_Σ) .* sst
    ∂2Ωκ = 2*(∂2β_Σ .* Ω - 2a .* ∂2r_Σ) .* sst + 2*(β_Σ .* Ω - 2a .* r_Σ) .* (-2μ)
    return κ_partials(κ, ∂1κ, ∂2κ, ∂Ωκ, ∂11κ, ∂12κ, ∂22κ, ∂1Ωκ, ∂2Ωκ)
end

a = 0.99

function static_lmt(a)
    rH  = 1.+√(1-a^2)
    ΩH  = a/(2rH)
    rng = linspace(rH, 2., 20)
    μ   = 0.

    Σ = rng.^2 + a^2 * μ^2
    C = -1 + 2*rng./Σ
    B = -4a*rng*(1-μ^2) ./ Σ
    A = ( (rng.^2 + a^2).^2 - a^2 *(rng.^2 -2a*rng + a^2) ) ./Σ *(1-μ^2)

    Ω1 = (-B + √(B.^2-4A.*C))./(2A)
    Ω2 = (-B - √(B.^2-4A.*C))./(2A)
    return Ω1/(0.5ΩH), Ω2/(0.5ΩH)

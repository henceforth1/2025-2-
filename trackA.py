import numpy as np
import json

# ============================================================
# TRACK A — FAST STABILITY FILTER (STEP1~STEP4)
# ============================================================

M_EARTH_TO_MSUN = 3.003e-6   # 1 M_earth = 3.003e-6 M_sun


# ------------------------------------------------------------
# Utility Samplers
# ------------------------------------------------------------
def rayleigh_trunc(size, sigma, max_val=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()-7
    x = rng.rayleigh(scale=sigma, size=size)
    if max_val is not None:
        mask = x > max_val
        while np.any(mask):
            x[mask] = rng.rayleigh(scale=sigma, size=np.sum(mask))
            mask = x > max_val
    return x


def mass_from_radius(R_earth, rng=None,
                     mode="piecewise",
                     alpha=2.0, k=1.0,
                     scatter_dex=0.25):
    if rng is None:
        rng = np.random.default_rng()
    """
    R_earth → M_earth 변환 (로그정규 산포 포함)
    """
    R = np.asarray(R_earth)
    if mode == "single":
        M = k * (R ** alpha)
    else:
        M = np.empty_like(R, dtype=float)
        mask1 = R <= 1.5
        mask2 = (R > 1.5) & (R <= 4.0)
        mask3 = R > 4.0
        M[mask1] = 1.0 * (R[mask1] ** 3.7)
        M[mask2] = 2.0 * (R[mask2] ** 1.6)
        M[mask3] = 10.0 * (R[mask3] ** 1.0)

    sigma_ln = scatter_dex * np.log(10)
    M *= np.exp(rng.normal(0.0, sigma_ln, size=R.size))
    return M


# ------------------------------------------------------------
# STEP 1 — Sample system (a, m, e, inc, R)
# ------------------------------------------------------------
def chain_by_delta(a1, m_chain, Mstar=1.0, delta_chain=None, m_in="Mearth"):
    """
    Δ-chain relation → generate a[i]
    """
    n = len(m_chain)
    if n == 1:
        return np.array([a1])

    if delta_chain is None or len(delta_chain) != n-1:
        raise ValueError("delta_chain length must be n-1")

    m_use = np.array(m_chain, dtype=float, copy=True)
    if m_in.lower() == "mearth":
        m_use *= M_EARTH_TO_MSUN

    a = np.empty(n, float)
    a[0] = a1

    for i in range(n-1):
        mu = ((m_use[i] + m_use[i+1]) / (3 * Mstar)) ** (1/3)
        k = float(delta_chain[i]) * mu
        k = min(k, 1.9)

        ratio = (1 + 0.5*k) / (1 - 0.5*k)
        ratio = max(ratio, 1.0001)

        a[i+1] = a[i] * ratio

    return a


def sample_system(n_planets=4,
                  Mstar=1.0,
                  a_inner=0.3,
                  mean_delta=9.5, sd_delta=1.0,
                  r_logmean=np.log(1.8), r_logstd=0.35,
                  mass_mode="piecewise", mass_scatter_dex=0.25,
                  sigma_e=0.04, e_max=0.35,
                  sigma_i=1.8, i_max=12.0,
                  rng=None):
    if rng is None:
        rng = np.random.default_rng()
    """
    Return dict: {a, m, e, inc, R, deltas}
    """
    R = np.exp(rng.normal(r_logmean, r_logstd, size=n_planets))
    m = mass_from_radius(R, rng=rng, mode=mass_mode,
                         scatter_dex=mass_scatter_dex)

    deltas = rng.normal(mean_delta, sd_delta, size=n_planets-1)
    deltas = np.clip(deltas, 3.5, None)

    a = chain_by_delta(a1=a_inner, m_chain=m, Mstar=Mstar,
                       delta_chain=deltas, m_in="Mearth")

    e = rayleigh_trunc(n_planets, sigma=sigma_e, max_val=e_max, rng=rng)
    inc = rayleigh_trunc(n_planets, sigma=sigma_i, max_val=i_max, rng=rng)

    return {"a": a, "m": m, "e": e, "inc": inc, "R": R, "deltas": deltas}


# ------------------------------------------------------------
# STEP 2 — Δ mutual Hill stability
# ------------------------------------------------------------
def mutual_hill_radius(a1, a2, m1, m2, Mstar=1.0, m_in="Mearth"):
    if m_in.lower() == "mearth":
        m1 *= M_EARTH_TO_MSUN
        m2 *= M_EARTH_TO_MSUN
    return ((m1 + m2)/(3*Mstar))**(1/3) * (a1 + a2)/2


def delta_mutual_hill(a1, a2, m1, m2, Mstar=1.0, m_in="Mearth"):
    RH = mutual_hill_radius(a1, a2, m1, m2, Mstar=Mstar, m_in=m_in)
    return (a2 - a1) / RH


def evaluate_delta_cut(a, m, Mstar=1.0, delta_crit=10.0, m_in="Mearth"):
    a = np.asarray(a)
    m = np.asarray(m)
    n = len(a)

    D = np.empty(n-1)
    for i in range(n-1):
        D[i] = delta_mutual_hill(a[i], a[i+1], m[i], m[i+1],
                                 Mstar=Mstar, m_in=m_in)

    ok = np.all(D >= delta_crit)
    return {"pass": ok, "deltas": D, "min_delta": float(D.min())}


# ------------------------------------------------------------
# STEP 3 — Orbit overlap cut
# ------------------------------------------------------------
def evaluate_orbit_overlap(a, e, tol=0.0):
    a = np.asarray(a)
    e = np.asarray(e)

    overlaps = []
    for i in range(len(a)-1):
        Q = a[i] * (1 + e[i])
        q = a[i+1] * (1 - e[i+1])
        if Q + tol >= q:
            overlaps.append(i)

    return {"pass": len(overlaps) == 0, "overlaps": overlaps}


# ------------------------------------------------------------
# STEP 4 — AMD stability
# ------------------------------------------------------------
def amd_pair_critical(a1, a2, m1, m2, Mstar=1.0, m_in="Mearth"):
    if m_in.lower() == "mearth":
        m1 *= M_EARTH_TO_MSUN
        m2 *= M_EARTH_TO_MSUN

    alpha = a1 / a2
    e_crit = 1 - alpha
    AMDcrit = (m1*np.sqrt(a1) + m2*np.sqrt(a2)) * (1 - np.sqrt(1 - e_crit**2))
    return AMDcrit


def amd_pair_current(a1, a2, m1, m2, e1, e2, Mstar=1.0, m_in="Mearth"):
    if m_in.lower() == "mearth":
        m1 *= M_EARTH_TO_MSUN
        m2 *= M_EARTH_TO_MSUN

    AMD1 = m1*np.sqrt(a1) * (1 - np.sqrt(1 - e1**2))
    AMD2 = m2*np.sqrt(a2) * (1 - np.sqrt(1 - e2**2))
    return AMD1 + AMD2


def evaluate_AMD_stability(a, m, e, Mstar=1.0, m_in="Mearth"):
    a = np.asarray(a)
    m = np.asarray(m)
    e = np.asarray(e)

    unstable = []
    for i in range(len(a)-1):
        cur = amd_pair_current(a[i], a[i+1], m[i], m[i+1], e[i], e[i+1],
                               Mstar=Mstar, m_in=m_in)
        crit = amd_pair_critical(a[i], a[i+1], m[i], m[i+1],
                                 Mstar=Mstar, m_in=m_in)
        if cur >= crit:
            unstable.append(i)

    return {"pass": len(unstable) == 0, "unstable_pairs": unstable}


# ------------------------------------------------------------
# TRACK A — Full evaluation (STEP2 + STEP3 + STEP4)
# ------------------------------------------------------------
def evaluate_trackA(a, m, e,
                    Mstar=1.0,
                    delta_crit=10.0,
                    overlap_tol=0.0,
                    m_in="Mearth"):
    """
    Track A = Δ-cut + orbit-overlap-cut + AMD-stability
    """
    step2 = evaluate_delta_cut(a, m, Mstar=Mstar,
                               delta_crit=delta_crit, m_in=m_in)

    step3 = evaluate_orbit_overlap(a, e, tol=overlap_tol)

    step4 = evaluate_AMD_stability(a, m, e, Mstar=Mstar, m_in=m_in)

    stable = step2["pass"] and step3["pass"] and step4["pass"]

    return {
        "stable": stable,
        "step2": step2,
        "step3": step3,
        "step4": step4,
    }

# ------------------------------------------------------------
# Track A 대량 샘플 생성 루프
# ------------------------------------------------------------

def generate_systems_trackA(
    Nsystems=50000,
    n_planets=5,
    Mstar=1.0,
    delta_crit=10.0,
    overlap_tol=0.0,
    seed=None,
):
    """
    Nsystems 개의 다행성계 생성 후 Track A 안정성 평가.
    
    반환:
      stable_list : Track A 통과한 시스템들의 dict 리스트
      stats       : 통계 (생성 개수, 안정 개수, 비율)
    """
    rng = np.random.default_rng(seed)

    stable_list = []
    n_stable = 0

    for k in range(Nsystems):
        # 1) 샘플 생성
        sys = sample_system(n_planets=n_planets, Mstar=Mstar)


        # 2) Track A 안정성 평가
        TA = evaluate_trackA(
            sys["a"], sys["m"], sys["e"],
            Mstar=Mstar,
            delta_crit=delta_crit,
            overlap_tol=overlap_tol
        )

        # 3) 통과한 시스템 저장
        if TA["stable"]:
            n_stable += 1
            stable_list.append({
                "a": sys["a"],
                "m": sys["m"],
                "e": sys["e"],
                "inc": sys["inc"],
                "R": sys["R"],
                "deltas": sys["deltas"],
                "trackA": TA
            })
        


    stats = {
        "N": Nsystems,
        "stable": n_stable,
        "unstable": Nsystems - n_stable,
        "stable_fraction": n_stable / Nsystems
    }
    return stable_list, stats

def save_trackA_candidates(filename, stable_list):
    import os
    import json
    import numpy as np

    # trackA.py가 있는 디렉토리 (절대경로)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 그 디렉토리에 저장
    filepath = os.path.join(base_dir, filename)

    serializable = []
    for sys in stable_list:
        serializable.append({
            "a": np.asarray(sys["a"]).tolist(),
            "m": np.asarray(sys["m"]).tolist(),
            "e": np.asarray(sys["e"]).tolist(),
            "inc": np.asarray(sys["inc"]).tolist(),
            "R": np.asarray(sys["R"]).tolist(),
            "deltas": np.asarray(sys["deltas"]).tolist(),
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"[INFO] Saved Track A candidates to:\n{filepath}")

    

    


# ============================================================
if __name__ == "__main__":

    stable_list, stats = generate_systems_trackA(
        Nsystems=50000,
        n_planets=5,
        delta_crit=10.0,
        overlap_tol=0.0,
        seed=123
    )

    print("---------- 결과 ----------")
    print("총 생성:", stats["N"])
    print("안정:", stats["stable"])
    print("불안정:", stats["unstable"])
    print("안정 비율:", f"{stats['stable_fraction']*100:.2f}%")

    if len(stable_list) > 0:
        print("\n첫 안정 시스템 예시:")
        print("a :", stable_list[0]["a"])
        print("m :", stable_list[0]["m"])
        print("e :", stable_list[0]["e"])

    # Track A 생존자 JSON 저장 (Track B에서 사용)
    if len(stable_list) > 0:
        save_trackA_candidates("trackA_candidates.json", stable_list)
import json
import numpy as np
import rebound

# 지구질량 → 태양질량 변환 (track A와 동일 상수)
M_EARTH_TO_MSUN = 3.003e-6


# ------------------------------------------------------------
# 1. Track A 후보 시스템 불러오기
# ------------------------------------------------------------
def load_trackA_candidates(filename="trackA_candidates.json"):
    """
    trackA.py2에서 저장한 JSON 파일을 읽어들임.
    각 원소는 dict: {a, m, e, inc, R, deltas}
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # list of dict


# ------------------------------------------------------------
# 2. Mutual Hill 반지름 (Track B에서 close encounter 판정용)
# ------------------------------------------------------------
def mutual_hill_radius(a1, a2, m1, m2, Mstar=1.0):
    """
    a1, a2 : AU
    m1, m2 : M_earth
    Mstar  : M_sun
    """
    m1_sun = m1 * M_EARTH_TO_MSUN
    m2_sun = m2 * M_EARTH_TO_MSUN
    return ((m1_sun + m2_sun) / (3.0 * Mstar)) ** (1.0 / 3.0) * (a1 + a2) / 2.0


# ------------------------------------------------------------
# 3. 하나의 시스템을 REBOUND Simulation으로 만들기
# ------------------------------------------------------------
def make_sim_from_system(sys, Mstar=1.0, seed=None):
    """
    sys: dict with keys "a", "m", "e", "inc" (deg), ...
    Mstar: stellar mass in Msun
    """
    rng = np.random.default_rng(seed)

    a_arr   = np.array(sys["a"],   dtype=float)
    m_arr   = np.array(sys["m"],   dtype=float)   # M_earth
    e_arr   = np.array(sys["e"],   dtype=float)
    inc_arr = np.array(sys["inc"], dtype=float)   # degrees

    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=Mstar)   # central star

    for a, m_earth, e, inc_deg in zip(a_arr, m_arr, e_arr, inc_arr):
        m_sun = m_earth * M_EARTH_TO_MSUN
        inc_rad = np.deg2rad(inc_deg)

        # 무작위 궤도 위상
        Omega  = rng.uniform(0.0, 2.0 * np.pi)
        omega  = rng.uniform(0.0, 2.0 * np.pi)
        M_anom = rng.uniform(0.0, 2.0 * np.pi)

        sim.add(m=m_sun, a=a, e=e, inc=inc_rad,
                Omega=Omega, omega=omega, M=M_anom)

    sim.move_to_com()
    sim.integrator = "whfast"  # 빠른 심플렉틱 적분기 사용

    return sim


# ------------------------------------------------------------
# 4. 하나의 시스템에 대해 Track B 안정성 평가
# ------------------------------------------------------------
def evaluate_trackB_system(sys,
                           sys_id,
                           Mstar=1.0,
                           N_orbits=1e6,
                           dt_factor=20,
                           check_interval_orbits=200,
                           close_encounter_k=1.0,
                           escape_factor=10.0,
                           max_ecc=0.5,
                           max_da_frac=0.5,
                           seed=None):
    """
    하나의 시스템(sys)에 대해 N-body 적분으로 안정성 판정.

    판정 기준:
      - |a - a0| / a0 > max_da_frac      → unstable
      - e > max_ecc                      → unstable
      - r > escape_factor * a_outer0     → escape → unstable
      - d_ij < close_encounter_k * R_H   → close encounter → unstable
    """
    sim = make_sim_from_system(sys, Mstar=Mstar, seed=seed)

    # 초기 궤도 요소 (REBOUND v4: sim.orbits() 사용)
    orbits0 = sim.orbits()       # star 제외한 모든 행성 orbit
    n_planets = len(orbits0)

    a0 = np.array([o.a for o in orbits0])
    e0 = np.array([o.e for o in orbits0])

    a_inner = a0[0]
    a_outer0 = a0[-1]

    # inner planet 기준 궤도 주기 (G=1, Mstar=1 기준)
    P_inner = 2.0 * np.pi * np.sqrt(a_inner ** 3 / Mstar)

    dt = P_inner / dt_factor
    sim.dt = dt

    T_end = N_orbits * P_inner
    check_interval = check_interval_orbits * P_inner

    t = 0.0
    next_check = check_interval

    B_stable = True
    reason = "no_instability_detected"

    m_arr = np.array(sys["m"], dtype=float)   # for Hill radius

    while t < T_end:
        # 한 스텝 적분
        t = min(t + dt, T_end)
        sim.integrate(t)

        # 검사 시점이 아니면 continue
        if t < next_check and t < T_end:
            continue

           # 현재 궤도/위치 정보 업데이트
        particles = sim.particles
        orbits = sim.orbits()    # 행성들 orbit (길이 = n_planets)

        a_now = np.array([o.a for o in orbits])
        e_now = np.array([o.e for o in orbits])

        # (1) a 변화율 검사
        da_frac = np.abs(a_now - a0) / a0
        if np.any(da_frac > max_da_frac):
            B_stable = False
            reason = "large_da"
            break

        # (2) 이심률 폭주 검사
        if np.any(e_now > max_ecc):
            B_stable = False
            reason = "eccentricity_growth"
            break

        # (3) 탈출 검사 (중심에서 너무 멀어지면)
        for j in range(1, n_planets + 1):   # particles[0] = star
            pj = particles[j]
            r = np.sqrt(pj.x**2 + pj.y**2 + pj.z**2)
            if r > escape_factor * a_outer0:
                B_stable = False
                reason = "escape"
                break
        if not B_stable:
            break

        # (4) close encounter 검사 (mutual Hill 기준)
        for i in range(n_planets - 1):
            for j in range(i + 1, n_planets):
                pi = particles[i + 1]
                pj = particles[j + 1]

                dx = pi.x - pj.x
                dy = pi.y - pj.y
                dz = pi.z - pj.z
                dij = np.sqrt(dx*dx + dy*dy + dz*dz)

                RH = mutual_hill_radius(a_now[i], a_now[j],
                                        m_arr[i], m_arr[j],
                                        Mstar=Mstar)
                if dij < close_encounter_k * RH:
                    B_stable = False
                    reason = "close_encounter"
                    break
            if not B_stable:
                break

        # 다음 검사 시점
        next_check += check_interval

    return {
        "id": int(sys_id),       # Track A candidates 리스트 인덱스
        "B_stable": bool(B_stable),
        "reason": reason
    }


# ------------------------------------------------------------
# 5. Track A 모든 후보에 대해 Track B 수행
# ------------------------------------------------------------
def run_trackB_for_all(
    input_file="trackA_candidates.json",
    output_file="trackB_results.json",
    Mstar=1.0,
    N_orbits=1e6,            # 시험용이면 1e5 정도로 줄여서 먼저 돌려봐도 됨
    dt_factor=20,
    check_interval_orbits=200,
    close_encounter_k=1.0,
    escape_factor=10.0,
    max_ecc=0.5,
    max_da_frac=0.5,
    seed=1234
):
    systems = load_trackA_candidates(input_file)
    rng = np.random.default_rng(seed)

    results = []

    print(f"[INFO] Loaded {len(systems)} Track A candidates from {input_file}")

    for idx, sys in enumerate(systems):
        sys_seed = rng.integers(0, 2**32 - 1)

        res = evaluate_trackB_system(
            sys=sys,
            sys_id=idx,
            Mstar=Mstar,
            N_orbits=N_orbits,
            dt_factor=dt_factor,
            check_interval_orbits=check_interval_orbits,
            close_encounter_k=close_encounter_k,
            escape_factor=escape_factor,
            max_ecc=max_ecc,
            max_da_frac=max_da_frac,
            seed=sys_seed
        )
        print(f"[Track B] idx={res['id']}, B_stable={res['B_stable']}, reason={res['reason']}")
        results.append(res)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Saved Track B results to {output_file}")


# ------------------------------------------------------------
# 6. 메인
# ------------------------------------------------------------
if __name__ == "__main__":
    run_trackB_for_all(
        input_file="trackA_candidates.json",
        output_file="trackB_results.json",
        Mstar=1.0,
        N_orbits=1e6,           # 너무 느리면 일단 1e5로 줄여서 테스트
        dt_factor=20,
        check_interval_orbits=200,
        close_encounter_k=1.0,
        escape_factor=10.0,
        max_ecc=0.5,
        max_da_frac=0.5,
        seed=2025
    )

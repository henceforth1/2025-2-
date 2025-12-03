
# ============================================
# peas-in-a-pod 분석 준비 — Step 1 & 2 only
#  - Step 1: CSV 읽기 (옛 데이터 그대로 사용)
#  - Step 2: 전처리 재사용 (리네임, 단위 통일, 파생변수 logP/logR)
# ============================================

import numpy as np
import pandas as pd

# ----- Step 1: 데이터 준비 (기존 CSV 그대로 사용) -----
#  - NASA Archive CSV의 주석('#') 무시
#  - BOM 가능성 대비 utf-8-sig
DATA_PATH = r"C:\Users\woori\OneDrive - 서울과학고등학교\문서\김창우\설곽\과제연구\2025 2학기\파이썬\PSCompPars1.csv"
df_raw = pd.read_csv(DATA_PATH, comment="#", encoding="utf-8-sig")

# ----- Step 2: 전처리 재사용 (리네임, 단위 통일, 파생변수) -----
# (A) 컬럼 리네임
rename_map = {
    "hostname":        "system_id",
    "pl_name":         "planet_id",
    "sy_pnum":         "n_planets",
    "sy_snum":         "n_stars",
    "pl_orbper":       "P_days",
    "pl_rade":         "R_earth",
    "pl_radj":         "R_jup",
    "pl_bmasse":       "M_earth",
    "pl_bmassj":       "M_jup",
    "discoverymethod": "disc_method",
    # 있으면 쓰는 확장 변수
    "pl_orbeccen":     "eccentricity",
    "pl_orbeccen1":    "eccentricity",
    "pl_dens":         "planet_density",
    "pl_dens1":        "planet_density",
}
df = df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns})
# 누락된 별칭은 NaN으로 생성해 두면 이후 코드가 안전해짐
for _, alias in rename_map.items():
    if alias not in df.columns:
        df[alias] = np.nan

# (B) 단위 통일 (목성 → 지구)
JUP_TO_EARTH_R = 11.209   # 1 R_J = 11.209 R_⊕
JUP_TO_EARTH_M = 317.8    # 1 M_J = 317.8  M_⊕
mask_R = df["R_earth"].isna() & df["R_jup"].notna()
df.loc[mask_R, "R_earth"] = df.loc[mask_R, "R_jup"] * JUP_TO_EARTH_R
mask_M = df["M_earth"].isna() & df["M_jup"].notna()
df.loc[mask_M, "M_earth"] = df.loc[mask_M, "M_jup"] * JUP_TO_EARTH_M

# (C) 파생변수 (peas-in-a-pod에 필요한 로그 스케일)
#  - 계 내부 순서정렬/인접쌍 비교에 logR, logP를 자주 씀
df["logP"] = np.where(df["P_days"] > 0, np.log10(df["P_days"]), np.nan)
df["logR"] = np.where(df["R_earth"] > 0, np.log10(df["R_earth"]), np.nan)

# (D) (선택) 라벨 — 이후 다른 분석에도 유용하니 추가
df["class"] = "multi"
df.loc[df["n_planets"] == 1, "class"] = "single"

# ---- 확인 출력 ----
print("✅ Step1&2 완료")
print("전체 행/열:", df.shape)
need = ["system_id","planet_id","n_planets","P_days","R_earth","M_earth",
        "logP","logR","disc_method","eccentricity","planet_density"]
print("필수/확장 컬럼 존재 여부:", {c: (c in df.columns) for c in need})
print("\n미리보기:")
print(df[need].head(8))

# ============================================
# Step 3: 다행성계 필터링 & 정렬(+계 내부 순서 k 부여)
# ============================================

# 3-1) 다행성계만 사용 (sy_pnum >= 2)
df_multi = df.loc[df["n_planets"] >= 2].copy()

# 3-2) 정렬을 위한 필수값 점검
#  - 주기(P_days)가 있어야 계 내부 정렬이 가능
#  - P_days > 0인 케이스만 사용 (0/음수/NaN 제거)
mask_validP = df_multi["P_days"].notna() & (df_multi["P_days"] > 0)
df_multi = df_multi.loc[mask_validP].copy()

# 3-3) 계 내부 주기 오름차순 정렬
#  - 동일 주기(동률)일 경우를 대비해 보조키로 반지름/행성ID를 추가(결정론적 순서 보장)
df_multi.sort_values(
    by=["system_id", "P_days", "R_earth", "planet_id"],
    ascending=[True, True, True, True],
    inplace=True
)

# 3-4) 계 내부 순서 인덱스 k (내곽=1, 외곽=N)
df_multi["k"] = df_multi.groupby("system_id").cumcount() + 1

# 3-5) 계별 행성 수(편의 컬럼) 및 외곽부터의 순서
df_multi["n_in_system"] = df_multi.groupby("system_id")["planet_id"].transform("size")
df_multi["k_from_outer"] = df_multi["n_in_system"] - df_multi["k"] + 1

# 3-6) 확인 출력
print("✅ Step3 완료 — 다행성계만 필터 & 계 내부 정렬 및 k 부여")
print("다행성계 행 수:", len(df_multi))
print("계(unique) 수:", df_multi["system_id"].nunique())
print("\n미리보기(상위 10행):")
print(df_multi[[
    "system_id","planet_id","n_planets","P_days","R_earth","logP","logR","k","k_from_outer","n_in_system"
]].head(10))

# ============================================
# Step 4: 계 단위 분석 지표 계산 (system_summary.csv 저장)
#   - [Fix-1] logR가 상수이면 Spearman 계산 생략(경고 제거)
#   - [Fix-2] system_id를 인덱스로 그룹화해 DeprecationWarning 제거
# ============================================

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, iqr

# ---- (안전) 유효 반지름 / 로그반지름만 사용 ----
valid_r = df_multi["R_earth"].notna() & (df_multi["R_earth"] > 0)
valid_logr = df_multi["logR"].notna()
df_multi_valid = df_multi.loc[valid_r & valid_logr].copy()

# ---- [Fix-2] 그룹 키 컬럼 제외를 위해 인덱스로 승격 후 필요한 컬럼만 선택 ----
cols_needed = ["R_earth", "logR", "k", "n_in_system"]
dfg = (
    df_multi_valid
    .set_index("system_id")[cols_needed]   # system_id를 인덱스로
)

def summarize_system(g: pd.DataFrame) -> pd.Series:
    """
    g: 단일 시스템의 서브-DataFrame (index.name == system_id)
    """
    sys_id = g.index.name  # group key
    logR = g["logR"].to_numpy()
    R    = g["R_earth"].to_numpy()
    k    = g["k"].to_numpy()

    # --- 인접쌍 지표 ---
    if logR.size >= 2:
        dlogR     = np.diff(logR)
        ratioR    = R[1:] / R[:-1]
        dlogR_med = np.nanmedian(dlogR)
        dlogR_mean= np.nanmean(dlogR)
        ratio_med = np.nanmedian(ratioR)
        ratio_gmn = np.exp(np.nanmean(np.log(ratioR)))
    else:
        dlogR_med = dlogR_mean = ratio_med = ratio_gmn = np.nan

    # --- 산포 지표 ---
    std_logR = np.nanstd(logR, ddof=1) if logR.size >= 2 else np.nan
    iqr_logR = iqr(logR, nan_policy="omit") if logR.size >= 1 else np.nan

    # --- [Fix-1] 서열 경향: logR가 상수이면 Spearman 계산 생략(경고 방지) ---
    if logR.size >= 2 and np.nanstd(logR) > 0 and np.nanstd(k) > 0:
        rho, pval = spearmanr(k, logR, nan_policy="omit")
    else:
        rho, pval = np.nan, np.nan

    # n_planets: n_in_system 있으면 그 값, 없으면 길이로 대체
    if "n_in_system" in g:
        try:
            n_planets = int(pd.Series(g["n_in_system"]).iloc[0])
        except Exception:
            n_planets = int(len(g))
    else:
        n_planets = int(len(g))

    return pd.Series({
        "system_id": sys_id,
        "n_planets": n_planets,
        "n_pairs": max(int(logR.size - 1), 0),
        "delta_logR_median": dlogR_med,
        "delta_logR_mean": dlogR_mean,
        "ratio_median": ratio_med,
        "ratio_geomean": ratio_gmn,
        "std_logR": std_logR,
        "IQR_logR": iqr_logR,
        "spearman_rho_k_logR": rho,
        "spearman_p": pval,
    })

# ---- 요약 실행: 인덱스(level=0 == system_id) 기준 그룹화 ----
system_summary = (
    dfg.groupby(level=0, sort=True, group_keys=False)
       .apply(summarize_system)
       .reset_index(drop=True)
)

# ---- 저장 ----
out_path = os.path.join(os.path.dirname(DATA_PATH), "system_summary.csv")
system_summary.to_csv(out_path, index=False, encoding="utf-8-sig")

print("✅ Step4 완료 — 경고 제거 버전")
print("요약 계 수:", len(system_summary))
print("저장 위치:", out_path)
print("\n미리보기(상위 10행):")
print(system_summary.head(10))


# ============================================
# Step 5: 무작위 대조군(null) 생성 — 전역 반지름 셔플
#   - 주기순서(k)와 각 계의 길이는 유지
#   - 반지름만 전역 풀에서 무작위 재배정
#   - n<2인 계는 분석에서 제외(인접쌍 불가)
#   - (Fix) R_pool을 n>=2 계로만 구성해 길이 불일치 문제 제거
#   - 결과: null_iter_summary.csv 저장
# ============================================

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------- 5-0) 관측 집계치(참고용) ----------
def aggregate_over_systems(system_summary: pd.DataFrame) -> dict:
    """관측(per-system) 지표를 한 번 더 집계한 '한 숫자' 기준선."""
    def med(s):
        s = pd.Series(s, dtype=float).dropna()
        return float(np.median(s)) if len(s) else np.nan
    def meannz(s):
        s = pd.Series(s, dtype=float).dropna()
        return float(np.mean(s)) if len(s) else np.nan

    return {
        "obs_delta_logR_median__across_systems_median": med(system_summary["delta_logR_median"]),
        "obs_std_logR__across_systems_mean":            meannz(system_summary["std_logR"]),
        "obs_spearman_rho__across_systems_median":      med(system_summary["spearman_rho_k_logR"]),
    }

# 관측 집계치(있으면 붙임; 없어도 문제 없음)
try:
    obs_agg = aggregate_over_systems(system_summary)
except NameError:
    obs_agg = {}

# ---------- 5-1) 셔플 풀 구성: n>=2 계만 사용 ----------
# (중요) 인접쌍 계산이 가능한 계만 null에 포함해야
#       R_pool 길이 == 계별 길이 합 이 보장됨.
sys_ids     = []
sys_lengths = []
k_arrays    = []
R_list      = []

for sys_id, g in dfg.groupby(level=0, sort=True, group_keys=False):
    n = len(g)
    if n < 2:
        continue  # 인접쌍 불가 → null에서 제외
    sys_ids.append(sys_id)
    sys_lengths.append(n)
    k_arrays.append(g["k"].to_numpy())
    R_list.append(g["R_earth"].to_numpy())

# 전역 반지름 풀 (n>=2 계만 결합)
R_pool = np.concatenate(R_list) if len(R_list) else np.array([], dtype=float)
sys_lengths = np.asarray(sys_lengths, dtype=int)
n_systems = len(sys_ids)
total_planets = int(sys_lengths.sum())

# 안전 체크
assert len(R_pool) == total_planets, "풀 길이와 계별 길이 합이 일치해야 합니다."
assert n_systems == len(k_arrays) == len(sys_lengths), "리스트 길이 불일치"

print("✅ R_pool 준비 완료")
print("  사용 계 수(n>=2):", n_systems)
print("  R_pool 길이:", len(R_pool))
print("  계별 행성 합:", total_planets)

# ---------- 5-2) per-system 지표 계산 도우미 ----------
def metrics_for_system(R: np.ndarray, k: np.ndarray) -> tuple:
    """
    R: 반지름(>0), k: 1..n
    return: (delta_logR_median, std_logR, spearman_rho)
    """
    logR = np.log10(R)
    if logR.size >= 2:
        dlogR_med = float(np.median(np.diff(logR)))
        std_logR  = float(np.std(logR, ddof=1))
        # Spearman: logR 또는 k가 상수면 NaN 처리 (경고 회피)
        if np.nanstd(logR) > 0 and np.nanstd(k) > 0:
            rho, _p = spearmanr(k, logR, nan_policy="omit")
            rho = float(rho)
        else:
            rho = np.nan
    else:
        dlogR_med = np.nan
        std_logR  = np.nan
        rho       = np.nan
    return dlogR_med, std_logR, rho

# ---------- 5-3) 반복 설정 ----------
N_ITER = 200        # 시간되면 1000까지 가능
RANDOM_SEED = 42    # 재현성
rng = np.random.default_rng(RANDOM_SEED)

# ---------- 5-4) null 반복: 풀 셔플 → 계별 배정 → 지표 산출/집계 ----------
null_records = []
progress_every = max(1, N_ITER // 10)

for t in range(1, N_ITER + 1):
    # (a) 풀 셔플
    shuffled = R_pool.copy()
    rng.shuffle(shuffled)

    # (b) 계별 슬라이스 배정 + per-system 지표
    per_sys_dlogR_med = []
    per_sys_std_logR  = []
    per_sys_rho       = []

    pos = 0
    for n, k in zip(sys_lengths, k_arrays):
        R_assign = shuffled[pos:pos + n]
        pos += n
        dmed, sstd, rho = metrics_for_system(R_assign, k)
        per_sys_dlogR_med.append(dmed)
        per_sys_std_logR.append(sstd)
        per_sys_rho.append(rho)

    # (c) per-system 지표들을 한 번 더 집계(관측과 동일한 방식)
    def nz_median(x):
        x = pd.Series(x, dtype=float).dropna()
        return float(np.median(x)) if len(x) else np.nan
    def nz_mean(x):
        x = pd.Series(x, dtype=float).dropna()
        return float(np.mean(x)) if len(x) else np.nan

    null_records.append({
        "iter": t,
        "delta_logR_median__across_systems_median": nz_median(per_sys_dlogR_med),
        "std_logR__across_systems_mean":            nz_mean(per_sys_std_logR),
        "spearman_rho__across_systems_median":      nz_median(per_sys_rho),
    })

    if (t % progress_every) == 0:
        print(f"[Step5] 진행상황: {t}/{N_ITER}")

null_iter_summary = pd.DataFrame(null_records)

# (선택) 관측 집계치 열로 함께 기록
for k, v in (obs_agg or {}).items():
    null_iter_summary[k] = v

# ---------- 5-5) 저장 ----------
base_dir = os.path.dirname(DATA_PATH)
out_null = os.path.join(base_dir, "null_iter_summary.csv")
null_iter_summary.to_csv(out_null, index=False, encoding="utf-8-sig")

print("✅ Step5 완료 — null 분포 생성 및 저장")
print("저장 위치:", out_null)
print("\n미리보기:")
print(null_iter_summary.head(10))

# ============================================
# Step 6: 관측 vs null 비교 (p-value + 시각화)
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- 데이터 불러오기 ----
base_dir = os.path.dirname(DATA_PATH)
system_summary = pd.read_csv(os.path.join(base_dir, "system_summary.csv"))
null_iter_summary = pd.read_csv(os.path.join(base_dir, "null_iter_summary.csv"))

# ---- 비교할 지표 리스트 ----
metrics = [
    ("delta_logR_median__across_systems_median", "ΔlogR (median across systems)"),
    ("std_logR__across_systems_mean", "std(logR) (mean across systems)"),
    ("spearman_rho__across_systems_median", "Spearman ρ (median across systems)")
]

# ---- 관측치 불러오기 (obs_* 열) ----
obs_vals = {
    "delta_logR_median__across_systems_median": null_iter_summary["obs_delta_logR_median__across_systems_median"].iloc[0],
    "std_logR__across_systems_mean": null_iter_summary["obs_std_logR__across_systems_mean"].iloc[0],
    "spearman_rho__across_systems_median": null_iter_summary["obs_spearman_rho__across_systems_median"].iloc[0],
}

# ---- p-value 계산 함수 (단측/양측 모두) ----
def permutation_p(obs, null_values, alternative="two-sided"):
    null_values = pd.Series(null_values).dropna()
    if len(null_values) == 0 or pd.isna(obs):
        return np.nan
    if alternative == "greater":
        return (null_values >= obs).mean()
    elif alternative == "less":
        return (null_values <= obs).mean()
    else:  # two-sided
        # 두 쪽 꼬리: null 평균 기준이 아니라, 관측치가 null 분포 중앙에서 얼마나 극단적인지
        diff = abs(null_values - null_values.mean())
        obs_diff = abs(obs - null_values.mean())
        return (diff >= obs_diff).mean()

# ---- 지표별 비교 ----
results = []

for key, label in metrics:
    null_vals = null_iter_summary[key]
    obs = obs_vals[key]
    pval = permutation_p(obs, null_vals, alternative="two-sided")
    results.append((label, obs, np.mean(null_vals), np.std(null_vals), pval))

    # 히스토그램 시각화
    plt.hist(null_vals, bins=30, alpha=0.7, color="skyblue", edgecolor="k")
    plt.axvline(obs, color="red", linestyle="--", label=f"Observed = {obs:.3f}")
    plt.title(f"Null vs Observed — {label}")
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# ---- 결과 출력 ----
print("✅ Step6 완료 — 관측 vs null 비교")
print("지표별 결과 (관측값, null 평균±표준편차, permutation p-value):\n")
for label, obs, mean, std, p in results:
    print(f"{label:40s} Obs={obs:.3f}, NullMean={mean:.3f}±{std:.3f}, p={p:.4f}")


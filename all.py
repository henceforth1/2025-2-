# ============================================
# PSCompPars1.csv — STEP 1~5 올인원 (요청 반영판)
# - 파일명: PSCompPars1.csv
# - 밀도 그래프: x축 0~20으로 제한
# - 이심률 그래프: 전체(0~1) + 확대(0.05~0.5) 두 장
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# STEP 1: CSV 읽기 (주석 무시)
# -------------------------
path = r"C:\Users\woori\OneDrive - 서울과학고등학교\문서\김창우\설곽\과제연구\2025 2학기\파이썬\PSCompPars1.csv"
df_raw = pd.read_csv(path, comment="#", encoding="utf-8-sig")

# -------------------------
# STEP 2: 컬럼 리네임 (eccentricity/density 포함)
# -------------------------
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
    # 이심률/밀도 컬럼(버전별 변형명 대응)
    "pl_orbeccen":     "eccentricity",
    "pl_orbeccen1":    "eccentricity",
    "pl_dens":         "planet_density",
    "pl_dens1":        "planet_density",
}
df = df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns})
for _, alias in rename_map.items():
    if alias not in df.columns:
        df[alias] = np.nan

# -------------------------
# STEP 3: 단위 통일 (목성 → 지구)
# -------------------------
JUP_TO_EARTH_R = 11.209
JUP_TO_EARTH_M = 317.8
mask_R = df["R_earth"].isna() & df["R_jup"].notna()
df.loc[mask_R, "R_earth"] = df.loc[mask_R, "R_jup"] * JUP_TO_EARTH_R
mask_M = df["M_earth"].isna() & df["M_jup"].notna()
df.loc[mask_M, "M_earth"] = df.loc[mask_M, "M_jup"] * JUP_TO_EARTH_M

# -------------------------
# STEP 4: 라벨 & 파생변수
# -------------------------
df["class"] = "multi"
df.loc[df["n_planets"] == 1, "class"] = "single"
df["logP"] = np.where(df["P_days"] > 0, np.log10(df["P_days"]), np.nan)

# 핵심 분석용(주기/반지름/라벨이 있는 행만)
df_core = df.dropna(subset=["P_days", "R_earth", "class"]).copy()

print("✅ 준비 완료 — 전체:", df.shape, "/ 핵심(df_core):", df_core.shape)
print("단행성계(single):", (df["class"]=="single").sum(),
      "| 다행성계(multi):", (df["class"]=="multi").sum())

# -------------------------
# STEP 5: EDA (히스토그램 + 중앙값/IQR 요약표)
# -------------------------

def hist_overlay(data, col, bins=40, title=None, xlabel=None, ylabel=None, xlim=None):
    plt.figure(figsize=(7,4))
    for c in ["single", "multi"]:
        x = data.loc[data["class"]==c, col].dropna()
        if len(x) == 0:
            continue
        plt.hist(x, bins=bins, alpha=0.5, density=True, label=c)
    if xlabel: plt.xlabel(xlabel)
    if not ylabel:
        plt.ylabel("Probability Density")
    else:
        plt.ylabel(ylabel)
    if title: plt.title(title)
    if xlim is not None: plt.xlim(*xlim)
    plt.legend()
    plt.tight_layout()
    plt.show()

# (A) 주기(logP)
hist_overlay(df_core, "logP", bins=50,
             title="Distribution of Orbital Periods (log10)",
             xlabel="log10(Period [days])")

# (B) 반지름(R_earth)
hist_overlay(df_core, "R_earth", bins=50,
             title="Distribution of Planet Radii",
             xlabel="Radius [Earth radii]")

# (C) 밀도 — x축 0~20으로 제한
if df_core["planet_density"].notna().sum() > 0:
    df_den = df_core.dropna(subset=["planet_density"])
    hist_overlay(df_den, "planet_density", bins=5000,
                 title="Distribution of Planet Density",
                 xlabel="Density [g/cm³]",
                 xlim=(0, 20))

# (D) 이심률 — 전체(0~1) 그래프
if df_core["eccentricity"].notna().sum() > 0:
    df_ecc = df_core.dropna(subset=["eccentricity"])
    hist_overlay(df_ecc, "eccentricity", bins=100,
                 title="Distribution of Orbital Eccentricity (Full)",
                 xlabel="Eccentricity",
                 xlim=(0, 1))
    # 이심률 — 확대(0.05~0.3) 그래프
    hist_overlay(df_ecc, "eccentricity", bins=200,
                 title="Distribution of Orbital Eccentricity (Zoom 0.05–0.5)",
                 xlabel="Eccentricity",
                 xlim=(0.05, 0.5))

# (E) 중앙값 & IQR 요약표
def summary_table(data: pd.DataFrame, variables):
    rows = []
    for var in variables:
        if var not in data.columns:  # 없는 컬럼은 스킵
            continue
        for c in ["single", "multi"]:
            x = data.loc[data["class"]==c, var].dropna().values
            if len(x) == 0:
                rows.append({"variable": var, "class": c, "N": 0,
                             "median": np.nan, "q1": np.nan, "q3": np.nan})
            else:
                rows.append({"variable": var, "class": c, "N": len(x),
                             "median": np.nanmedian(x),
                             "q1": np.quantile(x, 0.25),
                             "q3": np.quantile(x, 0.75)})
    return pd.DataFrame(rows)

vars_to_summarize = ["logP", "R_earth", "eccentricity", "planet_density"]
summary = summary_table(df_core, vars_to_summarize)

print("\n✅ 중앙값 & IQR 요약표")
print(summary.pivot(index="variable", columns="class",
                    values=["N", "median", "q1", "q3"]))

# -------------------------
# STEP 6: 통계 검정 (KS, Mann–Whitney U)
#  - df_core를 입력으로 사용 (STEP 1~5 이후)
# -------------------------
from scipy import stats
import numpy as np

def compare_distributions(data, col):
    """single vs multi 분포 비교: KS + Mann–Whitney + 중앙값 요약"""
    x = data.loc[data["class"]=="single", col].dropna().values
    y = data.loc[data["class"]=="multi",  col].dropna().values
    if len(x)==0 or len(y)==0:
        print(f"⚠️ {col}: 데이터 부족 (single {len(x)}, multi {len(y)})")
        return
    ks_stat, ks_p = stats.ks_2samp(x, y, alternative="two-sided")
    u_stat, u_p   = stats.mannwhitneyu(x, y, alternative="two-sided")
    print(f"\n📊 {col}")
    print(f"  single N={len(x)}, multi N={len(y)}")
    print(f"  KS test:      stat={ks_stat:.3f}, p={ks_p:.3e}")
    print(f"  Mann-Whitney: U={u_stat:.0f}, p={u_p:.3e}")
    print(f"  medians: single={np.nanmedian(x):.3f}, multi={np.nanmedian(y):.3f}")

# 실행할 변수
vars_to_test = ["logP", "R_earth", "eccentricity", "planet_density"]

print("\n✅ STEP 6: 통계 검정 결과")
for var in vars_to_test:
    if var in df_core.columns:
        compare_distributions(df_core, var)


# -------------------------
# STEP 7: Hot/Warm Jupiter 탐색
# -------------------------

# Hot Jupiter 정의: 반지름 > 8 R⊕, 주기 < 10일
hot_mask = (df_core["R_earth"] > 8) & (df_core["P_days"] < 10)
df_hot = df_core[hot_mask]

# Warm Jupiter 정의 (선택): 반지름 > 8 R⊕, 주기 10~100일
warm_mask = (df_core["R_earth"] > 8) & (df_core["P_days"].between(10, 100))
df_warm = df_core[warm_mask]

# --- 단/다행성별 집계 --- (STEP 7)

print("\n✅ STEP 7: Hot/Warm Jupiter 분포")

def summarize_group(df_sub, label):
    counts = df_sub.groupby("class")["planet_id"].count()
    total = df_sub.shape[0]
    print(f"\n{label} (총 {total}개)")
    for c in ["single", "multi"]:
        n = counts.get(c, 0)
        frac = n / total if total > 0 else 0
        print(f"  {c:<6}: {n}개 ({frac:.2%})")

summarize_group(df_hot, "Hot Jupiter")
summarize_group(df_warm, "Warm Jupiter")

# --- 산점도 (logP vs R_earth) ---
plt.figure(figsize=(7,6))
colors = df_core["class"].map({"single":"red","multi":"blue"})
plt.scatter(df_core["logP"], df_core["R_earth"], alpha=0.4, c=colors, s=20)

# Hot Jupiter 경계선 표시
plt.axvline(np.log10(10), color="k", linestyle="--", label="P=10 days")
plt.axhline(8, color="k", linestyle=":", label="R=8 R⊕")

plt.xlabel("log10(Period [days])")
plt.ylabel("Radius [Earth radii]")
plt.title("Hot Jupiter Region (R>8 R⊕ & P<10 days)")
plt.legend()
plt.tight_layout()
plt.show()

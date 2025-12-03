import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================
# 0. 파일 경로 설정
# =========================================
MERGED_FILE = Path("merged_trackAB.json")  # step3에서 만든 파일
OUT_MERGED_WITH_FEATURES = Path("merged_trackAB_with_features.json")
OUT_STATS_CSV = Path("struct_stats_by_stability.csv")
FIG_DIR = Path("fig_structure")
FIG_DIR.mkdir(exist_ok=True)


# =========================================
# 1. merged_trackAB.json 불러오기
#    (TrackA 파라미터 + B_stable + reason 이미 포함)
# =========================================
with open(MERGED_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print("Merged shape:", df.shape)
print("Columns:", df.columns.tolist())


# =========================================
# 2. per-system 요약 feature 계산
#    (a, m, R, e, inc, deltas가 리스트 형태라고 가정)
# =========================================
def list_to_array_safe(x):
    """리스트/튜플이면 np.array로, 아니면 np.array([x])로."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x, dtype=float)
    return np.array([x], dtype=float)


def add_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    required_keys = ["a", "m", "R", "e", "inc", "deltas"]
    for key in required_keys:
        if key not in df.columns:
            raise KeyError(f"'{key}' 컬럼이 없습니다. trackA_candidates/merge 단계에서 포함됐는지 확인하세요.")

    mean_deltas = []
    min_deltas = []
    std_log_m = []
    std_log_R = []
    std_log_a = []
    mean_e = []
    mean_inc = []
    n_planets = []

    for _, row in df.iterrows():
        a = list_to_array_safe(row["a"])
        m = list_to_array_safe(row["m"])
        R = list_to_array_safe(row["R"])
        e = list_to_array_safe(row["e"])
        inc = list_to_array_safe(row["inc"])
        deltas = list_to_array_safe(row["deltas"])

        # 행성 수
        n_planets.append(len(a))

        # Δ 요약값
        mean_deltas.append(deltas.mean())
        min_deltas.append(deltas.min())

        # 로그 스케일 분산 (peas-in-a-pod 정도 보는 지표)
        log_m = np.log10(m)
        log_R = np.log10(R)
        log_a = np.log10(a)

        std_log_m.append(log_m.std())
        std_log_R.append(log_R.std())
        std_log_a.append(log_a.std())

        # 궤도요소 평균
        mean_e.append(e.mean())
        mean_inc.append(inc.mean())

    df = df.copy()
    df["n_planets"] = n_planets
    df["mean_delta"] = mean_deltas
    df["min_delta"] = min_deltas
    df["std_log_m"] = std_log_m       # 질량이 비슷할수록 ↓
    df["std_log_R"] = std_log_R       # 반지름이 비슷할수록 ↓
    df["std_log_a"] = std_log_a       # 궤도반지름 로그 분산
    df["mean_e"] = mean_e
    df["mean_inc"] = mean_inc

    return df


df = add_summary_features(df)

# feature 붙인 버전 저장 (나중에 또 쓸 수 있게)
df.to_json(OUT_MERGED_WITH_FEATURES, orient="records", indent=2, force_ascii=False)
print(f"요약 feature 포함 merged 데이터 저장 완료: {OUT_MERGED_WITH_FEATURES}")


# =========================================
# 3. 안정 / 불안정 분리
# =========================================
if "B_stable" not in df.columns:
    raise KeyError("'B_stable' 컬럼이 없습니다. merged_trackAB.json에 포함됐는지 확인하세요.")

stable = df[df["B_stable"] == True].copy()
unstable = df[df["B_stable"] == False].copy()

print(f"Stable N = {len(stable)}, Unstable N = {len(unstable)}")


# =========================================
# 4. 통계 요약표 만들기
# =========================================
cols_to_compare = [
    "mean_delta",
    "min_delta",
    "std_log_m",
    "std_log_R",
    "std_log_a",
    "mean_e",
    "mean_inc"
]

rows = []
for col in cols_to_compare:
    if col not in df.columns:
        print(f"⚠ 경고: {col} 컬럼이 없어 통계에서 제외됨.")
        continue

    row = {
        "feature": col,
        "stable_mean": stable[col].mean(),
        "stable_std": stable[col].std(),
        "stable_median": stable[col].median(),
        "unstable_mean": unstable[col].mean(),
        "unstable_std": unstable[col].std(),
        "unstable_median": unstable[col].median(),
    }
    rows.append(row)

stats_df = pd.DataFrame(rows)
stats_df.to_csv(OUT_STATS_CSV, index=False)
print(f"안정/불안정 구조 비교 통계 저장 완료: {OUT_STATS_CSV}")
print(stats_df)


# =========================================
# 5. 히스토그램 & 박스플롯
# =========================================
for col in cols_to_compare:
    if col not in df.columns:
        continue

    # 히스토그램
    plt.figure()
    plt.hist(stable[col].dropna(), bins=20, alpha=0.5, label="Stable")
    plt.hist(unstable[col].dropna(), bins=20, alpha=0.5, label="Unstable")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"{col} distribution: Stable vs Unstable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"hist_{col}_stable_vs_unstable.png", dpi=200)
    plt.close()

    # 박스플롯
    plt.figure()
    plt.boxplot(
        [stable[col].dropna(), unstable[col].dropna()],
        tick_labels=["Stable", "Unstable"]
    )
    plt.ylabel(col)
    plt.title(f"{col} boxplot: Stable vs Unstable")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"box_{col}_stable_vs_unstable.png", dpi=200)
    plt.close()

print(f"그림 저장 완료: {FIG_DIR} 폴더 아래 hist_*, box_* PNG 확인")

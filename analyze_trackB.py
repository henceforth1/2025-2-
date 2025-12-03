import json
import pandas as pd

# 1) 여기 파일 이름을 네 json 이름으로 바꿔줘
JSON_FILE = "trackB_results.json"   # 예시 이름

# ----------------------------------------------------
# 1. JSON 불러오기
# ----------------------------------------------------
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# data가 리스트[ {dict}, {dict}, ... ] 형태라고 가정
df = pd.DataFrame(data)

print("=== 컬럼 목록 확인 ===")
print(df.columns)
print()

# 전체 시스템 개수
n_total = len(df)
print(f"전체 시스템 개수: {n_total}")
print()

# ----------------------------------------------------
# 2. 안정 / 불안정 개수 세기
# ----------------------------------------------------

# 2-1. is_stable 같은 불리언 컬럼이 있는 경우
if "is_stable" in df.columns:
    n_stable = (df["is_stable"] == True).sum()
    n_unstable = (df["is_stable"] == False).sum()
    print("=== is_stable 기준 요약 ===")
    print(f"  안정(True)   : {n_stable}")
    print(f"  불안정(False): {n_unstable}")
    print()

# 2-2. status 컬럼에 'stable', 'unstable' 같은 문자열이 있는 경우
if "status" in df.columns:
    print("=== status 값 분포 ===")
    print(df["status"].value_counts(dropna=False))
    print()

# ----------------------------------------------------
# 3. 실패 이유(reason)별 개수 세기
#    예: large_da, close_encounter, eccentricity_growth 등
# ----------------------------------------------------
if "reason" in df.columns:
    print("=== 실패 이유(reason)별 개수 ===")
    print(df["reason"].value_counts(dropna=False))
    print()
else:
    print("reason 컬럼이 없어서 실패 이유는 집계할 수 없음.")


# ==========================
# 1. Track B 결과 불러오기
# ==========================
with open("trackB_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("=== TrackB 컬럼 ===")
print(df.columns)
print()

# 전체 개수
total_N = len(df)
print(f"전체 시스템 개수: {total_N}\n")


# ==========================
# 2. 안정계 필터링
# ==========================
stable_df = df[df["reason"] == "no_instability_detected"].copy()
unstable_df = df[df["reason"] != "no_instability_detected"].copy()

N_stable = len(stable_df)
N_unstable = len(unstable_df)

print("=== 안정/불안정 요약 ===")
print(f"안정 (no_instability_detected): {N_stable}")
print(f"불안정: {N_unstable}")
print(f"안정률: {N_stable/total_N:.3f} ({100*N_stable/total_N:.1f}%)\n")


# ==========================
# 3. 실패 이유(reason)별 개수
# ==========================
print("=== reason별 개수 ===")
reason_count = df["reason"].value_counts()
print(reason_count)
print()

print("=== reason별 비율(%) ===")
print((reason_count / total_N * 100).round(2))
print()


# ==========================
# 4. 안정 계 ID 목록
# ==========================
print("=== 안정 계 ID 목록 ===")
print(stable_df["id"].tolist())
print()


# ==========================
# 5. 불안정 계 ID 목록 (optional)
# ==========================
print("=== 불안정 계 ID 목록 ===")
print(unstable_df["id"].tolist())
print()

print("=== 2단계 분석 완료 ===")

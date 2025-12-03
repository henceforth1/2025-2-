import json
import pandas as pd

# ================================
# 1. TrackA 후보 불러오기
# ================================
with open("trackA_candidates.json", "r", encoding="utf-8") as f:
    dataA = json.load(f)

dfA = pd.DataFrame(dataA)
dfA = dfA.reset_index(drop=True)   # 안전하게 index 재정렬
dfA["idx"] = dfA.index             # 0~433까지 번호 부여

print("TrackA rows:", len(dfA))
print("TrackA columns:", dfA.columns.tolist())


# ================================
# 2. TrackB 결과 불러오기
# ================================
with open("trackB_results.json", "r", encoding="utf-8") as f:
    dataB = json.load(f)

dfB = pd.DataFrame(dataB)
dfB = dfB.reset_index(drop=True)
dfB["idx"] = dfB.index

print("TrackB rows:", len(dfB))
print("TrackB columns:", dfB.columns.tolist())


# ================================
# 3. 병합 (index 기반)
# ================================
dfM = pd.merge(dfA, dfB[['idx', 'B_stable', 'reason']], on='idx', how='inner')

print("Merged rows:", len(dfM))
print("Merged columns:", dfM.columns.tolist())
print(dfM.head())


# ================================
# 4. 저장
# ================================
dfM.to_json("merged_trackAB.json", orient="records", indent=2, force_ascii=False)
print("===> merged_trackAB.json 저장 완료")

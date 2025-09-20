#!/usr/bin/env python3
"""
UID 기능이 추가된 새로운 FraudPreprocessor 테스트
"""
import pandas as pd
import numpy as np
from src.fraud.preprocessing import FraudPreprocessor

# 샘플 데이터 생성
np.random.seed(42)
sample_data = pd.DataFrame({
    'TransactionID': range(10),
    'TransactionDT': np.random.randint(0, 86400*30, 10),  # 30일간의 초
    'TransactionAmt': np.random.uniform(10, 1000, 10),
    'ProductCD': np.random.choice(['W', 'C', 'R', 'H'], 10),
    'card1': np.random.randint(1000, 9999, 10),
    'card2': np.random.choice([100, 200, 300, None], 10),
    'addr1': np.random.randint(100, 999, 10),
    'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', None], 10),
    'D1': np.random.randint(0, 365, 10),  # D1 컬럼 추가
    'isFraud': np.random.choice([0, 1], 10)
})

print("=== 원본 데이터 ===")
print(sample_data.head())
print()

# UID 기능이 포함된 전처리기 테스트
print("=== UID 기능 포함 전처리기 테스트 ===")
preprocessor_uid = FraudPreprocessor(use_uid_features=True)

# Fit
X_raw = sample_data.drop(columns=['isFraud'])
preprocessor_uid.fit(X_raw)

# Transform
X_transformed = preprocessor_uid.transform(X_raw)
print(f"변환된 피처 개수: {X_transformed.shape[1]}")
print(f"새로운 피처들: {list(X_transformed.columns)}")
print()
print("UID 관련 피처들:")
uid_features = [col for col in X_transformed.columns if 'uid' in col.lower() or 'd1n' in col.lower()]
print(uid_features)
print()

if uid_features:
    print("UID 피처 샘플 값들:")
    print(X_transformed[uid_features].head())
    print()

# 기존 전처리기와 비교
print("=== 기존 전처리기와 비교 ===")
preprocessor_old = FraudPreprocessor(use_uid_features=False)
preprocessor_old.fit(X_raw)
X_old = preprocessor_old.transform(X_raw)

print(f"기존 피처 개수: {X_old.shape[1]}")
print(f"새 피처 개수: {X_transformed.shape[1]}")
print(f"추가된 피처 개수: {X_transformed.shape[1] - X_old.shape[1]}")
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# 1. 데이터 불러오기 및 전처리
df = pd.read_csv("merged_dataset/merged_dataset.csv")
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df = df.sort_values('open_time')


# 파생 피처 추가 (예: 5분 이동평균, 표준편차, 종가 변화율)
df['ma_close_5'] = df['close'].rolling(window=5).mean()
df['std_close_5'] = df['close'].rolling(window=5).std()
df['return_close'] = df['close'].pct_change()
df = df.dropna()

# 타겟 변수: 다음 캔들의 low와 high
df['next_low'] = df['low'].shift(-1)
df['next_high'] = df['high'].shift(-1)
df = df.dropna()

# 모델에 사용할 피처와 타겟
feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'ma_close_5', 'std_close_5', 'return_close']
features = df[feature_cols].values
targets = df[['next_low', 'next_high']].values

# 정규화: MinMaxScaler 사용
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
features_scaled = feature_scaler.fit_transform(features)
targets_scaled = target_scaler.fit_transform(targets)

# 슬라이딩 윈도우 기법으로 시퀀스 생성
def create_sequences(features, targets, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(targets[i+window_size])
    return np.array(X), np.array(y)

window_size = 10  # 예: 이전 10개 캔들을 사용하여 예측
X, y = create_sequences(features_scaled, targets_scaled, window_size)

# 학습/테스트 분리 (순서를 유지)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# GRU 모델 구성
model = Sequential()
model.add(GRU(50, activation='tanh', return_sequences=True, input_shape=(window_size, X.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(50, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(2))  # 출력: 다음 캔들의 low와 high

model.compile(optimizer='adam', loss='mae')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
loss = model.evaluate(X_test, y_test)
print("Test Loss (MAE):", loss)

# 모델과 스케일러 저장
model.save('model/gru_model.h5')
joblib.dump(feature_scaler, 'model/feature_scaler.save')
joblib.dump(target_scaler, 'model/target_scaler.save')
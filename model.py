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

# 2. 파생 피처 추가 (피처 엔지니어링)
# 예: 5분 이동평균, 5분 종가 표준편차, 종가 변화율(수익률)
df['ma_close_5'] = df['close'].rolling(window=5).mean()
df['std_close_5'] = df['close'].rolling(window=5).std()
df['return_close'] = df['close'].pct_change()

# Rolling 윈도우 사용으로 인해 발생하는 NaN 제거
df = df.dropna()

# 타겟 변수 생성: 다음 캔들의 low와 high
df['next_low'] = df['low'].shift(-1)
df['next_high'] = df['high'].shift(-1)
df = df.dropna()

# 3. 모델에 사용할 피처 설정
# 기존 피처: open, high, low, close, volume, amount
# 추가 피처: ma_close_5, std_close_5, return_close
feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'ma_close_5', 'std_close_5', 'return_close']
features = df[feature_cols].values
targets = df[['next_low', 'next_high']].values

# 4. 정규화 (MinMaxScaler)
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
features_scaled = feature_scaler.fit_transform(features)
targets_scaled = target_scaler.fit_transform(targets)

# 5. 슬라이딩 윈도우 기법으로 시퀀스 생성
def create_sequences(features, targets, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(targets[i+window_size])
    return np.array(X), np.array(y)

window_size = 10  #이전 10개 캔들의 데이터를 사용하여 다음 캔들을 예측
X, y = create_sequences(features_scaled, targets_scaled, window_size)

# 6. 학습/테스트 데이터 분리 (시간 순서를 유지)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 7. GRU 모델 구성
model = Sequential()
model.add(GRU(50, activation='tanh', return_sequences=True, input_shape=(window_size, X.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(50, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(2))  # 출력: 다음 캔들의 low와 high

model.compile(optimizer='adam', loss='mae')
model.summary()

# 8. 모델 학습
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 9. 모델 평가
loss = model.evaluate(X_test, y_test)
print("Test Loss (MAE):", loss)

# 10. 마지막 테스트 시퀀스를 활용한 예측
pred_scaled = model.predict(X_test[-1][np.newaxis, :, :])
pred = target_scaler.inverse_transform(pred_scaled)
print("예측된 다음 캔들의 low와 high:", pred)

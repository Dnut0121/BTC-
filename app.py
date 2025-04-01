from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# 저장된 GRU 모델과 스케일러 불러오기
model = tf.keras.models.load_model('gru_model.h5')
feature_scaler = joblib.load('feature_scaler.save')
target_scaler = joblib.load('target_scaler.save')

# 모델이 입력받는 시퀀스의 길이와 피처 개수 (예제에서는 window_size=10, 피처 9개)
window_size = 10
num_features = 9

# 예시: 임의의 입력 데이터를 생성 (실제 웹에서는 사용자 입력 또는 업로드 파일로 처리 가능)
# 여기서는 간단히 0으로 채워진 데이터를 사용합니다.
dummy_sequence = np.zeros((window_size, num_features))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 실제 서비스에서는 request.form 또는 request.files 로 입력 데이터를 받아 전처리 후 예측합니다.
    # 여기서는 예시로 dummy_sequence 를 사용합니다.
    input_seq = np.array(dummy_sequence)
    input_seq = input_seq[np.newaxis, ...]  # shape: (1, window_size, num_features)

    pred_scaled = model.predict(input_seq)
    pred = target_scaler.inverse_transform(pred_scaled)

    result = {
        'predicted_next_low': float(pred[0][0]),
        'predicted_next_high': float(pred[0][1])
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

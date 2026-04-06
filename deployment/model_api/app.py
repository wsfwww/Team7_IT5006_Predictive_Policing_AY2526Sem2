from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 加载模型
model = joblib.load('lgbm_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return "IT5006 Phase 3 Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. 接收前端传入的 JSON 数据
        data = request.get_json()
        
        # 2. 转换为 DataFrame (必须保证输入的 43 个特征顺序与你训练时 X_train 的列一致)
        df = pd.DataFrame([data])
        
        # 3. 预测 (返回逮捕概率和 0/1 标签)
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return jsonify({
            'success': True,
            'prediction_label': int(prediction),
            'arrest_probability': float(probability)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
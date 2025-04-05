import argparse
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model, model_from_json

def load_lasso_model(lasso_path):
    return joblib.load(lasso_path)

def load_keras_model(model_path, weights_path=None):
    if model_path.endswith('.h5') and weights_path is None:
        return load_model(model_path)
    elif model_path.endswith('.json') and weights_path:
        with open(model_path, 'r') as f:
            model_json = f.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        return model
    else:
        raise ValueError("Unsupported model format: use .h5 or .json with weights")

def main():
    parser = argparse.ArgumentParser(
        description="Use LassoCV for feature selection and a TensorFlow model for prediction."
    )
    
    # 数据和输出路径
    parser.add_argument('-i', '--input_csv', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('-o', '--output_csv', type=str, required=True,
                        help='Path to save the prediction result CSV')
    
    # 模型路径
    parser.add_argument('-lm', '--lasso_model', type=str, required=True,
                        help='Path to trained LassoCV model (.pkl)')
    parser.add_argument('-nm', '--nn_model', type=str, required=True,
                        help='Path to TensorFlow model (.h5 or .json)')
    parser.add_argument('-w', '--weights_path', type=str, default=None,
                        help='Path to weights if using .json model')
    
    # 标签列和需要排除的列
    parser.add_argument("-l", "--label_column", type=str, required=True, 
                        help="Name of the label column in the input CSV file.")

    args = parser.parse_args()

    # Step 1: 读取数据
    data = pd.read_csv(args.input_csv)
    labels = data[args.label_column].values
    features = data.drop(columns=[args.label_column]).values

    # Step 3: 应用Lasso进行特征选择
    lasso_model = load_lasso_model(args.lasso_model)
    X_selected = lasso_model.transform(features)

    # Step 4: 调整GRU输入形状（samples, timesteps, features）
    X_selected = np.expand_dims(X_selected, axis=1)

    # Step 5: 加载模型 
    model = load_keras_model(args.nn_model, args.weights_path)

    # Step 6: 预测
    predictions = model.predict(X_selected).flatten()

    # Step 7: 保存预测结果
    pd.DataFrame({'prediction': predictions}).to_csv(args.output_csv, index=False)
    print(f"✅ Prediction results (only prediction column) saved to {args.output_csv}")

if __name__ == '__main__':
    main()

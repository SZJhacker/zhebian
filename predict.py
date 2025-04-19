#!/usr/bin/env python3
# -*- coding=utf-8 -*-

#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
predict.py - 使用zhebian模型进行预测  
"""  

import argparse  
import pandas as pd  
import numpy as np  
import joblib  
from tensorflow.keras.models import load_model, model_from_json  
# 导入自定义组件  
from zhebian_utils import SoftMask, custom_objects  

def predict(model_prefix, input_file, output_file, label_column):  
    """使用保存的zhebian模型进行预测"""  
    # 加载数据  
    data = pd.read_csv(input_file)  
    if label_column in data.columns:  
        features = data.drop(columns=[label_column]).values  
    else:  
        print(f"警告: 未找到标签列 '{label_column}'，使用所有列作为特征")  
        features = data.values  
    
    # 加载特征选择器  
    print(f"加载特征选择器: {model_prefix}_selector.joblib")  
    selector = joblib.load(f"{model_prefix}_selector.joblib")  
    
    # 应用特征选择  
    X_selected = selector.transform(features)  
    
    # 调整输入形状为(samples, timesteps, features)  
    X_selected = np.expand_dims(X_selected, axis=1)  
    
    # 加载模型  
    print(f"加载模型: {model_prefix}_model.h5")  
    model = load_model(f"{model_prefix}_model.h5", custom_objects=custom_objects)  
    
    # 预测  
    print("执行预测...")  
    predictions = model.predict(X_selected).flatten()  
    
    # 保存结果  
    pd.DataFrame({'prediction': predictions}).to_csv(output_file, index=False)  
    print(f"✅ 预测结果已保存至: {output_file}") 

def main():  
    parser = argparse.ArgumentParser(  
        description="Use LassoCV for feature selection and a zhebian model for prediction."  
    )  
    
    # 数据和输出路径  
    parser.add_argument('-i', '--input_csv', type=str, required=True,  
                        help='Path to input CSV file')  
    parser.add_argument('-o', '--output_csv', type=str, required=True,  
                        help='Path to save the prediction result CSV')  
    
    # 模型路径  
    parser.add_argument('-m', '--model_prefix', type=str, required=True,  
                        help='Prefix for model files'),  
    # 标签列和需要排除的列  
    parser.add_argument("-l", "--label_column", type=str, required=True,   
                        help="Name of the label column in the input CSV file.")  

    args = parser.parse_args()  

     
    predict(args.model_prefix, args.input_csv, args.output_csv, args.label_column) 
if __name__ == '__main__':  
    main()
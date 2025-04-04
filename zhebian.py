#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input, MultiHeadAttention, Add, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Define LassoGRU framework
def create_model(input_shape, hidden_dim=64, num_heads=4, ff_dim=128, dropout=0.1):
    inputs = Input(shape=input_shape)
    
    # GRU 提取特征
    x = GRU(hidden_dim, return_sequences=True)(inputs)
    
    # GRU第一层 GRU 生成最后时间步特征
    gru_last_step = GRU(hidden_dim, return_sequences=False)(x)
    
    # QKV 生成
    q = Dense(hidden_dim)(x)
    k = Dense(hidden_dim)(x)
    v = Dense(hidden_dim)(x)
    
    # Transformer 自注意力
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim // num_heads )(q, k, v)
    attention = Dropout(dropout)(attention)
    attention_out = Add()([x, attention])  # 残差连接
    attention_out = LayerNormalization()(attention_out)  # 归一化
        
    alpha = Dense(1, activation="sigmoid")(gru_last_step)  # 计算权重
    fused = alpha * gru_last_step + (1 - alpha) * GlobalAveragePooling1D()(attention_out)
    
    final_feature = GRU(hidden_dim, return_sequences=False)(tf.expand_dims(fused, axis=1))

    # 全连接层
    x = Dense(ff_dim, activation="relu")(final_feature)
    outputs = Dense(1)(x)  # 回归任务
    model = Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Reshape features for GRU input
def reshape_for_gru(data):
    return np.reshape(data, (data.shape[0], 1, -1)).astype(np.float32)

# Main function for feature selection and model training
def train_model(input_csv, label_column, output_model_path, validation_split=0.1):
    # Load and preprocess data
    data = pd.read_csv(input_csv)
    labels = data[label_column].values
    features = data.drop(columns=[label_column]).values

    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    # Lasso feature selection
    lasso = LassoCV(cv=5, n_jobs=-1)
    lasso.fit(features_scaled, labels)
    selector = SelectFromModel(estimator=lasso, prefit=True)
    features_selected = selector.transform(features_scaled)

    # Save scaler and selector for reuse
    # joblib.dump(scaler, f"{output_model_path}_scaler.joblib")
    joblib.dump(selector, f"{output_model_path}_selector.joblib")

    # Prepare data for GRU
    features_gru = reshape_for_gru(features_selected)

    # Create and train GRU model
    model = create_model(input_shape=(1, features_selected.shape[1]))
    callback = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min',restore_best_weights=True)
    model.fit(features_gru, labels, epochs=100, validation_split=validation_split, callbacks=[callback])

    # Save trained model
    model.save(f"{output_model_path}_nn.h5")
    print(f"Model saved to {output_model_path}.h5")

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="LassoGRU: A framework for feature selection and regression")
    parser.add_argument("-i", "--input_csv", type=str, required=True, 
                        help="Path to the input CSV file (must contain features and one label column).")
    parser.add_argument("-l", "--label_column", type=str, required=True, 
                        help="Name of the label column in the input CSV file.")
    parser.add_argument("-o", "--output_model_path", type=str, required=True, 
                        help="Path prefix for saving the trained model and preprocessing tools.")
    parser.add_argument("-v", "--validation_split", type=float, default=0.2, help="Fraction of data to use for validation (default: 0.2).")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_model_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Train the model
    train_model(input_csv=args.input_csv, 
                label_column=args.label_column, 
                output_model_path=args.output_model_path,
                validation_split=args.validation_split)

if __name__ == "__main__":
    main()
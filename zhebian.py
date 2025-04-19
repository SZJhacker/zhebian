#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import argparse  
import numpy as np  
import pandas as pd  
import tensorflow as tf  
from tensorflow.keras.layers import (Input, GRU, Dense, MultiHeadAttention,   
                                    Dropout, Add, LayerNormalization,   
                                    GlobalAveragePooling1D, Concatenate)  
from tensorflow.keras.models import Model  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.callbacks import EarlyStopping  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.linear_model import LassoCV  
from sklearn.feature_selection import SelectFromModel  
import os  
import joblib  

from zhebian_utils import SoftMask 

def create_model(input_shape, hidden_dim=64, num_heads=4, ff_dim=128, dropout=0.1, mask_mode="none"):  
    inputs = Input(shape=input_shape)  
    masked_inputs = SoftMask(num_features=input_shape[-1], mode=mask_mode, temperature=10.0, l1_reg=0.01)(inputs)  
    
    x = GRU(hidden_dim, return_sequences=True)(masked_inputs)  
    x = LayerNormalization()(x)  
    
    gru_last_step = GRU(hidden_dim, return_sequences=False)(x)  
    
    q = Dense(hidden_dim)(x)  
    k = Dense(hidden_dim)(x)  
    v = Dense(hidden_dim)(x)  
    
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim // num_heads)(q, k, v)  
    attention = Dropout(dropout)(attention)  
    attention_out = Add()([x, attention])  
    attention_out = LayerNormalization()(attention_out)  
    
    attention_pool = GlobalAveragePooling1D()(attention_out)  
    gate = Dense(hidden_dim, activation="softsign")(Concatenate()([gru_last_step, attention_pool]))  
    merged = gate * gru_last_step + (1 - gate) * attention_pool  
    
    final_feature = GRU(hidden_dim, return_sequences=False)(tf.expand_dims(merged, axis=1))  
    
    x = Dense(ff_dim, activation="relu")(final_feature)  
    x = LayerNormalization()(x)  
    outputs = Dense(1)(x)  
    
    model = Model(inputs, outputs)  
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])  
    return model  

def reshape_for_gru(data):  
    return np.reshape(data, (data.shape[0], 1, -1)).astype(np.float32)  

def preprocess_data(data_path, label_column, output_path):  
    """Data preprocessing and feature selection"""  
    # Load data  
    data = pd.read_csv(data_path)  
    
    # Separate features and labels  
    labels = data[label_column].values  
    features = data.drop(columns=[label_column])  
    
    # Data scaling  
    scaler = MinMaxScaler(feature_range=(0, 1))  
    X_scaled = scaler.fit_transform(features)  
    
    # Feature selection using SelectFromModel with LassoCV  
    lasso = LassoCV(cv=5, n_jobs=-1)  
    selector = SelectFromModel(estimator=lasso)  
    X_scaled_select = selector.fit_transform(X_scaled, labels)  
    
    # If no features are selected, use all features  
    if X_scaled_select.shape[1] == 0:  
        print("No features selected by Lasso, using all features.")  
        X_scaled_select = X_scaled  
    
    # Save the preprocessing models  
    joblib.dump(scaler, f"{output_path}_scaler.joblib")  
    joblib.dump(selector, f"{output_path}_selector.joblib")  
    
    # Prepare data for GRU model  
    features_gru = reshape_for_gru(X_scaled_select)  
    
    return features_gru, labels  

def train_model(features_gru, labels, input_shape, output_path, epochs=100,   
                validation_split=0.2, hidden_dim=64, num_heads=4,   
                ff_dim=128, dropout=0.1, mask_mode="none"):  
    """Train and save model"""  
    # Create model  
    model = create_model(  
        input_shape=input_shape,  
        hidden_dim=hidden_dim,  
        num_heads=num_heads,  
        ff_dim=ff_dim,  
        dropout=dropout,  
        mask_mode=mask_mode  
    )  
    
    # Early stopping callback  
    callback = EarlyStopping(  
        monitor='val_loss',  
        patience=10,  
        verbose=1,  
        mode='min',  
        restore_best_weights=True  
    )  
    
    # Train model  
    history = model.fit(  
        features_gru,   
        labels,   
        epochs=epochs,   
        validation_split=validation_split,   
        callbacks=[callback],  
        verbose=1  
    )  
    
    # Save model  
    model.save(f"{output_path}_model.h5")  
    
    return model, history  

def main():  
    parser = argparse.ArgumentParser(description="GRU-Transformer Model Training Tool")  
    
    # Required parameters  
    parser.add_argument("-i", "--input_file", type=str, required=True,  
                      help="Input CSV file path")  
    parser.add_argument("-l", "--label_column", type=str, required=True,  
                      help="Label column name")  
    
    # Optional parameters  
    # parser.add_argument("-o", "--output_dir", type=str, default=".",  
    #                   help="Output directory for saving models (default: current directory)")  
    parser.add_argument("-o", "--output_model", type=str, default=None,  
                      help="Prefix for output model files (default: input filename without extension)")  
    parser.add_argument("-hd", "--hidden_dim", type=int, default=64,  
                      help="Hidden layer dimension (default: 64)")  
    parser.add_argument("-nh", "--num_heads", type=int, default=4,  
                      help="Number of attention heads (default: 4)")  
    parser.add_argument("-fd", "--ff_dim", type=int, default=128,  
                      help="Feed-forward network dimension (default: 128)")  
    parser.add_argument("-d", "--dropout", type=float, default=0.1,  
                      help="Dropout rate (default: 0.1)")  
    parser.add_argument("-m", "--mask_mode", type=str, default="none",   
                      choices=["none", "softmax", "sigmoid", "relu"],  
                      help="SoftMask mode (default: none)")  
    parser.add_argument("-e", "--epochs", type=int, default=100,  
                      help="Training epochs (default: 100)")  
    parser.add_argument("-vs", "--validation_split", type=float, default=0.2,  
                      help="Validation set ratio (default: 0.2)")  
    
    args = parser.parse_args()  
    
    # Set model prefix if not provided  
    if args.output_model is None:  
        args.output_model = os.path.splitext(os.path.basename(args.input_file))[0]  
    
    # Create full output path  
    output_path = args.output_model  
    
    print(f"Processing data: {args.input_file}")  
    features_gru, labels = preprocess_data(  
        args.input_file,   
        args.label_column,  
        output_path  
    )  
    
    input_shape = (features_gru.shape[1], features_gru.shape[2])  
    print(f"Input shape: {input_shape}")  
    
    print("Training model...")  
    model, history = train_model(  
        features_gru,   
        labels,   
        input_shape,  
        output_path,  
        epochs=args.epochs,  
        validation_split=args.validation_split,  
        hidden_dim=args.hidden_dim,  
        num_heads=args.num_heads,  
        ff_dim=args.ff_dim,  
        dropout=args.dropout,  
        mask_mode=args.mask_mode  
    )  
    
    # Save label column information for future prediction  
    joblib.dump({  
        'label_column': args.label_column  
    }, f"{output_path}_label_info.joblib")  
    
    print(f"Model and preprocessors saved with prefix: {output_path}")  
    print(f"Files saved:")  
    print(f"  - {output_path}_model.h5 (Neural network model)")  
    print(f"  - {output_path}_scaler.joblib (Data scaler)")  
    print(f"  - {output_path}_selector.joblib (Feature selector)")  
    print(f"  - {output_path}_label_info.joblib (Label information)")  

if __name__ == "__main__":  
    main()
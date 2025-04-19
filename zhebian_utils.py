#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
zhebian_utils.py - 包含zhebian模型所需的自定义组件  
"""  

import tensorflow as tf  
import numpy as np  

class SoftMask(tf.keras.layers.Layer):  
    def __init__(self, num_features, mode="none", temperature=10.0, l1_reg=0.01, **kwargs):  
        # 添加 **kwargs 以接收和传递额外参数（如 name）  
        super(SoftMask, self).__init__(**kwargs)  
        self.mode = mode  
        self.num_features = num_features  
        self.l1_reg = l1_reg  
        
        self.log_temperature = self.add_weight(  
            shape=(),  
            initializer=tf.constant_initializer(np.log(temperature)),  
            trainable=True,  
            name="log_temperature"  
        )  
        
        self.mask_weights = self.add_weight(  
            shape=(num_features,),  
            initializer="ones",  
            trainable=True,  
            regularizer=tf.keras.regularizers.l1(l1_reg),  
            name="soft_mask_weights"  
        )  
        
    def call(self, inputs):  
        temperature = tf.exp(self.log_temperature)  
        temperature = tf.maximum(temperature, 1e-7)  
        
        if self.mode == "softmax":  
            mask = tf.nn.softmax(self.mask_weights / temperature)  
        elif self.mode == "sigmoid":  
            mask = tf.nn.sigmoid(self.mask_weights / temperature)  
        elif self.mode == "relu":  
            mask = tf.nn.relu(self.mask_weights / temperature)  
        else:  # "none" mode  
            mask = self.mask_weights  
            
        return inputs * mask  
    
    def get_config(self):  
        config = super().get_config()  
        config.update({  
            "num_features": self.num_features,  
            "mode": self.mode,  
            "temperature": tf.math.exp(self.log_temperature).numpy(),  
            "l1_reg": self.l1_reg  
        })  
        return config  

# 自定义对象字典，加载模型时使用  
custom_objects = {  
    "SoftMask": SoftMask  
}
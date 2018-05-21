# !/usr/bin/env bash
nohup python -u ApplicationTrainFeaturesV1.py '/data/finupcard_python/jyl/kaggle/Home_Credit_Default_Risk/raw_data' '/data/finupcard_python/jyl/kaggle/Home_Credit_Default_Risk/feature_data' 'train_feature_df.csv' > log1.out 2>&1 &
nohup python -u ApplicationTrainFeaturesV2.py '/data/finupcard_python/jyl/kaggle/Home_Credit_Default_Risk/raw_data' '/data/finupcard_python/jyl/kaggle/Home_Credit_Default_Risk/feature_data' 'test_feature_df.csv' > log2.out 2>&1 &

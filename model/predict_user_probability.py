import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from scipy.sparse import csr_matrix
import pyspark.ml.feature as E
import pyspark.sql.functions as F

# 初始化Spark环境
spark = SparkSession.builder \
    .appName("UserPrediction") \
    .master("local[4]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# 特征定义（与训练时保持一致）
n_host_cols_list = [
    'fre_all_host_10', 'fre_all_host_20', 'fre_all_host_30',
    'fre_ins_host_10', 'fre_ins_host_20', 'fre_ins_host_30',
    # ... 其他特征（与您的训练代码中的定义完全一致）
]

numerical_features = ['send_insure_30', 'send_insure_60', 'send_insure_90', 'send_insure_180',
                      'days_since_send_insure_date', 'send_loan_30', 'send_loan_60', 'send_loan_90', 'send_loan_180',
                      # ... 其他数值特征
                      ] + n_host_cols_list + ['duration_30', 'duration_60', 'duration_90', 'duration_120',
                        'interact_30', 'interact_60', 'interact_90', 'interact_120',
                        # ... 其他特征
                        ]

categorical_features = ['phone_model', 'browser_family', 'os_family', 'device_brand', 'city_code', 'province_code',
                        'sex', 'hashouse', 'social', 'overdue', 'tax', 'married', 'benke', 'kid', 'income', 'consumption', 'shebao']

text_features = ['host_0009', 'host_1019', 'host_2029', 'host_date_0009', 'host_date_1019', 'host_date_2029',
                 # ... 其他文本特征
                 ] + ['keypress_30', 'keypress_60', 'keypress_90', 'keypress_120',
                   'rule_name_30', 'rule_name_60', 'rule_name_90', 'rule_name_120',
                   'semantic_30', 'semantic_60', 'semantic_90', 'semantic_120'] + ['model_value']

def create_preprocessing_pipeline():
    """
    创建与训练时相同的数据预处理管道
    """
    string_indexers = [E.StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid='keep') for col in categorical_features]
    one_hot_encoders = [E.OneHotEncoder(inputCols=[col + "_index"], outputCols=[col + "_ohe"], handleInvalid='keep') for col in categorical_features]
    tokenizers = [E.Tokenizer(inputCol=col, outputCol=col + "_token") for col in text_features]
    hashing_tfs = [E.HashingTF(inputCol=col + "_token", outputCol=col + "_hash", numFeatures=16) for col in text_features]
    idfs = [E.IDF(inputCol=col + "_hash", outputCol=col + "_tfidf") for col in text_features]
    median_imputer = E.Imputer(inputCols=['new_age', 'active_duration', 'avg_price'], 
                              outputCols=['new_age', 'active_duration', 'avg_price'], 
                              strategy="median", missingValue=0)
    vector_assembler = E.VectorAssembler(
        inputCols=numerical_features + [col + "_ohe" for col in categorical_features] + [col + "_tfidf" for col in text_features],
        outputCol="features",
        handleInvalid='keep'
    )
    
    pipeline = Pipeline(stages=string_indexers + one_hot_encoders + tokenizers + hashing_tfs + idfs + [median_imputer, vector_assembler])
    return pipeline

def load_trained_model():
    """
    加载训练好的XGBoost模型
    """
    with open("best_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict_user_probability(user_data_df, uid_column='uid'):
    """
    预测用户概率
    
    参数:
    user_data_df: pandas DataFrame，包含用户数据和uid列
    uid_column: 用户ID列名，默认为'uid'
    
    返回:
    pandas DataFrame，包含uid和预测概率
    """
    # 1. 将pandas DataFrame转换为Spark DataFrame
    spark_df = spark.createDataFrame(user_data_df)
    
    # 2. 创建预处理管道（需要使用训练时保存的管道模型）
    # 注意：实际使用时应该加载训练时保存的pipeline_model
    try:
        # 尝试加载保存的pipeline模型
        pipeline_model = Pipeline.load("model/pipeline_model")
    except:
        # 如果没有保存的pipeline模型，则重新创建（不推荐，因为可能导致不一致）
        print("警告：未找到保存的pipeline模型，正在重新创建...")
        pipeline = create_preprocessing_pipeline()
        pipeline_model = pipeline.fit(spark_df)
    
    # 3. 数据预处理
    processed_data = pipeline_model.transform(spark_df)
    
    # 4. 提取特征向量
    features_df = processed_data.select(uid_column, "features").toPandas()
    
    # 5. 准备XGBoost输入数据
    # 将Spark的向量转换为scipy稀疏矩阵格式
    features_list = features_df['features'].tolist()
    rows, cols, data = [], [], []
    for row_idx, vec in enumerate(features_list):
        # 获取非零元素的索引和值
        indices = vec.indices
        values = vec.values
        rows.extend([row_idx] * len(indices))
        cols.extend(indices)
        data.extend(values)
    
    X = csr_matrix((data, (rows, cols)), shape=(len(features_list), len(features_list[0].indices)))
    user_ids = features_df[uid_column].values
    
    # 6. 创建DMatrix
    dmatrix = xgb.DMatrix(X)
    
    # 7. 加载模型并预测
    model = load_trained_model()
    predictions = model.predict(dmatrix)
    
    # 8. 返回结果
    result_df = pd.DataFrame({
        uid_column: user_ids,
        'probability': predictions
    })
    
    return result_df

# 使用示例
if __name__ == "__main__":
    # 假设您有新的用户数据，包含uid列
    # user_data = pd.read_csv("new_user_data.csv")  # 包含uid和所有特征列
    
    # 示例数据结构
    sample_data = {
        'uid': ['user_001', 'user_002', 'user_003'],
        'send_insure_30': [1, 0, 2],
        'send_insure_60': [2, 1, 3],
        # ... 其他所有特征列
        'phone_model': ['iPhone', 'Samsung', 'Huawei'],
        'browser_family': ['Safari', 'Chrome', 'Chrome'],
        # ... 确保包含所有训练时使用的特征
    }
    
    user_data = pd.DataFrame(sample_data)
    
    # 预测用户概率
    predictions = predict_user_probability(user_data, uid_column='uid')
    
    print("用户预测结果：")
    print(predictions)
    
    # 可以根据概率阈值进行分类
    predictions['predicted_class'] = (predictions['probability'] > 0.5).astype(int)
    print("\n包含分类结果：")
    print(predictions)
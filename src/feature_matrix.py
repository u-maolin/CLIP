import pandas as pd

def create_feature_matrix(df_snv, df_patient, features):
    """
    生成特征矩阵。
    """
    # 合并SNV、CNV和患者数据
    feature_matrix = pd.merge(snv_data, patient_info, on='SampleID', how='inner')
    feature_matrix = pd.merge(feature_matrix, cnv_data, on='SampleID', how='inner')
    # 只保留指定的特征
    return feature_matrix[features]


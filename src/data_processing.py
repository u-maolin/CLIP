import pandas as pd
import numpy as np

def read_snv_data(file_path):
    """
    读取SNV数据并进行预处理。
    """
    snv_data = pd.read_csv(file_path, sep='\t')
    
    # 处理数据（如log转换等）
    snv_data['VAF'] = snv_data['Percent.mutant.allele']
    snv_data['sum_log10_frag_size_enrich_score'] = snv_data['sum_log10_frag_size_enrich_score'].clip(-10, 10)
    snv_data['germlinebg_Bayesian_pval'] = snv_data['germlinebg_Bayesian_pval'].clip(lower=1e-5)
    snv_data['cfdnabg_Bayesian_pval'] = snv_data['cfdnabg_Bayesian_pval'].clip(lower=1e-5)
    
    # 处理潜在的空值
    snv_data.fillna(0, inplace=True)
    
    return snv_data

def preprocess_patient_info(file_path):
    """
    读取患者信息并进行必要的预处理。
    """
    patient_info = pd.read_csv(file_path, sep='\t')
    
    # 假设处理患者的样本ID，特征等
    return patient_info

def read_cnv_data(file_path_5mb, file_path_gistic):
    """
    读取并处理CNV数据。
    """
    cnv_5mb_data = pd.read_csv(file_path_5mb, sep='\t')
    cnv_gistic_data = pd.read_csv(file_path_gistic, sep='\t')

    # 合并两个CNV数据集
    cnv_data = pd.merge(cnv_5mb_data, cnv_gistic_data, on='SampleID', suffixes=('_5mb', '_gistic'))

    return cnv_data

def generate_feature_matrix(snv_data, cnv_data, patient_info, features):
    """
    根据SNV、CNV和患者信息生成特征矩阵。
    """
    # 合并SNV、CNV数据和患者信息
    feature_matrix = pd.merge(snv_data, patient_info, on='SampleID', how='inner')
    feature_matrix = pd.merge(feature_matrix, cnv_data, on='SampleID', how='inner')

    # 只保留指定的特征
    feature_matrix = feature_matrix[features]

    return feature_matrix

def log_transform_columns(df, columns):
    """
    对指定列进行log10转换，处理为0的情况。
    """
    for col in columns:
        df[col] = df[col].apply(lambda x: np.log10(x) if x > 0 else -10)
    
    return df


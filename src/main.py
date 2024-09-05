import argparse
from data_processing import read_snv_data,read_cnv_data, preprocess_patient_info, generate_feature_matrix
from model_training import train_lasso, train_naive_bayes, train_decision_tree, predict_model, train_test_split_data
from validation import cross_validate_model, plot_roc_curve

def main():
    parser = argparse.ArgumentParser(description="Early-stage cancer prediction tool.")
    parser.add_argument('--snv_file', type=str, required=True, help='Path to the SNV data file.')
    parser.add_argument('--cnv_file_5mb', type=str, required=True, help='Path to the 5MB CNV data file.')
    parser.add_argument('--cnv_file_gistic', type=str, required=True, help='Path to the GISTIC CNV data file.')
    parser.add_argument('--patient_info', type=str, required=True, help='Path to the patient info file.')
    parser.add_argument('--features', type=str, nargs='+', required=True, help='List of features to include.')
    parser.add_argument('--model', type=str, choices=['lasso', 'naive_bayes', 'decision_tree'], required=True, help='Model type to train.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size as a fraction of total data.')
    
    args = parser.parse_args()
    
    # 读取数据
    snv_data = read_snv_data(args.snv_file)
    cnv_data = read_cnv_data(args.cnv_file_5mb, args.cnv_file_gistic)
    patient_info = preprocess_patient_info(args.patient_info)
    
    # 生成特征矩阵
    feature_matrix = generate_feature_matrix(snv_data, patient_info, args.features)
    X = feature_matrix.drop('label', axis=1).values
    y = feature_matrix['label'].values
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=args.test_size)
    
    # 训练模型
    if args.model == 'lasso':
        model = train_lasso(X_train, y_train)
    elif args.model == 'naive_bayes':
        model = train_naive_bayes(X_train, y_train)
    elif args.model == 'decision_tree':
        model = train_decision_tree(X_train, y_train)
    
    # 进行交叉验证并输出结果
    auc_scores = cross_validate_model(model, X_train, y_train)
    print(f"Cross-validated AUC scores: {auc_scores}")
    
    # 预测和评估
    predictions = predict_model(model, X_test)
    plot_roc_curve(y_test, predictions)

if __name__ == "__main__":
    main()


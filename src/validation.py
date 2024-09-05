from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def cross_validate_model(model, X, y, cv=5):
    """
    使用K折交叉验证模型。
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    auc_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        auc = roc_auc_score(y_test, predictions)
        auc_scores.append(auc)
    
    return auc_scores

def plot_roc_curve(y_true, y_scores):
    """
    绘制ROC曲线。
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


from sklearn.linear_model import LassoCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_lasso(X, y, alpha=1.0):
    """
    使用Lasso回归模型进行训练。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用LassoCV进行Lasso回归的训练和交叉验证
    lasso = LassoCV(alphas=[alpha], cv=5, random_state=0)
    lasso.fit(X_scaled, y)
    
    return lasso

def train_naive_bayes(X, y):
    """
    使用朴素贝叶斯分类器进行训练。
    """
    nb = GaussianNB()
    nb.fit(X, y)
    
    return nb

def train_decision_tree(X, y):
    """
    使用决策树模型进行训练。
    """
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X, y)
    
    return dt

def predict_model(model, X):
    """
    使用模型进行预测。
    """
    predictions = model.predict(X)
    return predictions

def train_test_split_data(X, y, test_size=0.2):
    """
    将数据分为训练集和测试集。
    """
    return train_test_split(X, y, test_size=test_size, random_state=0)


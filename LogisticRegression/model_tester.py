from LogisticRegression import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
df = pd.read_csv("./data/Social_Network_Ads.csv")
X_full = df.iloc[:, [1,2,3]].values
Y_full = df['Purchased'].values


encoder = LabelEncoder()

X_full[:, 0] = encoder.fit_transform(X_full[:, 0])


model = LogisticRegression()
model_skl = LR()
scaler = StandardScaler()
X_full[:, [1,2]] = scaler.fit_transform(X_full[:, [1,2]])


X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size = 0.25, random_state = 42)

model.fit(X_train, Y_train, epochs = 50)
model_skl.fit(X_train, Y_train)

preds = model.predict(X_test)
preds_skl = model_skl.predict(X_test)
cm = confusion_matrix(Y_test, preds)    
cm_skl = confusion_matrix(Y_test, preds_skl)  
def print_confusion_matrix(confusion_matrix, class_names, figsize = (16,10), fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Sklearn model')
    plt.savefig('confusion_matrix')
    return fig

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

#print_confusion_matrix(cm, ['Purchased:No', 'Purchased:Yes'])

#print_confusion_matrix(cm_skl, ['Purchased:No', 'Purchased:Yes'])
model =LogisticRegression()
np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

model.fit(X,y, epochs = 10000)

plot_decision_boundary(lambda x: model.predict(x))
plt.title("Logistic Regression")

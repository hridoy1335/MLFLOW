import mlflow
import mlflow.sklearn 
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='hridoy1335', repo_name='MLFLOW', mlflow=True)



mlflow.set_tracking_uri('https://dagshub.com/hridoy1335/MLFLOW.mlflow')
# Load Wine dataset
wine = load_wine()
x = wine.data
y = wine.target
#Train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 2)
# Define the params for RF model
max_depth = 10
n_estimators = 5

mlflow.set_experiment('test')
# Define the model with mlflow
with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    # Train the model
    rf.fit(X_train, y_train)
    # Make predictions
    y_pred = rf.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion_matrics = confusion_matrix(y_test,y_pred)
    # Log the metrics
    mlflow.log_metric("accuracy", accuracy)
    # Log the model params
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    
    # plt.figure(figsize=(15,7))
    # sns.heatmap(confusion_matrics,annot=True)
    # plt.savefig('confusion_matrix1.png')
    
    
    # mlflow.log_artifact('confusion_matrix1.png')
    # mlflow.log_artifact(__file__)
    
    mlflow.set_tags({'author':'Hridoy', 'date':'2022-02-20'})
    
    mlflow.sklearn.log_model(rf,'RandomForestClassifier')
    
    
    print('accuracy :', accuracy)

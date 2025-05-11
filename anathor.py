from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
# import dagshub
# dagshub.init(repo_owner='hridoy1335', repo_name='MLFLOW', mlflow=True)

# mlflow.autolog()
# mlflow.set_tracking_uri('https://dagshub.com/hridoy1335/MLFLOW.mlflow')

data = load_breast_cancer()
x = pd.DataFrame(data.data,columns=data. feature_names)
y = pd.Series(data.target, name='target')
# Splitting into training and testing sets
X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Creating the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)
#Defining the parameter grid for GridSearchev
# with mlflow.start_run():
param_grid ={
'n_estimators': [10,50, 100],
'max _depth': [10, 20, 30]
}
# Applying GridSearchev(
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# Run without MLflow from here
grid_search.fit(X_train, y_train)
# Displaying the best params and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(best_params)
print(best_score)
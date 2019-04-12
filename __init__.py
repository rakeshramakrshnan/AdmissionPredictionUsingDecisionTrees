import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

# Data Study
source = pd.read_csv('Fall_2014.csv')
print(source.head())
print(source.tail(5))
print(source.columns.values)

# Data_Pre Processing
to_drop = ['Timestamp', 'Name', 'UniversityApplied', 'Date of Application', 'Date of Decision', 'Under Graduate Aggregate', 'Scale', 'Undergrad Univesity']
source.drop(to_drop, inplace=True, axis=1)

# Drop IELTS
source.drop('IELTS', inplace=True, axis=1)

# Fill NaN for AWA
source['AWA'].fillna(round(source['AWA'].mean(),1), inplace=True)
print(source['AWA'].values)

# Fill NaN for Toefl
source['TOEFL'].fillna(int(source['TOEFL'].mean()), inplace=True)
print(source['TOEFL'].values)

# Encode the result
encoder = LabelEncoder()
source['Result'] = encoder.fit_transform(source['Result'])
print(source['Result'].unique()) # Shows that all the values are encoded

# Convert training data to numeric
source['Percentage'] = source['Percentage'].apply(pd.to_numeric, errors='coerce') #coerce would change the non numeric to NaN
source['Percentage'].fillna(round(source['Percentage'].mean(), 2), inplace=True) # Change NaN to mean value

# Feature Scaling
#max = source['Percentage'].max()
source['GRE'] = preprocessing.minmax_scale(source['GRE'], feature_range=(0, 1))
source['GRE (Quants)'] = preprocessing.minmax_scale(source['GRE (Quants)'], feature_range=(0, 1))
source['AWA'] = preprocessing.minmax_scale(source['AWA'], feature_range=(0, 1))
source['TOEFL'] = preprocessing.minmax_scale(source['TOEFL'], feature_range=(0, 1))
source['Work-Ex'] = preprocessing.minmax_scale(source['Work-Ex'], feature_range=(0, 1))
source['International Papers'] = preprocessing.minmax_scale(source['International Papers'], feature_range=(0, 1))
source['Percentage'] = preprocessing.minmax_scale(source['Percentage'], feature_range=(0, 1))

# Test Train Split
features = ['GRE', 'GRE (Quants)', 'AWA', 'TOEFL', 'Work-Ex', 'International Papers', 'Percentage']
X = source[features]
y = source.Result
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(X_train.tail(5))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Optimizing Decision Tree Performance
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# View the decision tree
pydot = StringIO()
export_graphviz(clf, out_file=pydot, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(pydot.getvalue())
graph.write_png('DecisionTree.png')
Image(graph.create_png())

# Get values from user
n = 1
while n >= 1:
    print("Enter the GRE Score")
    GRE = int(input())
    print("Enter the GRE(Quants) score")
    GREQ = int(input())
    print("Enter the AWA Score")
    AWA = round(float(input()), 1)
    print("Enter the TOEFL Score")
    TOEFL = int(input())
    print("Enter the Work Experience")
    we = round(float(input()), 1)
    print("Enter the International Papers")
    ip = int(input())
    print("Enter the percentage")
    per = round(float(input()))
    # per = round(float(input())/max, 6)
    user_input = {'GRE': [GRE], 'GRE (Quants)': [GREQ], 'AWA': [AWA], 'TOEFL': [TOEFL],
                  'Work-Ex': [we], 'International Papers': [ip], 'Percentage': [per]}
    user_df = pd.DataFrame(user_input)
    print(user_df)
    y_pred = clf.predict(user_df)
    print(y_pred)
    result = "Rejected"
    if y_pred[0] == 1:
        result = "Accepted"
    else:
        result = "Rejected"
    print("The result is "+result)
    print("If you want to continue enter 1 else enter 0")
    n = int(input())






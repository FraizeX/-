import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
df=pd.read_csv("train.csv")
df.drop(["reason", "traveltime", "Mjob","Fjob","address", "paid", "absences","schoolsup","higher"],axis=1,inplace=True)
df.fillna(-1, inplace=True)
def sex(sex):
    if sex == "F":
        return 0
    else:
        return 1
def Medu(Medu):
    if Medu =="none" or Medu == -1:
        return 0
    if Medu =="primary education (4th grade)":
        return 1
    if Medu =="5th to 9th grade":
        return 2
    if Medu =="secondary education":
        return 3
    if Medu =="higher education":
        return 4
def Fedu(Fedu):
    if Fedu =="none" or Fedu == -1:
        return 0
    if Fedu =="primary education (4th grade)":
        return 1
    if Fedu =="5th to 9th grade":
        return 2
    if Fedu =="secondary education":
        return 3
    if Fedu =="higher education":
        return 4
def famsize(famsize):
    if famsize=="greater than 3 persons":
        return 0
    else:
        return 1
def activities(activities):
    if activities == 'no':
        return 1
    else:
        return 0
def internet(internet):
    if internet == 'yes':
        return 0
    else:
        return 1
def nursery(nursery):
    if nursery == 'yes':
        return 1
    else:
        return 0
def guardian(guardian):
    if guardian == -1:
        return 0
    if guardian =="father":
        return 1
    if guardian =="mother":
        return 2
    if guardian =="other":
        return 3
def studytime(studytime):
    if studytime == "less than 2 hours":
        return 0
    if studytime == "2 to 5 hours":
        return 1
    if studytime == "5 to 10 hours":
        return 2
    if studytime == "more than 10 hours":
        return 3
def failures(failures):
    if failures == 0:
        return 0
    if failures == 1:
        return 1
    if failures == 2:
        return 2
    if failures == 3:
        return 3
def famsup(famsup):
    if famsup == "yes":
        return 1
    else:
        return 0
def freetime(freetime):
    if freetime == "very low":
        return 0
    elif freetime == 'low':
        return 1
    elif freetime == 'medium':
        return 2
    elif freetime == 'high':
        return 3
    elif freetime == "very high":
        return 4
def famrel(famrel):
    if famrel == 'very bad':
        return 0
    elif famrel == "bad":
        return 1
    elif famrel == 'normal':
        return 2
    elif famrel == 'good':
        return 3
    elif famrel == 'excellent':
        return 4
    
df["sex"] = df["sex"].apply(sex)
df["famrel"] = df["famrel"].apply(famrel)
df["freetime"] = df["freetime"].apply(freetime)
df["famsup"] = df["famsup"].apply(famsup)
df["failures"] = df["failures"].apply(failures)
df["studytime"] = df["studytime"].apply(studytime)
df["guardian"] = df["guardian"].apply(guardian)
df["nursery"] = df["nursery"].apply(nursery)
df["internet"] = df["internet"].apply(internet)
df["activities"] = df["activities"].apply(activities)
df["famsize"] = df["famsize"].apply(famsize)
df["Fedu"] = df["Fedu"].apply(Fedu)
df["Medu"] = df["Medu"].apply(Medu)

x = df.drop('result', axis = 1)
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
a = StandardScaler()
x_train = a.fit_transform(x_train)
x_test = a.transform(x_test)
c = KNeighborsClassifier(n_neighbors = 3)
c.fit(x_train, y_train)
y_pred = c.predict(x_test)
accuracy_score(y_test, y_pred) * 100
confusion_matrix(y_test, y_pred)
print(y_test, y_pred)
print(accuracy_score(y_test, y_pred) * 100)
print(confusion_matrix(y_test, y_pred))

#df.info()
#a=df.groupby(by='famrel').mean()
#print(a)
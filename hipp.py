import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import csv

df=pd.read_csv("e:/_MCA/programs_codes/Python/insurance.csv")

def desc_dataset():
    print(df)
    df.info()
    df.describe()
    df.isnull().sum()

def pie():
     # //////////////////////////////////////////////////////////////
    print("\n\n=======================================================")
    print("          Pie Chart for SEX, Smoker and Region")
    print("=======================================================\n\n")

    features = ['sex', 'smoker', 'region']

    plt.subplots(figsize=(10, 5))
    for i, col in enumerate(features):
        plt.subplot(1, 3, i + 1)

        x = df[col].value_counts()
        plt.pie(x.values,
                labels=x.index,
                autopct='%1.1f%%')

    plt.show()

def comp():
    print("\n\n===============================================================")
    print("   Comparison Between Charges paid between different groups ")
    print("===============================================================\n\n")



    features = ['sex', 'children', 'smoker', 'region']

    plt.figure(figsize=(17, 10))

    for i, col in enumerate(features):
        plt.subplot(2, 2, i + 1)
        df.groupby(col)['charges'].mean().plot.bar()
        plt.title(f'Mean Charges by {col}')

    plt.show()

def scatterplot():
    print("\n\n=======================================================")
    print("      Scatter Plot of charges paid vs age and BMI  ")
    print("=======================================================\n\n")
    features = ['age', 'bmi']

    plt.subplots(figsize=(17, 7))
    for i, col in enumerate(features):
        plt.subplot(1, 2, i + 1)
        sns.scatterplot(data=df, x=col,
                    y='charges',
                    hue='smoker')
    plt.show()

def boxplot():
    print("\n\n=======================================================")
    print("                 Box Plot ")
    print("=======================================================\n\n")
    print("Box Plot for Age, BMI")

    df.drop_duplicates(inplace=True)
    sns.boxplot(df['age'])
    plt.show()

    sns.boxplot(df['bmi'])
    plt.show()

def bpoutlier():
    Q1=df['bmi'].quantile(0.25)
    Q2=df['bmi'].quantile(0.5)
    Q3=df['bmi'].quantile(0.75)
    iqr=Q3-Q1
    lowlim=Q1-1.5*iqr
    upplim=Q3+1.5*iqr
    print("Lower Limit : ",lowlim)
    print("Upper Limit : ",upplim)

def heat():
    heatmap_data = df.pivot_table(values='charges', index='region', columns='sex', aggfunc='mean')

    # Create the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title("Heatmap of Charges by Region and Sex")
    plt.show()

def hist():
        
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x="age", kde=True, bins=20)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

# //////////////////////////////////////////////////////////////
    print("\n\n=======================================================")
    print("                 Box Plot Outlier ")
    print("=======================================================\n\n")

    from feature_engine.outliers import ArbitraryOutlierCapper
    arb=ArbitraryOutlierCapper(min_capping_dict={'bmi':13.6749},max_capping_dict={'bmi':47.315})
    df[['bmi']]=arb.fit_transform(df[['bmi']])
    sns.boxplot(df['bmi'])
    plt.show()


def predict():
    


    print("\n\n=======================================================")
    print("                     Encoding ")
    print("=======================================================\n\n")


    df['sex']=df['sex'].map({'male':0,'female':1})
    df['smoker']=df['smoker'].map({'yes':1,'no':0})
    df['region']=df['region'].map({'northwest':0, 'northeast':1,'southeast':2,'southwest':3})



    # //////////////////////////////////////////////////////////////

    print("\n\n=======================================================")
    print("               Model Development")
    print("=======================================================\n\n")


    X=df.drop(['charges'],axis=1)
    Y=df[['charges']]
    from sklearn.linear_model import LinearRegression,Lasso
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    l1=[]
    l2=[]
    l3=[]
    cvs=0
    for i in range(40,50):
        xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=i)
        lrmodel=LinearRegression()
        lrmodel.fit(xtrain,ytrain)
        l1.append(lrmodel.score(xtrain,ytrain))
        l2.append(lrmodel.score(xtest,ytest))
        cvs=(cross_val_score(lrmodel,X,Y,cv=5,)).mean()
        l3.append(cvs)
        df1=pd.DataFrame({'train acc':l1,'test acc':l2,'cvs':l3})
    print(df1)

    # //////////////////////////////////////////////////////////////

    print("\n\n=======================================================")
    print("              Linerar Regression : ")
    print("=======================================================\n\n")


    xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)
    lrmodel=LinearRegression()
    lrmodel.fit(xtrain,ytrain)



    print(lrmodel.score(xtrain,ytrain))
    print(lrmodel.score(xtest,ytest))
    print(cross_val_score(lrmodel,X,Y,cv=5,).mean())


    # //////////////////////////////////////////////////////////////
    print("\n\n=======================================================")
    print("                     SVR")
    print("=======================================================\n\n")
    from sklearn.metrics import r2_score

    svrmodel=SVR()
    svrmodel.fit(xtrain,ytrain)

    ypredtrain1=svrmodel.predict(xtrain)
    ypredtest1=svrmodel.predict(xtest)

    print(("Training : "),r2_score(ytrain,ypredtrain1))
    print(("Testing : "),r2_score(ytest,ypredtest1))
    print(("Mean : "),cross_val_score(svrmodel,X,Y,cv=5,).mean())



    # //////////////////////////////////////////////////////////////
    print("\n\n=======================================================")
    print("          Random Forest Regression : ")
    print("=======================================================\n\n")


    rfmodel=RandomForestRegressor(random_state=42)
    rfmodel.fit(xtrain,ytrain)
    ypredtrain2=rfmodel.predict(xtrain)
    ypredtest2=rfmodel.predict(xtest)
    print(("Training : "),r2_score(ytrain,ypredtrain2))
    print(("Testing : "),r2_score(ytest,ypredtest2))
    print(("Mean : "),cross_val_score(rfmodel,X,Y,cv=5,).mean())



    # //////////////////////////////////////////////////////////////

    print("\n\n=======================================================")
    print("                    Final Model :")
    print("=======================================================\n\n")



    df.drop(df[['sex','region']],axis=1,inplace=True)
    Xf=df.drop(df[['charges']],axis=1)
    X=df.drop(df[['charges']],axis=1)

    xtrain,xtest,ytrain,ytest=train_test_split(Xf,Y,test_size=0.2,random_state=42)
    finalmodel=XGBRegressor(n_estimators=15,max_depth=3,gamma=0)
    finalmodel.fit(xtrain,ytrain)

    ypredtrain4=finalmodel.predict(xtrain)
    ypredtest4=finalmodel.predict(xtest)

    print("Training Accuracy : ",r2_score(ytrain,ypredtrain4))
    print("Test Accuracy : ",r2_score(ytest,ypredtest4))
    print("CV Score : ",cross_val_score(finalmodel,X,Y,cv=5,).mean())


    # //////////////////////////////////////////////////////////////
    print("\n\n=======================================================")
    print("          Saving Model in a dump file")
    print("=======================================================\n\n")
    from pickle import dump
    dump(finalmodel,open('insurancemodelf.pkl','wb'))

    print("Done...")

    age = int(input("Enter age: "))
    sex = input("Enter sex: ")
    bmi = float(input("Enter BMI: "))
    children = int(input("Enter number of children: "))
    smoker = input("Smoker? (Yes/No): ")
    region = input("Enter region: ")
    


    # //////////////////////////////////////////////////////////////
    print("\n\n=======================================================")
    print("                Predicting on new Data")
    print("=======================================================\n\n")
    new_data=pd.DataFrame({'age':age,'sex':sex,'bmi':bmi,'children':children,'smoker':smoker,'region':region},index=[0])
    new_data['smoker']=new_data['smoker'].map({'yes':1,'no':0})
    new_data=new_data.drop(new_data[['sex','region']],axis=1)
    print(finalmodel.predict(new_data))

def user_predict():
    """ """
    print("\n\n=======================================================")
    print("                     Encoding ")
    print("=======================================================\n\n")


    df['sex']=df['sex'].map({'male':0,'female':1})
    df['smoker']=df['smoker'].map({'yes':1,'no':0})
    df['region']=df['region'].map({'northwest':0, 'northeast':1,'southeast':2,'southwest':3})



    # //////////////////////////////////////////////////////////////

    print("\n\n=======================================================")
    print("               Model Development")
    print("=======================================================\n\n")


    X=df.drop(['charges'],axis=1)
    Y=df[['charges']]
    from sklearn.linear_model import LinearRegression,Lasso
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    l1=[]
    l2=[]
    l3=[]
    cvs=0
    for i in range(40,50):
        xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=i)
        lrmodel=LinearRegression()
        lrmodel.fit(xtrain,ytrain)
        l1.append(lrmodel.score(xtrain,ytrain))
        l2.append(lrmodel.score(xtest,ytest))
        cvs=(cross_val_score(lrmodel,X,Y,cv=5,)).mean()
        l3.append(cvs)
        df1=pd.DataFrame({'train acc':l1,'test acc':l2,'cvs':l3})
    # print(df1)

    # //////////////////////////////////////////////////////////////

    print("\n\n=======================================================")
    print("              Linerar Regression : ")
    print("=======================================================\n\n")


    xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)
    lrmodel=LinearRegression()
    lrmodel.fit(xtrain,ytrain)



    # print(lrmodel.score(xtrain,ytrain))
    # print(lrmodel.score(xtest,ytest))
    # print(cross_val_score(lrmodel,X,Y,cv=5,).mean())


    # //////////////////////////////////////////////////////////////
    print("\n\n=======================================================")
    print("                     SVR")
    print("=======================================================\n\n")
    from sklearn.metrics import r2_score

    svrmodel=SVR()
    svrmodel.fit(xtrain,ytrain)

    ypredtrain1=svrmodel.predict(xtrain)
    ypredtest1=svrmodel.predict(xtest)

    # print(("Training : "),r2_score(ytrain,ypredtrain1))
    # print(("Testing : "),r2_score(ytest,ypredtest1))
    # print(("Mean : "),cross_val_score(svrmodel,X,Y,cv=5,).mean())



    # //////////////////////////////////////////////////////////////
    print("\n\n=======================================================")
    print("          Random Forest Regression : ")
    print("=======================================================\n\n")


    rfmodel=RandomForestRegressor(random_state=42)
    rfmodel.fit(xtrain,ytrain)
    ypredtrain2=rfmodel.predict(xtrain)
    ypredtest2=rfmodel.predict(xtest)
    # print(("Training : "),r2_score(ytrain,ypredtrain2))
    # print(("Testing : "),r2_score(ytest,ypredtest2))
    # print(("Mean : "),cross_val_score(rfmodel,X,Y,cv=5,).mean())



    # //////////////////////////////////////////////////////////////

    print("\n\n=======================================================")
    print("                    Final Model :")
    print("=======================================================\n\n")



    df.drop(df[['sex','region']],axis=1,inplace=True)
    Xf=df.drop(df[['charges']],axis=1)
    X=df.drop(df[['charges']],axis=1)

    xtrain,xtest,ytrain,ytest=train_test_split(Xf,Y,test_size=0.2,random_state=42)
    finalmodel=XGBRegressor(n_estimators=15,max_depth=3,gamma=0)
    finalmodel.fit(xtrain,ytrain)

    ypredtrain4=finalmodel.predict(xtrain)
    ypredtest4=finalmodel.predict(xtest)

    # print("Training Accuracy : ",r2_score(ytrain,ypredtrain4))
    # print("Test Accuracy : ",r2_score(ytest,ypredtest4))
    # print("CV Score : ",cross_val_score(finalmodel,X,Y,cv=5,).mean())


    # //////////////////////////////////////////////////////////////
    print("\n\n=======================================================")
    print("          Saving Model in a dump file")
    print("=======================================================\n\n")
    from pickle import dump
    dump(finalmodel,open('insurancemodelf.pkl','wb'))

    # print("Done...")

    age = int(input("Enter age: "))
    sex = input("Enter sex: ")
    bmi = float(input("Enter BMI: "))
    children = int(input("Enter number of children: "))
    smoker = input("Smoker? (Yes/No): ")
    region = input("Enter region: ")
    


    # //////////////////////////////////////////////////////////////
    print("\n\n=======================================================")
    print("                Predicting on new Data")
    print("=======================================================\n\n")
    new_data=pd.DataFrame({'age':age,'sex':sex,'bmi':bmi,'children':children,'smoker':smoker,'region':region},index=[0])
    new_data['smoker']=new_data['smoker'].map({'yes':1,'no':0})
    new_data=new_data.drop(new_data[['sex','region']],axis=1)
    charges = float(finalmodel.predict(new_data))
    print(charges)

    satisfied = input("Are you satisfied with this prediction?  ")
    
    if satisfied == 'yes' or satisfied == 'YES' or satisfied == 'Yes':
        first_name = input("Enter first name: ")
        last_name = input("Enter last name: ")
        

        # Data to be added to the CSV file
        new_data = [first_name, last_name, age, sex, bmi, children, smoker, region, charges]
        dataset_data = [age, sex, bmi, children, smoker, region, charges]
        # CSV file name
        csv_file = "userdata.csv"
        csv_dataset = "insurance.csv"

        # Open the CSV file in append mode and write the new data
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_data)

        with open(csv_dataset, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(dataset_data)
            
        print("Data added to the CSV file successfully.")
    elif satisfied == 'no' or satisfied == 'NO'  or satisfied == 'No':
        print("We'll try to improve our algorithm...")
    else:
        print("Invalid Input! Please enter yes or no.")



def visualize():
    while True:
        print("1. Pie Chart")
        print("2. Box Plot for Age and Bmi")
        print("3. Compare all data with Box Plot")
        print("4. Boxplot Outlier")
        print("5. Scatter Plot")
        print("6. Histogram")
        print("7. Heat Map")
        
        print("9. goto admin")
        
        choice = input("Enter your choice: ")

        if choice == '1':
            pie()
        elif choice == '2':
            
            boxplot()
        elif choice == '3':
            comp()
        elif choice == '4':
            bpoutlier()
        elif choice == '5':
            scatterplot()
        elif choice == '6':
            hist()
        elif choice == '7':
            heat()
        elif choice == '9':
            admin()
        
        else:
             print("Invalid Choice! Please enter a valid option.")


def admin():
    print('Admin')
    while True:
        print("1. Describe dataset")
        print("2. Visualize Data")
        print("3. Show predictions")
        print("4. Logout")
        choice = input("Enter your choice: ")

        if choice == '1':
            desc_dataset()
        elif choice == '2':
            visualize()
        elif choice == '3':
            predict()
        elif choice == '4':
            break
        else:
             print("Invalid Choice! Please enter a valid option.")



# Function to register a new user
def register(username, password):
    with open('users.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, password])
    print("Registration successful!")

# Function to check if a user exists
def is_user_exists(username):
    with open('users.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == username:
                return True
    return False

# Function to verify login credentials
def login(username, password):
    with open('users.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == username and row[1] == password:
                return True
    return False



# Main program loop
while True:
    print("1. Admin Login")
    print("2. User Registration")
    print("3. User Login")
    print("4. Quit")

    choice = input("Enter your choice: ")

    if choice == '1':
         admin_username = input("Username : ")
         admin_password = input("Password : ")
         if(admin_username=="ankit" and admin_password=="123"):
              admin()
    elif choice == '2':
        username = input("Enter a username: ")
        if is_user_exists(username):
            print("Username already exists. Please choose a different one.")
        else:
            password = input("Enter a password: ")
            register(username, password)
    elif choice == '3':
        username = input("Enter your username: ")
        password = input("Enter your password: ")
        if login(username, password):
            print("Login successful!")
            user_predict()
            
        else:
            print("Login failed. Please check your credentials.")
    elif choice == '4':
        break
    else:
        print("Invalid choice. Please select a valid option.")




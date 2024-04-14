from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,Loan_Approval_Prediction,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Loan_Approval_Status(request):
        expense = 0
        kg_price=0
        if request.method == "POST":


            Loan_ID= request.POST.get('Loan_ID')
            Gender= request.POST.get('Gender')
            Married= request.POST.get('Married')
            Dependents= request.POST.get('Dependents')
            Education= request.POST.get('Education')
            Self_Employed= request.POST.get('Self_Employed')
            ApplicantIncome= request.POST.get('ApplicantIncome')
            CoapplicantIncome= request.POST.get('CoapplicantIncome')
            LoanAmount= request.POST.get('LoanAmount')
            Loan_Amount_Term= request.POST.get('Loan_Amount_Term')
            Credit_History= request.POST.get('Credit_History')
            Property_Area= request.POST.get('Property_Area')


            df = pd.read_csv("Bank_Dataset.csv")

            df['label'] = df['Loan_Status'].map({'N': 0, 'Y': 1})
            df['LoanAppId'] = df['Loan_ID']
            # df.drop(['Loan_ID','Loan_Status'],axis=1,inplace=True)
            X = df['LoanAppId']
            y = df['label']

            print(X)
            print(y)

            from sklearn.feature_extraction.text import CountVectorizer

            cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))
            X = cv.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            predictors = []

            print("SVM")
            # SVM Model
            from sklearn import svm

            lin_clf = svm.LinearSVC()
            lin_clf.fit(X_train, y_train)
            predict_svm = lin_clf.predict(X_test)
            svm_acc = accuracy_score(y_test, predict_svm) * 100
            print(svm_acc)
            from sklearn.metrics import confusion_matrix, f1_score
            print(confusion_matrix(y_test, predict_svm))
            print(classification_report(y_test, predict_svm))
            predictors.append(('svm', lin_clf))

            # Logistic Regression Model
            print("Logistic Regression")
            from sklearn.linear_model import LogisticRegression
            logreg = LogisticRegression(random_state=42)
            logreg.fit(X_train, y_train)
            predict_log = logreg.predict(X_test)
            logistic = accuracy_score(y_test, predict_log) * 100
            print(logistic)
            from sklearn.metrics import confusion_matrix, f1_score
            print(confusion_matrix(y_test, predict_log))
            print(classification_report(y_test, predict_log))
            predictors.append(('logistic', logreg))

            # Decision Tree Classifier
            print("Decision Tree Classifier")
            dtc = DecisionTreeClassifier()
            dtc.fit(X_train, y_train)
            dtcpredict = dtc.predict(X_test)
            print("ACCURACY")
            print(accuracy_score(y_test, dtcpredict) * 100)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, dtcpredict))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, dtcpredict))
            predictors.append(('DecisionTreeClassifier', dtc))

            # Random Forest Classifier
            print("Random Forest Classifier")
            from sklearn.ensemble import RandomForestClassifier
            RFC = RandomForestClassifier(random_state=0)
            RFC.fit(X_train, y_train)
            pred_rfc = RFC.predict(X_test)
            RFC.score(X_test, y_test)
            print("ACCURACY")
            print(accuracy_score(y_test, pred_rfc) * 100)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, pred_rfc))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, pred_rfc))
            predictors.append(('RandomForestClassifier', RFC))

            classifier = VotingClassifier(predictors)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            data = [Loan_ID]

            vect = cv.transform(data).toarray()
            my_prediction = classifier.predict(vect)

            if my_prediction == 1:
                val = 'Approved'

            else:
                val = 'Not Approved'

            print(val)

            Loan_Approval_Prediction.objects.create(Loan_ID=Loan_ID,
            Gender=Gender,
            Married=Married,
            Dependents=Dependents,
            Education=Education,
            Self_Employed=Self_Employed,
            ApplicantIncome=ApplicantIncome,
            CoapplicantIncome=CoapplicantIncome,
            LoanAmount=LoanAmount,
            Loan_Amount_Term=Loan_Amount_Term,
            Credit_History=Credit_History,
            Property_Area=Property_Area,
            Prediction=val)


            return render(request, 'RUser/Predict_Loan_Approval_Status.html',{'objs':val})
        return render(request, 'RUser/Predict_Loan_Approval_Status.html')


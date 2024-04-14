


from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,Loan_Approval_Prediction,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')


def viewtreandingquestions(request,chart_type):
    dd = {}
    pos,neu,neg =0,0,0
    poss=None
    topic = Loan_Approval_Prediction.objects.values('ratings').annotate(dcount=Count('ratings')).order_by('-dcount')
    for t in topic:
        topics=t['ratings']
        pos_count=Loan_Approval_Prediction.objects.filter(topics=topics).values('names').annotate(topiccount=Count('ratings'))
        poss=pos_count
        for pp in pos_count:
            senti= pp['names']
            if senti == 'positive':
                pos= pp['topiccount']
            elif senti == 'negative':
                neg = pp['topiccount']
            elif senti == 'nutral':
                neu = pp['topiccount']
        dd[topics]=[pos,neg,neu]
    return render(request,'SProvider/viewtreandingquestions.html',{'object':topic,'dd':dd,'chart_type':chart_type})

def View_All_Loan_Approval_Prediction(request):

    obj = Loan_Approval_Prediction.objects.all()
    return render(request, 'SProvider/View_All_Loan_Approval_Prediction.html', {'objs': obj})

def Find_Loan_Approval_Type_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Not Approved'
    print(kword)
    obj = Loan_Approval_Prediction.objects.all().filter(Q(Prediction=kword))
    obj1 = Loan_Approval_Prediction.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Approved'
    print(kword1)
    obj1 = Loan_Approval_Prediction.objects.all().filter(Q(Prediction=kword1))
    obj11 = Loan_Approval_Prediction.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)


    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/Find_Loan_Approval_Type_Ratio.html', {'objs': obj})


def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = Loan_Approval_Prediction.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def negativechart(request,chart_type):
    dd = {}
    pos, neu, neg = 0, 0, 0
    poss = None
    topic = Loan_Approval_Prediction.objects.values('ratings').annotate(dcount=Count('ratings')).order_by('-dcount')
    for t in topic:
        topics = t['ratings']
        pos_count = Loan_Approval_Prediction.objects.filter(topics=topics).values('names').annotate(topiccount=Count('ratings'))
        poss = pos_count
        for pp in pos_count:
            senti = pp['names']
            if senti == 'positive':
                pos = pp['topiccount']
            elif senti == 'negative':
                neg = pp['topiccount']
            elif senti == 'nutral':
                neu = pp['topiccount']
        dd[topics] = [pos, neg, neu]
    return render(request,'SProvider/negativechart.html',{'object':topic,'dd':dd,'chart_type':chart_type})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})

def likeschart1(request,like_chart):
    charts =detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart1.html", {'form':charts, 'like_chart':like_chart})

def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="TrainedData.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Loan_Approval_Prediction.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.Loan_ID, font_style)
        ws.write(row_num, 1, my_row.Gender, font_style)
        ws.write(row_num, 2, my_row.Married, font_style)
        ws.write(row_num, 3, my_row.Dependents, font_style)
        ws.write(row_num, 4, my_row.Education, font_style)
        ws.write(row_num, 5, my_row.Self_Employed, font_style)
        ws.write(row_num, 6, my_row.ApplicantIncome, font_style)
        ws.write(row_num, 7, my_row.CoapplicantIncome, font_style)
        ws.write(row_num, 8, my_row.LoanAmount, font_style)
        ws.write(row_num, 9, my_row.Loan_Amount_Term, font_style)
        ws.write(row_num, 10, my_row.Credit_History, font_style)
        ws.write(row_num, 11, my_row.Property_Area, font_style)
        ws.write(row_num, 12, my_row.Prediction, font_style)

    wb.save(response)
    return response

def Train_Test_DataSets(request):

    detection_accuracy.objects.all().delete()

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
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

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
    detection_accuracy.objects.create(names="Logistic Regression", ratio=logistic)

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
    detection_accuracy.objects.create(names="Decision Tree Classifier", ratio=accuracy_score(y_test, dtcpredict) * 100)

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
    detection_accuracy.objects.create(names="Random Forest Classifier", ratio=accuracy_score(y_test, pred_rfc) * 100)
    obj = detection_accuracy.objects.all()

    return render(request,'SProvider/Train_Test_DataSets.html', {'objs': obj})















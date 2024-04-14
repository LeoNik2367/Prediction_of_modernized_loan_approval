from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class Loan_Approval_Prediction(models.Model):

    Loan_ID= models.CharField(max_length=300)
    Gender= models.CharField(max_length=300)
    Married= models.CharField(max_length=300)
    Dependents= models.CharField(max_length=300)
    Education= models.CharField(max_length=300)
    Self_Employed= models.CharField(max_length=300)
    ApplicantIncome= models.CharField(max_length=300)
    CoapplicantIncome= models.CharField(max_length=300)
    LoanAmount= models.CharField(max_length=300)
    Loan_Amount_Term= models.CharField(max_length=300)
    Credit_History= models.CharField(max_length=300)
    Property_Area= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)


class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



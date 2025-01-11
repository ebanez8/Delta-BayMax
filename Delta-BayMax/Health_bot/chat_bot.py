import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DiagnosisBot:
    def __init__(self, data, target):
        self.le = preprocessing.LabelEncoder()
        self.symptoms_dict = {}
        self.current_symptoms = [0] * len(data.columns)
        
        # Define x and y
        x = data
        y = target
        
        # Model initialization
        self.le.fit(y)
        y = self.le.transform(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        self.model = DecisionTreeClassifier()
        self.model.fit(x_train, y_train)
        
        for index, symptom in enumerate(data.columns):
            self.symptoms_dict[symptom] = index

    def readn(self, nstr):
        engine = pyttsx3.init()
        engine.setProperty('voice', "english+f5")
        engine.setProperty('rate', 130)
        engine.say(nstr)
        engine.runAndWait()
        engine.stop()

    def predict_disease(self):
        prediction = self.model.predict([self.current_symptoms])
        return self.le.inverse_transform(prediction)[0]

# Global dictionaries
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

# Define the base directory and data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYMPTOM_SEVERITY_PATH = os.path.join(BASE_DIR, 'MasterData', 'symptom_severity.csv')
SYMPTOM_DESCRIPTION_PATH = os.path.join(BASE_DIR, 'MasterData', 'symptom_Description.csv')
SYMPTOM_PRECAUTION_PATH = os.path.join(BASE_DIR, 'MasterData', 'symptom_precaution.csv')

def getSeverityDict():
    global severityDictionary
    with open(SYMPTOM_SEVERITY_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:  # Ensure there are at least 2 columns
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)

def getDescription():
    global description_list
    with open(SYMPTOM_DESCRIPTION_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:  # Ensure there are at least 2 columns
                _description = {row[0]: row[1]}
                description_list.update(_description)

def getprecautionDict():
    global precautionDictionary
    with open(SYMPTOM_PRECAUTION_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 5:  # Ensure there are at least 5 columns
                _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
                precautionDictionary.update(_prec)

def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum += severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        print("You should take the consultation from doctor.")
    else:
        print("It might not be that bad but you should take precautions.")

def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t", end="->")
    name = input("")
    print("Hello, ", name)

def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []

def sec_predict(symptoms_exp):
    df = pd.read_csv(os.path.join(BASE_DIR, 'Data', 'Training.csv'))
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X.columns)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node, bot):
    node = node[0]
    val = node.nonzero()
    disease = bot.le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def tree_to_code(tree, feature_names, X, df, bot):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("\nEnter the symptom you are experiencing  \t\t", end="->")
        disease_input = input("")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days = int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node], bot)
            red_cols = X.columns
            symptoms_given = red_cols[df.loc[present_disease].values[0].nonzero()]
            print("Are you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                print(syms, "? : ", end='')
                while True:
                    inp = input("")
                    if inp == "yes" or inp == "no":
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ", end="")
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])
            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            precautions = precautionDictionary.get(second_prediction[0], [])
            print("Take following precautions : ")
            for i, j in enumerate(precautions):
                print(i + 1, ")", j)

    recurse(0, 1)

# Initialize dictionaries
getSeverityDict()
getDescription()
getprecautionDict()
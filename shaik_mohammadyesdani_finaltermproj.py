# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss, accuracy_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

warnings.filterwarnings("ignore")

# Loading the Dataset
heartDatasetlink = r"https://raw.githubusercontent.com/Yesdani20/Datasets/refs/heads/main/Heart_Disease_Dataset.csv"
try:
    heartDataframe = pd.read_csv(heartDatasetlink)
    if heartDataframe.empty:
        print("The given dataset is empty")
    else:
        print(f"The dataset is loaded, It has {heartDataframe.shape[0]} rows and {heartDataframe.shape[1]} columns")
except FileNotFoundError:
    print(f"The file at {heartDatasetlink} was not found, Please check the file path")
except pd.errors.EmptyDataError:
    print("The file exists but is empty, Please re-check the contents in the file")
except Exception as e:
    print(f"An error occurred loading the dataset: {e}")

print("Information About Dataset")
print(heartDataframe.info())
print()

# Finding Missing Values
print("Finding Missing Values in Dataset")
print(heartDataframe.isnull().sum())
print()

print("Dataframe before Encoding:")
print(heartDataframe)
print()

# Checking for Categorical Features in the Dataframe
print("Checking for Categorical Features in the Dataframe")
heartdfCategoricalCol = heartDataframe.select_dtypes(include='object').columns.tolist()
for name in heartdfCategoricalCol:
    uniqueValues = heartDataframe[name].unique()
    print(f"Types of Values in \"{name}\":", uniqueValues)
print()

# Preprocessing the Data
label_Encoder = LabelEncoder()
# Changing Categorical Features to Numerical
for name in heartdfCategoricalCol:
    heartDataframe[name] = label_Encoder.fit_transform(heartDataframe[name])
print("Dataframe after Encoding:")
print(heartDataframe)
print()

# Seperating Features and Target Labels
heartFeat = heartDataframe.iloc[:, :-1]
heartTarg = heartDataframe.iloc[:, -1]

# Visualization of Target Labels
sns.countplot(x=heartTarg, palette=["green", "red"])
plt.xlabel("Heart Disease No ('0') or Yes ('1')")
plt.ylabel("Count")
plt.title("Visualization of Target Values")
plt.show()
print()

# Percentage difference between target values
print("Information About the Graph")
yesHeartDisease, noHeartDisease = heartTarg.value_counts()
totalValues = heartTarg.count()
print(f"{(noHeartDisease / totalValues) * 100}% of data predicts which is {noHeartDisease} No Heart Disease")
print(f"{(yesHeartDisease / totalValues) * 100}% of data predicts which is {yesHeartDisease} Yes Heart Disease")
print()

# Correlation Matrix
plt.figure(figsize=(10, 8))
corre = heartFeat.corr()
sns.heatmap(corre, annot=True, fmt=".2f", cbar=True)
plt.title("Correlation Matrix for Dataset")
plt.show
print()

# Histogram for all Features
heartFeat.hist(figsize=(10, 10))
plt.title("Histogram for all Features")
plt.show()
print()

# Pairplot
sns.pairplot(heartDataframe, hue="HeartDisease")
plt.title("Pairplot for all Data")
plt.show()
print()

# Splitting Training and Testing Data
heartFeat_Train, heartFeat_Test, heartTarg_Train, heartTarg_Test = train_test_split(heartFeat, heartTarg, test_size=0.2, random_state=42)

# Normalizing the Training Data
scaler = StandardScaler()
heartFeat_Train_std = scaler.fit_transform(heartFeat_Train)
heartFeat_Test_std = scaler.transform(heartFeat_Test)

# Reshaping data for LSTM
heartFeat_TrainReshaped = heartFeat_Train_std.reshape((heartFeat_Train_std.shape[0], heartFeat_Train_std.shape[1], 1))
heartFeat_TestReshaped = heartFeat_Test_std.reshape((heartFeat_Test_std.shape[0], heartFeat_Test_std.shape[1], 1))

# Plotting for Confusion Matrix
def plotConfusionMatrix(matrix, al):
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {al}")
    plt.show()
    print()

# Plotting for ROC_AUC Curve
def plotROCAUC(alg, fpR, tpR, rOCAUc):
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpR, tpR, color="darkorange", lw=2, label=f"ROC Curve {rOCAUc}")
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{alg}_ROC_AUC_Curve")
    plt.legend(loc="upper left")
    plt.show()
    print()

def perform_Metrics(alg, yTarg_TrainFold, yTarg_TestFold, yPred, yProb):
    # True Negative, False Positive, False Negative, True Positive
    tN, fP, fN, tP = confusion_matrix(yTarg_TestFold, yPred).ravel()
    if alg == "Random_Forest":
        print(f"Confusion Matrix for Random Forest Train Data: \n{confusion_matrix(yTarg_TestFold, yPred)}")
        plotConfusionMatrix(confusion_matrix(yTarg_TestFold, yPred), alg)
    elif alg == "K-Nearset_Neighbors":
        print(f"Confusion Matrix for K-Nearest Neighbor Train Data: \n{confusion_matrix(yTarg_TestFold, yPred)}")
        plotConfusionMatrix(confusion_matrix(yTarg_TestFold, yPred), alg)
    elif alg == "Long_Short-Term_Memory":
        print(f"Confusion Matrix for Long Short Term Memory Train Data: \n{confusion_matrix(yTarg_TestFold, yPred)}")
        plotConfusionMatrix(confusion_matrix(yTarg_TestFold, yPred), alg)

    if alg == "Random_Forest_Test":
        print(f"Confusion Matrix for Random Forest Test Data: \n{confusion_matrix(yTarg_TestFold, yPred)}")
        plotConfusionMatrix(confusion_matrix(yTarg_TestFold, yPred), alg)
        fpR, tpR, _ = roc_curve(yTarg_TestFold, yProb)
        if len(fpR) > 1 and len(tpR) > 1:
            rOCAUc = auc(fpR, tpR)
            plotROCAUC(alg, fpR, tpR, rOCAUc)
        else:
            raise ValueError("Insufficient data points for ROC Curve.")
    elif alg == "K-Nearset_Neighbors_Test":
        print(f"Confusion Matrix for K-Nearest Neighbor Test Data: \n{confusion_matrix(yTarg_TestFold, yPred)}")
        plotConfusionMatrix(confusion_matrix(yTarg_TestFold, yPred), alg)
        fpR, tpR, _ = roc_curve(yTarg_TestFold, yProb)
        if len(fpR) > 1 and len(tpR) > 1:
            rOCAUc = auc(fpR, tpR)
            plotROCAUC(alg, fpR, tpR, rOCAUc)
        else:
            raise ValueError("Insufficient data points for ROC Curve.")
    elif alg == "Long_Short-Term_Memory_Test":
        print(f"Confusion Matrix for Long Short Term Memory Test Data: \n{confusion_matrix(yTarg_TestFold, yPred)}")
        plotConfusionMatrix(confusion_matrix(yTarg_TestFold, yPred), alg)
        fpR, tpR, _ = roc_curve(yTarg_TestFold, yProb)
        if len(fpR) > 1 and len(tpR) > 1:
            rOCAUc = auc(fpR, tpR)
            plotROCAUC(alg, fpR, tpR, rOCAUc)
        else:
            raise ValueError("Insufficient data points for ROC Curve.")

    # True Positive Rate
    if (tP + fN) > 0:
        tPR = tP / (tP + fN)
    else:
        tPR = 0
    # True Negative Rate
    if (tN + fP) > 0:
        tNR = tN / (tN + fP)
    else:
        tNR = 0
    # False Positive Rate
    if (tN + fP) > 0:
        fPR = fP / (tN + fP)
    else:
        fPR = 0
    # False Negative Rate
    if (tP + fN) > 0:
        fNR = fN / (tP + fN)
    else:
        fNR = 0
    # False Discovery Rate
    if (tP + fP) > 0:
        fDR = fP / (tP + fP)
    else:
        fDR = 0
    # Recall or Sensitivity
    if (tP + fN) > 0:
        r = tP / (tP + fN)
    else:
        r = 0
    # Precision (Quality of Positive Prediction)
    if (tP + fP) > 0:
        p = tP / (tP + fP)
    else:
        p = 0
    # F1 Measure
    if (2 * tP + fP + fN) > 0:
        f1 = (2 * tP) / (2 * tP + fP + fN)
    else:
        f1 = 0
    # Accuracy
    if (tP + fP + fN + tN) > 0:
        accur = (tP + tN) / (tP + fP + fN + tN)
    else:
        accur = 0
    # Error Rate
    if (tP + fP + fN + tN) > 0:
        erRt = (fP + fN) / (tP + fP + fN + tN)
    else:
        erRt = 0
    # Balanced Accuracy
    baccur = (tPR + tNR) / 2
    # True Skill Statistics
    tSS = (tP / (tP + fN)) - (fP / (fP + tN))
    # Heidke Skill Score
    if (((tP + fN) * (fN + tN)) + ((tP + fP) * (fP + tN))) > 0:
        hSS = (2 * ((tP * tN) - (fP * fN))) / (((tP + fN) * (fN + tN)) + ((tP + fP) * (fP + tN)))
    else:
        hSS = 0
    aucccc = roc_auc_score(yTarg_TestFold, yProb)
    brier = brier_score_loss(yTarg_TestFold, yProb)
    # Calculate Baseline Brier Score
    baselineProb = [yTarg_TrainFold.mean()] * len(yTarg_TestFold)
    brierBaseline = brier_score_loss(yTarg_TestFold, baselineProb)
    if brierBaseline > 0:
        brierSkill = 1 - (brier / brierBaseline)
    else:
        brierSkill = 0
    metricsDt = {
        "True Positive (TP)": tP,
        "True Negative (TN)": tN,
        "False Positive (FP)": fP,
        "False Negative (FN)": fN,
        "Sensitivity (TPR)": tPR,
        "Specificity (TNR)": tNR,
        "False Positive Rate (FPR)": fPR,
        "False Negative Rate (FNR)": fNR,
        "Recall (r)": r,
        "Precision (P)": p,
        "F1 Measure (F1)": f1,
        "Accuracy": accur,
        "Error Rate": erRt,
        "Balanced Accuracy": baccur,
        "True Skill Statistics (TSS)": tSS,
        "Heidke Skill Score (HSS)": hSS,
        "ROC_AUC Score": aucccc,
        "Brier Score": brier,
        "Brier Skill Score": brierSkill
    }
    return metricsDt

# Random Forest Function
def randomForest(algori, rF_Lt, heartX_TrainFold_std, heartY_TrainFold, heartX_TestFold_std, heartY_TestFold):
    rFImp = RandomForestClassifier(n_estimators=100, random_state=42)
    rFImp.fit(heartX_TrainFold_std, heartY_TrainFold)
    rFPred = rFImp.predict(heartX_TestFold_std)
    rFProb =rFImp.predict_proba(heartX_TestFold_std)[:, 1]
    rFMetrics = perform_Metrics(algori, heartY_TrainFold, heartY_TestFold, rFPred, rFProb)
    rF_Lt.append(rFMetrics)
    return rF_Lt

# K-Nearest Neighbors Function
def kNearest_Neighbors(algori, kNN_Lt, heartX_TrainFold_std, heartY_TrainFold, heartX_TestFold_std, heartY_TestFold):
    knnImp = KNeighborsClassifier(n_neighbors=7)
    knnImp.fit(heartX_TrainFold_std, heartY_TrainFold)
    kNNPred = knnImp.predict(heartX_TestFold_std)
    kNNProb = knnImp.predict_proba(heartX_TestFold_std)[:, 1]
    kNNMetrics = perform_Metrics(algori, heartY_TrainFold, heartY_TestFold, kNNPred, kNNProb)
    kNN_Lt.append(kNNMetrics)
    return kNN_Lt

# Long Short Term Memory Function
def lSTM(algori, lstm_Lt, heartX_TrainFold_lstm, heartY_TrainFold, heartX_TestFold_lstm, heartY_TestFold):
    lstmImp = Sequential()
    lstmImp.add(LSTM(64, input_shape=(heartX_TrainFold_lstm.shape[1], 1), activation='relu', return_sequences=True))
    lstmImp.add(Dropout(0.2))
    lstmImp.add(LSTM(32, activation='relu'))
    lstmImp.add(Dropout(0.2))
    lstmImp.add(Dense(1, activation='sigmoid'))
    lstmImp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstmImp.fit(heartX_TrainFold_lstm, heartY_TrainFold, epochs=10, batch_size=16, verbose=0)
    lstmPred = (lstmImp.predict(heartX_TestFold_lstm) > 0.5).astype("int32")
    lstmProb = lstmImp.predict(heartX_TestFold_lstm).flatten()
    lstmMetrics = perform_Metrics(algori, heartY_TrainFold, heartY_TestFold, lstmPred, lstmProb)
    lstm_Lt.append(lstmMetrics)
    return lstm_Lt

rFtrain_Lt = []
kNNtrain_Lt = []
lstmtrain_Lt = []
fold = []

# Implementing StratifiedKFold as the Data is imbalanced
kFold_stratified = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for iterNum, (heartTrainInd, heartTestInd) in enumerate(kFold_stratified.split(heartFeat_Train, heartTarg_Train), 1):
    print(f"Fold Number: {iterNum}")
    fold.append(f"Fold_{iterNum}")
    
    # Splitting the Data for folds
    heartFeat_TrainFold, heartFeat_TestFold = heartFeat_Train.iloc[heartTrainInd], heartFeat_Train.iloc[heartTestInd]
    heartTarg_TrainFold, heartTarg_TestFold = heartTarg_Train.iloc[heartTrainInd], heartTarg_Train.iloc[heartTestInd]
    
    # Normalizing the Training and Test Data
    scaler = StandardScaler()
    heartFeat_TrainFold_std = scaler.fit_transform(heartFeat_TrainFold)
    heartFeat_TestFold_std = scaler.transform(heartFeat_TestFold)
    
    # Reshaping the Dataset for LSTM
    heartFeat_TrainFold_lstm = heartFeat_TrainFold_std.reshape(-1, heartFeat_TrainFold_std.shape[1], 1)
    heartFeat_TestFold_lstm = heartFeat_TestFold_std.reshape(-1, heartFeat_TestFold_std.shape[1], 1)
    
    # Random Forest Implementation
    rMetrics = randomForest("Random_Forest", rFtrain_Lt, heartFeat_TrainFold_std, heartTarg_TrainFold, heartFeat_TestFold_std, heartTarg_TestFold)
    
    # KNN Implementation
    knnMetrics = kNearest_Neighbors("K-Nearset_Neighbors", kNNtrain_Lt, heartFeat_TrainFold_std, heartTarg_TrainFold, heartFeat_TestFold_std, heartTarg_TestFold)
    
    # Long Short Term Memory (LSTM) Implementation
    lstmMetrics = lSTM("Long_Short-Term_Memory", lstmtrain_Lt, heartFeat_TrainFold_lstm, heartTarg_TrainFold, heartFeat_TestFold_lstm, heartTarg_TestFold)

# Converting Each Algorithm metrics to DataFrame
rF_df = pd.DataFrame(rMetrics).T
rF_df.columns = fold
kNN_df = pd.DataFrame(knnMetrics).T
kNN_df.columns = fold
lstm_df = pd.DataFrame(lstmMetrics).T
lstm_df.columns = fold

# Printing the Performance Metrics for each Algorithm for each Fold
print(f"Performance Metrics for 'Random Forest' Algorithm Using Stratified KFold for each Fold: \n{rF_df}\n")
print(f"Performance Metrics for 'K-Nearest Neighbor' Algorithm Using Stratified KFold for each Fold: \n{kNN_df}\n")
print(f"Performance Metrics for 'Long Short Term Memory (LSTM)' Algorithm Using Stratified KFold for each Fold: \n{lstm_df}\n")

# Printing Average for each Algorithm
rFAvg_df = rF_df.mean(axis=1)
kNNAvg_df = kNN_df.mean(axis=1)
lstmAvg_df = lstm_df.mean(axis=1)
avgPerformDF = pd.DataFrame({"Random Forest": rFAvg_df, "K-Nearest Neighbor": kNNAvg_df, "Long Short Term Memory (LSTM)": lstmAvg_df})
print(f"Comparing Average Performance Metrics for 'Random Forest', 'K-Nearest Neighbor', 'Long Short Term Memory (LSTM)' Algorithms Using Stratified KFold on Training Data: \n{avgPerformDF}\n")

# Normalizing for evaluating the Testing Data
scaler = StandardScaler()
heartFeat_Train_teststd = scaler.fit_transform(heartFeat_Train)
heartFeat_Test_teststd = scaler.transform(heartFeat_Test)

rFtest_Lt = []
kNNtest_Lt = []
lstmtest_Lt = []

# Reshaping the Test Data for LSTM
heartFeat_Traintest_lstm = heartFeat_Train_teststd.reshape(-1, heartFeat_Train_teststd.shape[1], 1)
heartFeat_Testtest_lstm = heartFeat_Test_teststd.reshape(-1, heartFeat_Test_teststd.shape[1], 1)

# Random Forest Implementation on Test Data
rTest = randomForest("Random_Forest_Test", rFtest_Lt, heartFeat_Train_teststd, heartTarg_Train, heartFeat_Test_teststd, heartTarg_Test)
    
# KNN Implementation on Test Data
kTest = kNearest_Neighbors("K-Nearset_Neighbors_Test", kNNtest_Lt, heartFeat_Train_teststd, heartTarg_Train, heartFeat_Test_teststd, heartTarg_Test)

# Long Short Term Memory (LSTM) Implementation on Test Data
lTest = lSTM("Long_Short-Term_Memory_Test", lstmtest_Lt, heartFeat_Traintest_lstm, heartTarg_Train, heartFeat_Testtest_lstm, heartTarg_Test)

# Converting Each Algorithm metrics to DataFrame for Test Data
rFtest_df = pd.DataFrame(rTest).T
kNNtest_df = pd.DataFrame(kTest).T
lstmtest_df = pd.DataFrame(lTest).T

# Performance Metrics for Test Data
testPerformDF = pd.concat([rFtest_df, kNNtest_df, lstmtest_df], axis=1)
algorithms = ['Random Forest', 'K-Nearest Neighbor', 'Long Short Term Memory (LSTM)']
testPerformDF.columns = algorithms
print(f"Comparing Performance Metrics for 'Random Forest', 'K-Nearest Neighbor', 'Long Short Term Memory (LSTM)' Algorithms on Testing Data: \n{testPerformDF}\n")


# Best Accuracy on Test Data
accuracy_Dt = {"Random Forest": rTest[0]["Accuracy"], "K-Nearest Neighbor": kTest[0]["Accuracy"], "Long Short Term Memory (LSTM)": lTest[0]["Accuracy"]}
max_Accuracy = max(accuracy_Dt, key=accuracy_Dt.get)
print(f"{max_Accuracy} has more Accuracy among all three algorithm which is {accuracy_Dt[max_Accuracy] * 100}% on Test Data")

# Best Accuracy on Train Data
accuracyTr_Dt = {"Random Forest": rFAvg_df["Accuracy"], "K-Nearest Neighbor": kNNAvg_df["Accuracy"], "Long Short Term Memory (LSTM)": lstmAvg_df["Accuracy"]}
max_AccuracyTr = max(accuracyTr_Dt, key=accuracyTr_Dt.get)
print(f"{max_AccuracyTr} has more Accuracy among all three algorithm which is {accuracyTr_Dt[max_AccuracyTr] * 100}% on Train Data")

prompt = input("Please Press enter to exit:")

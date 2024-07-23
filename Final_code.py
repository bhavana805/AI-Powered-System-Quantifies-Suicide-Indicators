from tkinter import *
from tkinter import filedialog, messagebox
import tkinter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten,Dropout
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier

# Create the main tkinter window
main = tkinter.Tk()
main.title("AI-Powered System Quantifies Suicide Indicators and Identifies Suicide Related Content in Online Posts")
main.geometry("1300x1200")

# Initialize global variables
filename, X, Y, classifier = None, None, None, None
X_train, X_test, y_train, y_test = None, None, None, None
le = None
accuracy, precision, recall, fscore = [], [], [], []

def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END, filename + " loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END, "Dataset before applying machine translation\n\n")
    text.insert(END, str(dataset.head()))

def processDataset():
    global dataset
    text.delete('1.0', END)
    label = dataset.groupby('attempt_suicide').size()
    label.plot(kind="bar")
    dataset.fillna(0, inplace=True)
    text.insert(END, "All missing values are replaced with 0\n")
    text.insert(END, "Total processed records found in dataset: " + str(dataset.shape[0]) + "\n\n")
    plt.show()

def translation():
    global X_train, X_test, y_train, y_test, X, Y, le, dataset
    text.delete('1.0', END)
    dataset.drop(['time', 'income'], axis=1, inplace=True)
    
    cols = ['gender', 'sexuallity','age', 'race', 'bodyweight', 'virgin', 'prostitution_legal', 'pay_for_sex', 'friends',
            'social_fear', 'depressed', 'what_help_from_others', 'attempt_suicide', 'employment', 
            'job_title', 'edu_level', 'improve_yourself_how']
    
    le = LabelEncoder()
    for col in cols:
        dataset[col] = pd.Series(le.fit_transform(dataset[col].astype(str)))
    
    text.insert(END, "Dataset after applying machine translation\n\n")
    text.insert(END, str(dataset) + "\n\n")
    
    Y = dataset['attempt_suicide'].values
    dataset.drop(['attempt_suicide'], axis=1, inplace=True)
    X = dataset.values
    
    sm = SMOTE(random_state=42,k_neighbors=2)
    X, Y = sm.fit_resample(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    text.insert(END, "Total records used to train machine learning LITE GBM Algorithm: " + str(X_train.shape[0]) + "\n")
    text.insert(END, "Total records used to test machine learning LITE GBM Algorithm: " + str(X_test.shape[0]) + "\n")

def trainCNN():
    global X_train, X_test, y_train, y_test, X, Y, classifier, accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    
    text.delete('1.0', END)
    
    # Reshape X and Y for CNN input
    XX = X.reshape(X.shape[0], X.shape[1], 1, 1)
    YY = to_categorical(Y)
    
    X_train1 = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
    print(X_train1.shape[0],'  | ', X_train1.shape[1])
    X_test1 = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    
    # Initialize the Sequential model
    classifier = Sequential()
    
    # Add Convolutional layers
    classifier.add(Convolution2D(32, (1, 1), input_shape=(X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(1, 1)))
    classifier.add(Convolution2D(32, (1, 1), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(1, 1)))
    
    # Flatten the output of the convolutional layers
    classifier.add(Flatten())
    
    # Add Dense layers
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=y_train1.shape[1], activation='softmax'))
    print(classifier.summary())
    # Compile the model
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fit the model on training data
    classifier.fit(XX, YY, batch_size=16, epochs=50, shuffle=True, verbose=2)
    
    # Make predictions on test data
    predict = classifier.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    
    # Calculate evaluation metrics
    a = accuracy_score(y_test1, predict) * 100
    p = precision_score(y_test1, predict, average='macro') * 100
    r = recall_score(y_test1, predict, average='macro') * 100
    f = f1_score(y_test1, predict, average='macro') * 100
    
    # Append metrics to global lists
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    # Display evaluation metrics on tkinter Text widget
    text.insert(END, "CNN Accuracy on Test Data: " + str(a) + "\n")
    text.insert(END, "CNN Precision on Test Data: " + str(p) + "\n")
    text.insert(END, "CNN Recall on Test Data: " + str(r) + "\n")
    text.insert(END, "CNN F1 Score on Test Data: " + str(f) + "\n\n")

def RFTraining():
    global accuracy, X_train, X_test, y_train, y_test, X, Y,rf
    rf = RandomForestClassifier(class_weight='balanced')
    rf.fit(X_train, y_train.ravel())
    
    predict = rf.predict(X_test)
    
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END, "Existing Random Forest Accuracy on Test Data: " + str(a) + "\n")
    text.insert(END, "Existing Random Forest Precision on Test Data: " + str(p) + "\n")
    text.insert(END, "Existing Random Forest Recall on Test Data: " + str(r) + "\n")
    text.insert(END, "Existing Random Forest F1 Score on Test Data: " + str(f) + "\n\n")

def predict():
    global classifier, text
    
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    testData = pd.read_csv(filename)
    temp = testData.values
    testData.fillna(0, inplace=True)
    testData.drop(['time', 'income','attempt_suicide'], axis=1, inplace=True)
    #testData = to_categorical(testData)
    '''
    
    testData.replace({'Yes': 1, 'No': 0}, inplace=True)
    
    # Normalize the data
    scaler = MinMaxScaler()
    testData_normalized = scaler.fit_transform(testData)
    '''

    # Reshape the data
    cols = ['gender', 'sexuallity','age', 'race', 'bodyweight', 'virgin', 'prostitution_legal', 'pay_for_sex', 'friends',
            'social_fear', 'depressed', 'what_help_from_others',  'employment', 
            'job_title', 'edu_level', 'improve_yourself_how']
    
    le = LabelEncoder()
    for col in cols:
        testData[col] = pd.Series(le.fit_transform(testData[col].astype(str)))

    testData_reshaped = testData.values.reshape(testData.shape[0], testData.shape[1], 1, 1)

    
    # Make predictions
    predictions = classifier.predict(testData_reshaped)
    predicted_classes = np.argmax(predictions, axis=1)
    
    text.insert(END, str(predicted_classes) + "\n\n")
    
    # Display original data and predictions
    for i in range(len(predicted_classes)):
        if predicted_classes[i] == 1:
            text.insert(END,str(temp[i])+" ====> No SUICIDAL Depression Detected\n\n")
        if predicted_classes[i] == 0:
            text.insert(END,str(temp[i])+" ====> SUICIDAL Depression Detected\n\n")

# UI Elements
font = ('times', 16, 'bold')
title = Label(main, text='AI-Powered System Quantifies Suicide Indicators and Identifies Suicide Related Content in Online Posts', bg='LightGoldenrod1', fg='medium orchid', font=font, height=3, width=120)
title.place(x=5, y=5)

font1 = ('times', 13, 'bold')

uploadButton = Button(main, text="Upload Suicide Detection Dataset", command=uploadDataset, font=font1)
uploadButton.place(x=10, y=100)

processButton = Button(main, text="Process Dataset", command=processDataset, font=font1)
processButton.place(x=300, y=100)

translationButton = Button(main, text="Apply Translation", command=translation, font=font1)
translationButton.place(x=500, y=100)

cnnButton = Button(main, text="Train CNN Algorithm", command=trainCNN, font=font1)
cnnButton.place(x=700, y=100)

rfButton = Button(main, text="Train Random Forest Algorithm", command=RFTraining, font=font1)
rfButton.place(x=900, y=100)

predictButton = Button(main, text="Predict Suicide", command=predict, font=font1)
predictButton.place(x=1200, y=100)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=150, font=font1)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=150)

main.config(bg='RoyalBlue1')
main.mainloop()

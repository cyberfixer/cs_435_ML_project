from numpy.random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptron_implementation import Perceptron
def SeperateClasses(df,classA, classB):
    indiceToDelete = []
    row = 0
    for i in df:
        if df[row][4] != classA and df[row][4] != classB:
            indiceToDelete.append(row)
        row+=1

    db = np.delete(df, indiceToDelete,axis=0)
    db = np.where(db == classA, 1, db)
    db = np.where(db == classB, -1 ,db )
    return db

def findWhatClassIsIt(classA, classB):
    df = pd.read_csv("train.data")
    df = df.to_numpy()
    df = SeperateClasses(df,classA, classB)
    shuffle(df)

    #x_train contains the features
    #y_train is the result
    x_train = df[:,0:4]
    y_train = df[:, 4]

    df = pd.read_csv("test.data")
    df = df.to_numpy()
    df = SeperateClasses(df,classA, classB)

    x_test = df[:,0:4]
    y_test = df[:,4]

    #My Implementation
    pp = Perceptron(30)
    pp.train(x_train, y_train)
    y_predicted = pp.predict(x_test)

    result = np.where(y_predicted==y_test, 1, -1)

    print(classA, classB,  "Errors: ", pp.errors_)

    correct = 0
    for i in result:
        if i == 1:
            correct += 1
    accuracy = str((correct /len(result))*100)
    print("Prediction Accuracy: %"+ accuracy)

    return accuracy, pp.errors_


accuracy = [None] * 3
errors = [None] * 3

accuracy[0], errors[0] = findWhatClassIsIt("class-1", "class-2")
accuracy[1], errors[1] = findWhatClassIsIt("class-2", "class-3")
accuracy[2], errors[2] = findWhatClassIsIt("class-3", "class-1")

plt.title("Training Epoch-Error Graph")
plt.xlabel("Epoch")
plt.ylabel("Errors")

plt.plot(errors[0], label = "class-1 class-2")
plt.plot(errors[1], label = "class-2 class-3")
plt.plot(errors[2], label = "class-3 class-1")


plt.legend()
plt.show()



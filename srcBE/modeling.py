#File to model the AST and ngram output vectors
#compare,def,while,for,if,var,expr,binOp,assign,num,str,from,import
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

print(__doc__)

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, average_precision_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import pandas as pd

#deep learning layered model
def dlmodel(file,n):

    data = file
    dataset = loadtxt(data, delimiter=',')

    # split into input (X) and output (y) variables
    x = dataset[:,0:n]
    y = dataset[:,n]

    #defining the keras model and layers... using relu/sigmoid as an activation function
    model = Sequential()
    model.add(Dense(12, input_dim=n, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) #not relu
    

    #compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x,y, epochs=100, batch_size=10)

    _, accuracy = model.evaluate(x,y)
    print('Accuracy: %.2f' % (accuracy*100))

    # make class predictions with the model
    predictions = model.predict_classes(x)
    # summarize the first 5 cases
    for i in range(10):
       correctness = "not correct"
       if(predictions[i] == y[i]):
           correctness = "correct"
       print('%s => %d (expected %d) ...%s' % (x[i].tolist(), predictions[i], y[i], correctness))


    #These metrics are all global metrics, but Keras works in batches. As a result, it might be more misleading than helpful.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    X_train = X_train[:, ~np.isnan(X_train).any(axis=0)]
    X_test = X_test[:, ~np.isnan(X_test).any(axis=0)]

    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
    model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=1)

    model.fit(X_train, y_train)
    y_score = model.predict(X_test)


#Calculates the roc curve for a given classifier
def roc(file, n, type):
    # Import some data to play with

    data = '/Users/blovs/CSCI Hw/AutomatedSourceCodeAnalysis-main/Results/3gCodeJamSoln.csv'
    dataset = loadtxt(data, delimiter=',')

    # split into input (X) and output (y) variables
    X = dataset[:,0:529]
    y = dataset[:,529]

    # Binarize the output
    y = label_binarize(y, classes=[0, 1])
    n_classes = y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

    X_train = X_train[:, ~np.isnan(X_train).any(axis=0)]
    X_test = X_test[:, ~np.isnan(X_test).any(axis=0)]

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
             lw=lw, label='4-gram (area = %0.2f)' % roc_auc["micro"])

    data = file
    dataset = np.genfromtxt(data, delimiter=',')

    # split into input (X) and output (y) variables
    X = dataset[:,0:n]
    y = dataset[:,n]

    # Binarize the output
    y = label_binarize(y, classes=[0, 1])
    n_classes = y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='red',
             lw=lw, label='AST (area = %0.2f)' % roc_auc["micro"])

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for '+ type +' data')
    plt.legend(loc="lower right")
    plt.show()


#implement random forest classifier
def randforest(file, n):
    data = file


#regression tree classifer (needs work)
def regtree(file, n):
    data = file
    RSEED = 50

    dataset = loadtxt(data, delimiter=',')

    # split into input (X) and output (y) variables
    x = dataset[:,0:n]
    y = dataset[:,n]

    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    X_train = X_train[:, ~np.isnan(X_train).any(axis=0)]
    X_test = X_test[:, ~np.isnan(X_test).any(axis=0)]

    # Create the model with 100 trees
    model = RandomForestClassifier(n_estimators=n,
                               random_state=RSEED,
                               max_features = n,
                               n_jobs=-1, verbose = 1)
    model.fit(x, y)

    n_nodes = []
    max_depths = []

    for ind_tree in model.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    print(f'Average number of nodes {int(np.mean(n_nodes))}')
    print(f'Average maximum depth {int(np.mean(max_depths))}')

    train_rf_predictions = model.predict(X_train)
    train_rf_probs = model.predict_proba(X_train)[:, 1]

    rf_predictions = model.predict(X_test)
    rf_probs = model.predict_proba(X_test)[:, 1]

    #stats for regression tree model
    def evaluate_model(n,file):

        from sklearn.metrics import precision_score, recall_score

        data = file
        dataset = loadtxt(data, delimiter=',')

        X = dataset[:, 0:n]
        y = dataset[:, n]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        X_train = X_train[:, ~np.isnan(X_train).any(axis=0)]
        X_test = X_test[:, ~np.isnan(X_test).any(axis=0)]

        # Create the model with 100 trees
        tree = LogisticRegression(random_state=0).fit(X, y)

        # Make probability predictions
        train_probs = tree.predict_proba(X_train)[:n, 1]
        probs = tree.predict_proba(X_test)[:n, 1]

        train_predictions = tree.predict(X_train)
        predictions = tree.predict(X_test)

        baseline = {}

        baseline['recall'] = recall_score(y_test, [1 for _ in range(len(y_test))])
        baseline['precision'] = precision_score(y_test, [1 for _ in range(len(y_test))])

        results = {}

        results['recall'] = recall_score(y_test, predictions)
        results['precision'] = precision_score(y_test, predictions)

        train_results = {}
        train_results['recall'] = recall_score(y_train, train_predictions)
        train_results['precision'] = precision_score(y_train, train_predictions)

        print('hi')

        for metric in ['recall', 'precision', 'roc']:
            print(
                f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

    evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs, n, file)


#Combine Data using naive bayes
def bayes(file, n):

    n = 0

#Trying to find most important feature (in-progress)
def featureselection(data,n):
        import ast
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        import numpy as np
        n=n-1

        cols = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
        df = pd.read_csv(data)
        df.head()
        print(df.describe())
        # sns.pairplot(df[cols], height=2.0)
        # plt.show()

        from sklearn.preprocessing import StandardScaler
        stdsc = StandardScaler()
        X_std = stdsc.fit_transform(df[cols].iloc[:, range(0, n)].values)
        cov_mat = np.cov(X_std.T)
        plt.figure(figsize=(7, 7))
        sns.set(font_scale=1.5)
        hm = sns.heatmap(cov_mat,
                         cbar=True,
                         annot=True,
                         square=False,
                         fmt='.2f',
                         annot_kws={'size': 6},
                         cmap='coolwarm',
                         yticklabels=cols,
                         xticklabels=cols)
        plt.title('Covariance matrix showing correlation coefficients', size=n)
        plt.tight_layout()
        plt.show()


#logistic regression classifier
def logReg(file, n):
    data = '/Users/blovs/CSCI Hw/AutomatedSourceCodeAnalysis-main/Results/3gCodeJamSoln.csv'
    dataset = np.genfromtxt(data, delimiter=',')
    # split into input (X) and output (y) variables

    print("im here")
    X = dataset[:, 0:n]
    y = dataset[:, n]

    print("done with datasets")

    # Binarize the output
    y = label_binarize(y, classes=[0, 1])
    n_classes = y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

    X_train = X_train[:, ~np.isnan(X_train).any(axis=0)]
    X_test = X_test[:, ~np.isnan(X_test).any(axis=0)]

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(LogisticRegression())
    classifier.fit(X_train, y_train)

    print("classifer done")
    y_score = classifier.decision_function(X_test)

    displayroc = True #set true to display the roc curve

    if(displayroc==True):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        print("done with if statement")
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"], color='red',
                 lw=lw, label='90 train 10 test (area = %0.2f)' % roc_auc["micro"])

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for AST and 4-gram data')
        plt.legend(loc="lower right")
        plt.show()

    displayImportance = True
    if(displayImportance == True):
        # Adapted from https://machinelearningmastery.com/calculate-feature-importance-with-python/
        importance = classifier.coef_[0]

        print(importance)

        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))


        # plot feature importance
        from matplotlib import pyplot
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()

#statistics including precision, recall, and f-measure
def stats(file, n):
    from sklearn.metrics import precision_score
    from sklearn.neural_network import MLPClassifier
    import pandas as pd
    data = file

    print(data)

    #dataset = np.genfromtxt(data, delimiter=',', filling_values=0.0)

    df= pd.read_csv(file)

    print(pf)

    #print(dataset)

    # split into input (X) and output (y) variables
    x = dataset[:,0:n]
    y = dataset[:,n]

    #x = x[~np.isnan(x)]
   # y= y[~np.isnan(y)]

    print(x)
    #print(y)

    print("im before train")

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    print("im after train")

    count = 0
    for i in y_train:
        count = count +1

    print(count)

    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1) #Using a multi-layer perceptron classifier

    #classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000)) #using a logistic regression classifier

    #print(X_train)
    X_train = X_train[:, ~np.isnan(X_train).any(axis=0)]
    X_test = X_test[:, ~np.isnan(X_test).any(axis=0)]

    print(X_test)

    

    print("Im here")
    classifier.fit(X_train, y_train)
    print("im here now")
    y_score = classifier.predict(X_test)

    precision = precision_score(y_test, y_score, average='weighted', zero_division= 0)
    recall = recall_score(y_test, y_score, average='weighted', zero_division= 0)
    fmeasure = f1_score(y_test, y_score, average='weighted', zero_division= 0)

    print('Average precision score: {0:0.4f}'.format(precision))
    print('Average recall score: {0:0.4f}'.format(recall))
    print('Average f measure score: {0:0.4f}'.format(fmeasure))

    

def MLPClass(file, n):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    import sklearn.metrics as metrics
    from sklearn.metrics import roc_curve, auc, average_precision_score, recall_score, f1_score, precision_score
    #df= pd.read_csv('/Users/blovs/CSCI Hw/AutomatedSourceCodeAnalysis-main/Results/3gCodeJamSoln.csv', delimiter= ',')
    df= pd.read_csv(file, delimiter= ',')
    df.fillna(0)

    print(df)


    x= df.iloc[:,1:n]
    y= df.iloc[:,n]



   







#print(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

    print(y_train)

    print(X_train)


    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    classifier.fit(X_train, y_train)
    y_score = classifier.predict(X_test)

    precision = precision_score(y_test, y_score, average='weighted', zero_division= 0)
    recall = recall_score(y_test, y_score, average='weighted', zero_division= 0)
    fmeasure = f1_score(y_test, y_score, average='weighted', zero_division= 0)

    print('Average precision score: {0:0.4f}'.format(precision))
    print('Average recall score: {0:0.4f}'.format(recall))
    print('Average f measure score: {0:0.4f}'.format(fmeasure))


    # Binarize the output
    y = label_binarize(y, classes=[0, 1])
    n_classes = y.shape[1]



    #displayroc = True #set true to display the roc curve

   # if(displayroc==True):
        # Compute ROC curve and ROC area for each class
    #    fpr = dict()
     #   tpr = dict()
      #  roc_auc = dict()
       # for i in range(n_classes):
        #    fpr[i], tpr[i], _ = roc_curve([y_test[:, i]], [y_score[:, i]])
         #   roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        #print("done with if statement")
        #fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        


        #plt.figure()
        #lw = 2
        #plt.plot(fpr["micro"], tpr["micro"], color='red',lw=lw, label='90 train 10 test (area = %0.2f)' % roc_auc["micro"])

        #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        #plt.xlim([0.0, 1.0])
        #plt.ylim([0.0, 1.05])
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title('ROC for AST and 4-gram data')
        #plt.legend(loc="lower right")
        #plt.show()

    # calculate the fpr and tpr for all thresholds of the classification
    probs = classifier.predict_proba(X_test)
    preds = probs[:,1]
    y_test= y_test.tolist()
    y_score= y_score.tolist()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr, tpr, threshold = metrics.roc_curve([0, 1], preds)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def main():
    #data file
    data = '/Users/blovs/CSCI Hw/AutomatedSourceCodeAnalysis-main/Results/3gCodeJamSoln.csv'
    #number of attributes
    n=511
    n=n-1

    #put function here
   # stats(data,n)
    MLPClass(data,n)
   

    #logReg(data, n)
    


if __name__ == "__main__":
     main()

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from scipy.special import expit
import math


from sklearn.svm import SVC

class LinearTransform:

    # This class does the linear transformation of inputs
    # Called from within ELM

    def __init__(self,a,b):
        self.a = a
        self.b = b

    def tf(self,x):
        return np.dot(self.a,x) + self.b


class ELM:

    # Initializes N Layer ELM and evaluates the inputs till the last hidden layer
    # n_units must be a list of hidden units in each layer

    # Choose the random activations used for your model by removing them from the list in ELM.rand_function

    # Add more if required but the signature must be copied correctly.

    def __init__(self,n_layers=1,n_units=(2),input_dim=10):

        self.n_layers = n_layers
        self.n_units = n_units
        self.input_dim = input_dim

        assert(len(n_units)==n_layers),"n_units must have "+str(n_layers)+" values"

        self.layers = []

        for i in range(0,n_layers):
            layer_i = []
            for j in range(0,n_units[i]):
                if(i==0):
                    layer_i.append((LinearTransform(a=np.random.rand(self.input_dim),b=np.random.rand()),self.rand_function()))
                else:
                    layer_i.append((LinearTransform(a=np.random.rand(n_units[i-1]),b=np.random.rand()),self.rand_function()))

            self.layers.append(layer_i)

    def evaluate(self,X):

        H_in = X

        for i in self.layers:
            H_out = []
            for j in H_in:
                H_out.append([k[1].__call__(k[0].tf(j)) for k in i])

            H_in = np.array(H_out)

        return H_in

    def sigmoid(self,h):
        return expit(h)

    def tanh(self,h):
        return np.tanh(h)

    def relu(self,h):
        if (h <= 0):
            return 0
        else:
            return h

    def leaky_relu(self,h):
        if (h < 0):
            return 0.01 * h
        else:
            return h

    def sin(self,h):
        return math.sin(h)

    def cos(self,h):
        return math.cos(h)

    def gauss(self,h):
        return math.exp(-(h ** 2))

    def rand_function(self):

        list_fns = [self.sigmoid,self.tanh,self.relu,self.leaky_relu,self.gauss]

        idx = np.random.randint(0,len(list_fns))

        return list_fns[idx]

# Load image dataset
images,targets = datasets.load_digits(return_X_y=True)

images = images.reshape(images.shape[0],-1)
images = images/np.max(images)

images_train,images_test, targets_train,targets_test = train_test_split(images,targets,test_size=0.2,shuffle=True)



#SVM on un-transformed data:
logreg = SVC()
logreg.fit(X=images_train,y=targets_train)

# Evaluate logreg accuracy on test
logreg_predicts_test = logreg.predict(images_test)
logreg_predicts_train = logreg.predict(images_train)

#ANN creation

model = Sequential()
model.add(Dense(units=16,activation='sigmoid',input_dim=images_train.shape[1]))
model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(images_train,to_categorical(targets_train),epochs=30,verbose=0)

ann_predicts_test = model.predict_classes(images_test,verbose=0)
ann_predicts_train = model.predict_classes(images_train,verbose=0)



# Initialize ELM class

elm = ELM(n_layers=1, n_units=[100000], input_dim=images_train.shape[1])

train_elm_transform = elm.evaluate(images_train)
test_elm_transform = elm.evaluate(images_test)

# ELM SVM on data:
elmlogreg = SVC()
elmlogreg.fit(X=train_elm_transform,y=targets_train)

elmlogreg_predicts_test = elmlogreg.predict(test_elm_transform)
elmlogreg_predicts_train = elmlogreg.predict(train_elm_transform)

print "##########################################"
print "BASIC LOGISTIC REGRESSION Accuracy on TEST:",accuracy_score(targets_test,logreg_predicts_test)
print "BASIC LOGISTIC REGRESSION Accuracy on TRAIN:",accuracy_score(targets_train,logreg_predicts_train)
print "##########################################"

print "##########################################"
print "ANN Accuracy on TEST:",accuracy_score(targets_test,ann_predicts_test)
print "ANN Accuracy on TRAIN:",accuracy_score(targets_train,ann_predicts_train)
print "##########################################"

print "##########################################"
print "ELM LOGISTIC REGRESSION Accuracy on TEST:",accuracy_score(targets_test,elmlogreg_predicts_test)
print "ELM LOGISTIC REGRESSION Accuracy on TRAIN:",accuracy_score(targets_train,elmlogreg_predicts_train)
print "##########################################"


from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from scipy.special import expit
import math
from keras.datasets import mnist
import tqdm

from sklearn.linear_model import LogisticRegression

class LinearTransform:
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def tf(self,x):
        return np.dot(self.a,x) + self.b

class ELM:

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

        list_fns = [self.sigmoid,self.tanh,self.relu,self.leaky_relu,self.sin,self.cos,self.gauss]

        idx = np.random.randint(0,len(list_fns))

        return list_fns[idx]

# Load image dataset
images,targets = datasets.load_digits(return_X_y=True)

images = images.reshape(images.shape[0],-1)
images = images/np.max(images)

images_train,images_test, targets_train,targets_test = train_test_split(images,targets,test_size=0.2,shuffle=True)

# (images_train, targets_train), (images_test, targets_test) = mnist.load_data()
#
# images_train = images_train.reshape(images_train.shape[0],-1).astype('float32')
# images_test = images_test.reshape(images_test.shape[0],-1).astype('float32')
#
# images_train/=255.0
# images_test/=255.0


# Basic logistic regression on data:
logreg = LogisticRegression(multi_class='multinomial',n_jobs=-1,solver= 'lbfgs')
logreg.fit(X=images_train,y=targets_train)
# Evaluate logreg accuracy on test
logreg_predicts_test = logreg.predict(images_test)


#ANN creation
# model = Sequential()
# model.add(Dense(units=16,activation='sigmoid',input_dim=images_train.shape[1]))
# model.add(Dense(units=10,activation='softmax'))
#
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#
# model.fit(images_train,to_categorical(targets_train),epochs=30,verbose=0)
#
# ann_predicts_test = model.predict_classes(images_test,verbose=0)


# Initialize ELM class




print "##########################################"
print "\n","BASIC LOGISTIC REGRESSION Accuracy:",accuracy_score(targets_test,logreg_predicts_test)
print "##########################################"



# print "##########################################"
# print "\n","ANN Accuracy:",accuracy_score(targets_test,ann_predicts_test)
# print "##########################################"



elm_acc_list = []

for i in tqdm(0,100):
    elm = ELM(n_layers=1, n_units=[1000], input_dim=images_train.shape[1])

    train_elm_transform = elm.evaluate(images_train)
    test_elm_transform = elm.evaluate(images_test)

    # ELM logistic regression on data:
    elmlogreg = LogisticRegression(multi_class='multinomial',n_jobs=-1,solver= 'lbfgs')
    elmlogreg.fit(X=train_elm_transform,y=targets_train)
    # Evaluate logreg accuracy on test
    elmlogreg_predicts_test = elmlogreg.predict(test_elm_transform)

    print "##########################################"
    print "\n","ELM LOGISTIC REGRESSION Accuracy:",accuracy_score(targets_test,elmlogreg_predicts_test)
    print "##########################################"

    elm_acc_list.append(accuracy_score(targets_test,elmlogreg_predicts_test))


print "##########################################"
print "\n","ELM LOGISTIC REGRESSION MEAN Accuracy:",np.mean(np.array(elm_acc_list))
print "##########################################"

print "##########################################"
print "\n","BASIC LOGISTIC REGRESSION Accuracy:",accuracy_score(targets_test,logreg_predicts_test)
print "##########################################"
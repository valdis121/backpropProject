import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv('iris.csv')
encoder = LabelEncoder()
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, -1].values

y1 = encoder.fit_transform(y)

Y = pd.get_dummies(y1).values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
np.random.seed(42)

def error_square(out,w):
    res=0
    for i in range(len(out)):
        res+=0.5*(w[i]-out[i])**2
    return res
class Layer:



    def __init__(self):

        pass

    def calculateError(self,out,w):
        return w-out
    def derivateActivation(self,inp):
        return inp
    def forward(self, inp):
        # a dummy layer returns the input
        return inp

    def backward(self, inp, grad_outp):
        pass


class ReLU(Layer):
    def __init__(self):
        pass
    def derivateActivation(self,inp):
        res=np.copy(inp)
        for i in range(len(inp)):
            if inp[i] < 0:
                res[i]=0
            else:
                res[i]=1
        return res

        return 1
    def forward(self, inp):
        return np.maximum(0, inp)

    def backward(self, inp, grad_outp):
        grad_outp1 = np.ndarray(shape=(len(inp),))
        for i in range(len(grad_outp)):
            grad_outp1[i] = np.sum(grad_outp[i])
        return grad_outp1*self.derivateActivation(inp)



class Sigmoid(Layer):
    def __init__(self):
        pass

    def derivateActivation(self,inp):
        return self.forward(inp)*(1-self.forward(inp))

    def forward(self, inp):
        return 1 / (1 + np.exp(-inp))

    def backward(self, inp, grad_outp):
        grad_outp1=np.ndarray(shape=(len(inp),))
        for i in range(len(grad_outp)):
            grad_outp1[i]=np.sum(grad_outp[i])
        return grad_outp1*self.derivateActivation(inp)


class Dense(Layer):
    def __init__(self, inp_units, outp_units, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(
            loc=0.0, scale=np.sqrt(2 / (inp_units + outp_units)),
            size=(inp_units, outp_units))
        self.neuron_count = outp_units
        self.updatesWeights = np.copy(self.weights)
        self.biases = np.zeros(outp_units)

    def update_weights(self):
        for x in range(len(self.weights)):
            for y in range(self.get_neuron_count()):
                self.weights[x][y]+=self.updatesWeights[x][y]

    def get_neuron_count(self):
        return self.neuron_count

    def forward(self, inp):
        return np.dot(inp, self.weights) + self.biases

    def backward(self, inp, grad_outp):
        for i in range(len(inp)):
            for x in range(self.get_neuron_count()):
                self.updatesWeights[i][x]=self.learning_rate*grad_outp[x]*inp[i]
        return self.weights*grad_outp




class MLP():
    def __init__(self):
        self.layers = []

    def add_layer(self, neuron_count, inp_shape=None, activation='ReLU'):
        if len(self.layers) == 0 and inp_shape is None:
            raise ValueError("Must defined input shape for first layer")

        if inp_shape is None:
            inp_shape = self.layers[-2].get_neuron_count()

        self.layers.append(Dense(inp_shape, neuron_count))
        if activation == 'sigmoid':
            self.layers.append(Sigmoid())
        elif activation == 'ReLU':
            self.layers.append(ReLU())
        else:
            raise ValueError("Unknown activation function", activation)

    def forward(self, X):
        activations = []
        layer_input = X
        for l in self.layers:

            activations.append(l.forward(layer_input))
            layer_input = activations[-1]

        return activations

    def predict(self, X):
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)

    def fit(self, X, Y,epohs=1000):
        good = 0
        all = 0
        errors = []
        for epoch in range(epohs+1):
            for i in zip(X,Y):
                outputs=self.forward(i[0])
                error=self.layers[-1].calculateError(outputs[-1],i[1])
                error=error*self.layers[-1].derivateActivation(outputs[-2])
                error=self.layers[-2].backward(outputs[-3],error)
                if self.predict(i[0])==np.argmax(i[1],axis=-1):
                    good+=1
                all+=1
                errors.append(error_square(outputs[-1],i[1]))
                for x in range(-3,-len(self.layers),-2):
                    error=self.layers[x].backward(outputs[x-1],error)
                    if x==-len(self.layers)+1:
                        error = self.layers[x - 1].backward(i[0], error)
                    else:
                        error=self.layers[x-1].backward(outputs[x-2],error)

                for layer in self.layers:
                    if isinstance(layer,Dense):
                        layer.update_weights()
            if epoch%50==0 and epoch!=0:
                print("Epoch {}: Accuracy={},Error={}".format(epoch,round(good/all,4),sum(errors)/len(errors)))
                good=0
                all=0
                errors=[]
        print("End of training.")


                        


        


if __name__ == '__main__':
    network = MLP()
    network.add_layer(6, 4, activation='sigmoid')
    network.add_layer(7, activation='sigmoid')
    network.add_layer(3, activation='sigmoid')
    network.fit(X_train,y_train)
    print("Tests:")
    y_pred = network.predict(X_test)

    y_test_class = np.argmax(y_test, axis=1)

    from sklearn.metrics import classification_report,confusion_matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test_class, y_pred))
    print("Classification report:")
    print(classification_report(y_test_class, y_pred))

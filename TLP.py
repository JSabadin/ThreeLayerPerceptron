import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import itertools


class TLP():
    def __init__(self,X,Y,N_l2,epochs,Beta):
        """@param X: are train samples (np.array)
           @param Y: are train samples (np.array) 
           @param N_l2: number of neurons in hidden layer
           @param epochs: num of epochs in backpropagation algorithm
           @param Beta: learning rate wich is <1

            How to use:
           @param train() -> trains the train data
           @param test(self,X,Y) -> tests the test data
           @param plt_results(self,mode) -> plots the results
        """

        self.X = X   # Smaples
        self.Y = Y   # Labels of samples
        self.X_test = 0
        self.y_test = 0
        self.N_l2 = N_l2 # Num of neurons in hidden layer
        self.n_samples = self.X.shape[0]
        self.layer_sizes = np.array([self.X.shape[1]]+[self.N_l2-1]+[self.Y.shape[1]]) #We pay attention to bias [N_l2-1]
        self.init_weights()
        # Lerning parameters
        self.epochs = epochs  # repetition of backrpopagation
        self.Beta = Beta   # learning rate
        # For validation of training and testing set
        self.val_acc = []
        self.train_acc = []
        self.test_acc = []


    # MAT FUNKCIJE
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
        
    def sigmoid_derivate(self,x):
        return x*(1-x)

     
    def init_weights(self):
        # Initialization of weights and bias
        self.w = []
        self.b = []
        for i in range(self.layer_sizes.shape[0]-1):
            # weights are set as random numbers betwen -0.5 in 0.5
            self.w.append(np.random.uniform(-0.5,0.5,size=[self.layer_sizes[i+1],self.layer_sizes[i]]))
            self.b.append(np.zeros((self.layer_sizes[i+1],1)))
        self.w = np.array(self.w, dtype=object)
        self.b = np.array(self.b, dtype=object)

    def init_layers(self):
        # Initialization of layers
        self.x = [np.empty((layer,1)) for layer in self.layer_sizes]

    def forward_propagation(self,sample):
        x_l = sample
        self.x[0] = x_l  
        for i in range(len(self.w)):
            x_l = self.sigmoid(np.matmul(self.w[i],self.x[i]) + self.b[i])
            self.x[i+1] = x_l 
        
    def backward_propagation(self,y_target):
        d = (y_target - self.x[-1])*self.sigmoid_derivate(self.x[-1]) # * is elementwise operator! thats what we want!
        for i in range(1,len(self.w)+1):
            delta_w = self.Beta*np.matmul(d,np.transpose(self.x[-i-1]))
            delta_b = self.Beta*d
            self.w[-i] += delta_w
            self.b[-i] += delta_b
            d = self.sigmoid_derivate(self.x[-i-1])*np.matmul(np.transpose(self.w[-i]),d)


    def plt_results(self):
        '''    
                @param "train": then the train results will be plotet
                @param "test": then the test results will be plotet 
                plt.show() has to be used afterward
        '''
        epochs = np.arange(1,self.epochs+1)
        if(isinstance(self.X_test, (np.ndarray)) & isinstance(self.y_test, (np.ndarray))):
            plt.figure()
            plt.title("Training and testing accuarcy") 
            plt.xlabel("epochs") 
            plt.ylabel("accuarcy in %") 
            plt.plot(epochs,np.array(self.train_acc)*100, label = f"Train, Beta={self.Beta}, N_l2 ={self.N_l2} ")
            plt.plot(epochs,np.array(self.test_acc)*100, label = f"Test, Beta={self.Beta}, N_l2 ={self.N_l2}")
            plt.legend()
            plt.show()
        else:
            plt.figure()
            plt.title("Training accuarcy") 
            plt.xlabel("epochs") 
            plt.ylabel("accuarcy in %") 
            plt.plot(epochs,np.array(self.train_acc)*100, label = f"Train, Beta={self.Beta}, N_l2 ={self.N_l2} ")
            plt.legend()
            plt.show()


    def fit(self, X_test,y_test):
        # To fit new test data!
        self.X_test = X_test
        self.y_test = y_test
        
    def evaluate(self,X,Y):
        # returns 1 if X  is clasified as Y and 0 if not.
        prediction = self.predict(X)
        return int(np.all(Y==prediction))

    def predict(self,X):
        # We retrun prediciton of class for. X has to be np.array!
        X_ = X.reshape(self.layer_sizes[0],1)
        self.init_layers()
        self.forward_propagation(X_)
        prediction = self.to_categorical(np.transpose(self.x[-1]))
        return prediction
    
    def to_categorical(self,y):  
        # We retrun vector with one 1 and all other elements 0. For example [1 0 0 0] 
        categorical = np.zeros_like(y.reshape(self.layer_sizes[-1]))
        categorical[np.argmax(y)] = 1
        return categorical

    def train(self):
        print("training has started:")
        for epoch in tqdm(range(1,self.epochs+1)):
            self.y_predicted = []
            self.init_layers()
            train_acc = 0
            test_acc = 0
            for feature, name in zip(self.X,self.Y):
                # For every sample we do forward and backward propagation
                self.forward_propagation(feature.reshape(self.layer_sizes[0],1)) # (2,) ->(2,1) pretvorba v vektor
                self.backward_propagation(name.reshape(self.layer_sizes[-1],1))   # ista pretvorba
                train_acc += self.evaluate(feature,name)
            train_acc = train_acc/len(self.X)
            self.train_acc.append(train_acc)
            # if we added test set:
            if(isinstance(self.X_test, (np.ndarray)) & isinstance(self.y_test, (np.ndarray))):
                for feature, name in zip(self.X_test,self.y_test):
                    test_acc += self.evaluate(feature,name)
                    self.y_predicted.append(self.predict(feature))
                test_acc = test_acc/len(self.X_test)
                self.test_acc.append(test_acc)
                print( f" training acc = {int(self.train_acc[-1]*100)}%,testing acc = {int(self.test_acc[-1]*100)}%, epoch = {epoch}")
                # if we didnt add test list
            else:  
                print( f" training acc = {int(self.train_acc[-1]*100)}%, epoch = {epoch}")
        print(" training is finished, all weights are corrected")
        

    def plt_confusion_matrix_test_data(self,axes_names):
        cm = np.zeros((self.layer_sizes[-1], self.layer_sizes[-1])) 
        y_test = [np.argmax(self.y_test[i]) for i in range(len(self.y_test))]
        y_predicted =[np.argmax(self.y_predicted[i]) for i in range(len(self.y_predicted))]
        for test,true in zip(y_test,y_predicted):
            cm[test][true] +=1
        for i in range(len(axes_names)):
            cm[i,:]  =  cm[i,:]/np.sum(cm[i,:])
        plt.imshow(cm)
        plt.xticks(np.arange(len(axes_names)), axes_names)
        plt.yticks(np.arange(len(axes_names)), axes_names)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    
            

    # We need categorical function to transform class number into vector
def transform_to_vector(y,num_class):  
    # We retrun vector with one 1 and all other elements 0. For example [1 0 0 0] 
    vector = np.zeros(num_class)
    vector[y-1] = 1
    return vector    




if __name__ == "__main__":

    ############################################################################################################################################
    # XOR Problem
    X_train = np.array([[0,0], [0,1],[1,0],[1,1]])
    y_train = np.array(([[0,1],[1,0],[1,0],[0,1]]))  

    # Defining and training our network
    ann = TLP(X_train,y_train,4,7000,0.2)
    ann.train()  

    #Showing results of our network
    for x,y in zip(X_train,y_train):
        print(f"sample {x}, marked as {y}, is classified as {ann.predict(x)}  ")
    ann.plt_results()
    
    # ############################################################################################################################################
    # WORDS PROBLEM
    # Reading train and teset data and saving it into np.array
    read_csv_train = pd.read_csv('isolet1+2+3+4.csv',header=None)              
    all_train_data = np.array([read_csv_train.iloc[i] for i in range(len(read_csv_train))])# we sort data into 6238 sets, each long 618.
    read_csv_test = pd.read_csv('isolet5.csv',header=None) 
    all_test_data = np.array([read_csv_test.iloc[i] for i in range(len(read_csv_test))])# we sort data into 1559 sets, each long 618.
    # A-Z -> 26 classes
    num_class = int(np.max(np.array(read_csv_train.iloc[: , -1]))) #Number of classes

    # Saving data for classifcation problem
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for data_train in all_train_data:
        X_train.append((data_train[:-1]))
        y_train.append(transform_to_vector(int(data_train[-1]),num_class))
    X_train,y_train = np.array(X_train),np.array(y_train)

    for data_test in all_test_data:
        X_test.append((data_test[:-1]))
        y_test.append(transform_to_vector(int(data_test[-1]),num_class))
    X_test,y_test = np.array(X_test),np.array(y_test)

   # DEFINING AND TRAINING OUR NETWORK WITH PARAMETERS WICH OBTAINED BEST EXPERIMENT RESULTS
    ann = TLP(X_train,y_train,140,15,0.3)
    ann.fit(X_test,y_test)
    ann.train()  
    ann.plt_confusion_matrix_test_data(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
    ann.plt_results()



    #############################################################################################################################################
    # TRAINING AND TESTING OUR DATA ON NEW DATASET
    # SECURITY SUSPICIOUS SOUNDS
    with open('test.npy', 'rb') as f:
        X_train= np.load(f)
        y_train = np.load(f)

    with open('test.npy', 'rb') as f:
        X_test = np.load(f)
        y_test= np.load(f)

    ann = TLP(X_train,y_train,140,20,0.3)
    ann.fit(X_test,y_test)
    ann.train()
    ann.plt_results()
    cm_plot_labels = ["alarm","dog_bark","explosion","glass_breaking","scream","shooting","siren"]
    ann.plt_confusion_matrix_test_data(cm_plot_labels)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class RBFN(object):

    def __init__(self, hidden_shape, sigma=1.0):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def _calculate_interpolation_matrix(self, X):
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        G = np.zeros((len(X), self.hidden_shape))
        #for data_point_arg, data_point in enumerate(X):
        #    for center_arg, center in enumerate(self.centers):
        #        G[data_point_arg, center_arg] = self._kernel_function(center, data_point)
        
        for data_point_arg, data_point in enumerate(X):
            for i in range(len(model.centers)):
                G[data_point_arg, i] = np.exp(self.sigma[i]*np.linalg.norm(model.centers[i]-data_point)**2)
        return G

    def _select_centers(self, X,Y):
        """ Random choose"""
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        """ linear equal distribuited """
        #centers = np.linspace(0, len(X), self.hidden_shape)
        """ K_means to choose centers """
        #print(X.shape)
        #time.sleep(5)
    
        kmeans = KMeans(n_clusters=self.hidden_shape, random_state=0).fit(X)
        kmeans.labels_
        self.centers = kmeans.cluster_centers_
        for i in range(len(centers)):
            self.sigma[i] = self.centers[i]/max(abs(centers))
        return centers

    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers = self._select_centers(X,Y)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions

y = total.reshape(-1,1)[0:10]

x = np.array(range(len(y)))[:, None]    
for i in range(2,100,1):
    x = np.linspace(-10,0, 100)[:, None]       # [batch, 1]
    y = np.cos(x)*x 
    sigma = np.ones(i)
    model = RBFN(hidden_shape=i, sigma=sigma )    
    model.fit(x,y)
    y_hat = model.predict(x)
    x_center = []
    y_center = []
    for j in range(len(model.centers)):
        y_center.append(y[np.round(x,2) == np.round(model.centers[j],2)])
        x_center.append((x[np.round(x,2) == np.round(model.centers[j],2)])[0])
    plt.scatter(x_center,y_center)
    plt.plot(x,y_hat)
    plt.plot(x,y)
    #plt.savefig(str(i)+'neurons'+'.jpg')
    plt.clf()
    
    plt.scatter(x_center,y_center)
    plt.plot(x,y_hat)
    plt.plot(x,y)
    x = np.linspace(0, 10, 1000)[:, None]       # [batch, 1]
    y = np.cos(x)*x 
    y_hat = model.predict(x)
    plt.plot(x,y_hat)
    plt.plot(x,y)
    plt.savefig('img/RBF/'+str(i)+'extrapolation_neurons'+'.jpg')
    plt.clf()
 
    

'''

k = np.exp(sigma*np.linalg.norm(model.centers-model.data_point)**2) #kernel function
model.centers
G = np.zeros((len(X), 49))
for data_point_arg, data_point in enumerate(X):
    for i in range(len(model.centers)):
        G[data_point_arg, i] = np.exp(sigma[i]*np.linalg.norm(model.centers[i]-data_point)**2)

center_arg, center
weig = np.dot(np.linalg.pinv(G), y)
pred = np.dot(G,weig)

y_hat = model.predict(x)
plt.plot(y_hat)
plt.plot(y)
plt.plot(pred)
x = np.linspace(1, 3, 100)[:, None] 
y = x ** 3  

'''



 




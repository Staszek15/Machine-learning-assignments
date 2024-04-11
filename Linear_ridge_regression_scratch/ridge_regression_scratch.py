import numpy as np
import pytest
from sklearn.linear_model import Ridge

class RidgeRegr:

    def __init__(self, alpha = 0.0, learning_rate=0.005, M=100000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.M = M

    def fit(self, X, Y):
        # input:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Finds theta (approximately) minimising quadratic loss function L with Ridge penalty,
        # using an iterative method.
        n, m = X.shape
        X = np.hstack((np.ones((n, 1)), X))   # add column of 1
        self.theta = np.zeros((1,m+1))  # (1,2)

        for _ in range(self.M):
          
            #print("X shape", X.shape)  #(4,2)
            #print("self.theta shape", self.theta.shape)  #(1,2)
            #print("Y.reshape shape", Y.reshape(-1,1).shape)  #(4,1)
            
            errors = (Y.reshape(-1,1) - X @ self.theta.T)
            #print("errors",  errors.shape)
            
            gradient_theta_0 = -2 * np.sum(errors, axis=0)[0]  # 0 take float from array
            #print("gradient0 shape", gradient_theta_0.shape)
            
            gradient_theta_rest = -2 * errors.T @ X[:,1:] + (2*self.alpha*self.theta.T[1:]).T  #(1,2)+(1,2)
            #print("gradient_rest shape", gradient_theta_rest.shape) #(1,2)
            
            #concatenate gradients
            gradient = np.insert(gradient_theta_rest, 0, gradient_theta_0, axis=1)
            #print("gradient shape", gradient.shape)
            #print("theta shape", self.theta.shape)
            
            #update thetas
            self.theta -= self.learning_rate * gradient

        return self
    
    def predict(self, X):
        # input:
        #  X = np.array, shape = (k, m)
        # returns:
        #  Y = wektor(f(X_1), ..., f(X_k))
        k, m = X.shape
        X = np.column_stack((np.ones(k), X))
        predicted = X @ self.theta.T
        print(predicted.shape)
        return predicted.T[0]




####################### TEST #######################

def test_RidgeRegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,2,10]).reshape((3,1))
    alpha = 0.3
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    print("Expected:", expected)
    print("Actual:", actual)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)

def test_RidgeRegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    alpha = 0.4
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    print("Expected:", expected)
    print("Actual:", actual)
    assert list(actual) == pytest.approx(list(expected), rel=1e-8)
    



if __name__ == "__main__":
    test_RidgeRegressionInOneDim()
    test_RidgeRegressionInThreeDim()
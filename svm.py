import numpy as np
import cvxopt

class SVM(object):
    def __init__(self, s_address, l_address):
        self.s_address = s_address
        self.l_address = l_address
        '''
        Here the sample and label means the location of sample and label
        We will use the location to open the dataset
        '''
        self._read_data()
        self._split_data()

    def _read_data(self):
        self.sample = np.loadtxt(self.s_address, dtype = float, delimiter = ',')
        self.label = np.loadtxt(self.l_address, dtype = float, delimiter = ',')
        #self.sample = self.sample[:4123]
        #self.label = self.label[:4123]

    def _split_data(self):
        n_samples, n_features = self.sample.shape
        sample_size = int(0.7 * n_samples)
        self.test_sample = self.sample[sample_size:]
        self.sample = self.sample[:sample_size]
        self.test_label = self.label[sample_size:]
        self.label = self.label[:sample_size]

    def training(self):
        self.kernel = kernel
        self.parameter = parameter
        n_samples, n_features = self.sample.shape
        print(n_samples)
        
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            if i % 100 == 0:
                print("Work until i ", i, " out of total num ", n_samples)
            for j in range(i, n_samples):
                tmp = np.dot(self.sample[i], self.sample[j])
                K[i,j] = tmp
                K[j,i] = tmp
    
        P = cvxopt.matrix(np.outer(self.label,self.label) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(self.label, (1,n_samples))
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))
        
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(sol['x'])
        
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv_x = self.sample[sv]
        self.sv_y = self.label[sv]
        
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n] - np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)
        
        if self.kernel == 'Linear':
        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv_x[n]


    def predict(self, X):
        result = np.dot(X, self.w) + self.b
        return np.sign(result)


    def testing(self, X, y):
        pred = self.predict(X)
        rtn = 0
        for i in range(X.shape[0]):
            if pred[i] == y[i]:
                rtn += 1
        return rtn * 1.0 / X.shape[0]




class KernelSVM(SVM):
    def training(self, kernel = 'Linear', parameter = 1.0):
        self.kernel = kernel
        self.parameter = parameter
        n_samples, n_features = self.sample.shape
        print(n_samples)

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            if i % 100 == 0:
                print("Work until i ", i, " out of total num ", n_samples)
            for j in range(i, n_samples):
                tmp = self.kernel_function(self.sample[i], self.sample[j])
                K[i,j] = tmp
                K[j,i] = tmp
                '''
                K[i,j] = self.kernel_function(self.sample[i], self.sample[j])
                '''

        P = cvxopt.matrix(np.outer(self.label,self.label) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(self.label, (1,n_samples))
        b = cvxopt.matrix(0.0)
        G, h = self.generate_Gh(n_samples)

        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(sol['x'])

        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv_x = self.sample[sv]
        self.sv_y = self.label[sv]

        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n] - np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)
        
        if self.kernel == 'Linear': 
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv_x[n]

    def generate_Gh(self, n_samples):
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))
        return G, h

    def predict(self, X):
        if self.kernel == 'Linear':
            result = np.dot(X, self.w) + self.b
            return np.sign(result)
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv_x):
                    s += a * sv_y * self.kernel_function(X[i], sv)
                y_predict[i] = s
            result = y_predict + self.b
            return np.sign(result)

    def testing(self, X, y):
        pred = self.predict(X)
        rtn = 0
        for i in range(X.shape[0]):
            if pred[i] == y[i]:
                rtn += 1
        return rtn * 1.0 / X.shape[0]

    def kernel_function(self, x, y):
        '''
        d only used by Polynominal
        sigma only used by Gaussian and Laplace
        self.parameters representing both d and sigma
        '''
        if self.kernel == 'Linear':
            return np.dot(x, y)
        elif self.kernel == 'Gaussian':
            return np.exp(-1 * (np.linalg.norm(x-y)**2) / (2 * (self.parameter ** 2)))
        elif self.kernel == 'Laplace':
            return np.exp(-1 * np.linalg.norm(x-y) / self.parameter)
        else:
            return np.dot(x, y) ** self.parameter

 


class SoftMarginSVM(KernelSVM):
    def input_Margin(self, C):
        self.C = C

    def generate_Gh(self, n_samples):
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        return G,h


class ISVM(SoftMarginSVM):
    def online_training(self, kernel = 'Linear', parameter = 1.0):
        n_samples, n_featurs = self.sample.shape
        iter_num = int(n_samples / 2000)
        if n_samples % 2000 != 0:
            iter_num += 1
        self.kernel = kernel
        self.parameter = parameter
        if n_samples <= 2000:
            self.training(kernel, parameter)
            return
        for x in range(iter_num):
            print("start iteration number: ", x)
            if x == 0:
                this_sample = np.copy(self.sample[:2000])
                this_label = np.copy(self.label[:2000])
                t_samples, t_features = this_sample.shape
                K = np.zeros((t_samples, t_samples))
                for i in range(t_samples):
                    for j in range(i, t_samples):
                        tmp = self.kernel_function(this_sample[i], this_sample[j])
                        K[i,j] = tmp
                        K[j,i] = tmp

                P = cvxopt.matrix(np.outer(this_label,this_label) * K)
                q = cvxopt.matrix(np.ones(t_samples) * -1)
                A = cvxopt.matrix(this_label, (1,t_samples))
                b = cvxopt.matrix(0.0)
                G, h = self.generate_Gh(t_samples, 0)

                sol = cvxopt.solvers.qp(P, q, G, h, A, b)
                a = np.ravel(sol['x'])

                sv = a > 1e-5
                ind = np.arange(len(a))[sv]
                self.a = a[sv]
                self.sv_x = this_sample[sv]
                self.sv_y = this_label[sv]
                sv_num = self.sv_x.shape[0]
            else:
                if x != iter_num -1:
                    this_sample = np.copy(self.sample[x*2000:x*2000+2000])
                    this_label = np.copy(self.label[x*2000:x*2000+2000])
                else:
                    this_sample = np.copy(self.sample[x*2000:])
                    this_label = np.copy(self.label[x*2000:])
                this_sample = np.concatenate((this_sample, self.sv_x), axis = 0)
                this_label = np.concatenate((this_label, self.sv_y), axis = 0)
                t_samples, t_features = this_sample.shape
                K = np.zeros((t_samples, t_samples))
                for i in range(t_samples):
                    for j in range(i, t_samples):
                        tmp = self.kernel_function(this_sample[i], this_sample[j])
                        K[i,j] = tmp
                        K[j,i] = tmp

                P = cvxopt.matrix(np.outer(this_label,this_label) * K)
                q = cvxopt.matrix(np.ones(t_samples) * -1)
                A = cvxopt.matrix(this_label, (1,t_samples))
                b = cvxopt.matrix(0.0)
                G, h = self.generate_Gh(t_samples, sv_num)

                sol = cvxopt.solvers.qp(P, q, G, h, A, b)
                a = np.ravel(sol['x'])
                a_copy = np.copy(a)
                fil = a[(a_copy.argsort()[::-1][0:1000])[-1]]
                sv = a > fil
                ind = np.arange(len(a))[sv]
                self.a = a[sv]
                self.sv_x = this_sample[sv]
                self.sv_y = this_label[sv]
                sv_num = self.sv_x.shape[0]

        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n] - np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)
        
        if self.kernel == 'Linear': 
            self.w = np.zeros(t_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv_x[n]

    def generate_Gh(self, t_samples, sv_num):
        if sv_num == 0:
            L = 0
        else:
            L = (t_samples - sv_num) / sv_num * 1.0
        tmp1 = np.diag(np.ones(t_samples) * -1)
        tmp2 = np.identity(t_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(t_samples)
        tmp2 = np.ones(t_samples) * self.C
        for i in range(sv_num):
            tmp2[-1 * i - 1] = tmp2[-1 * i - 1] / self.C * L
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        return G,h
            
            


if __name__ == "__main__":
    '''
    model = KernelSVM("data/sample_1.csv.xls", "data/label_1.csv.xls")
    model.training()
    print(model.testing(model.test_sample, model.test_label))

    Linear = KernelSVM("data/sample_2.csv.xls", "data/label_2.csv.xls")
    Linear.training(kernel = 'Linear')
    print(Linear.testing(Linear.test_sample, Linear.test_label))
    
    Gaussain = KernelSVM("data/sample_2.csv.xls", "data/label_2.csv.xls")
    Gaussain.training(kernel = 'Gaussian', parameter = 5.0)
    print(Gaussain.testing(Gaussain.test_sample, Gaussain.test_label))
    
    Laplace = KernelSVM("data/sample_2.csv.xls", "data/label_2.csv.xls")
    Laplace.training(kernel = 'Laplace', parameter = 5.0)
    print(Laplace.testing(Laplace.test_sample, Laplace.test_label))

    Polynominal = KernelSVM("data/sample_2.csv.xls", "data/label_2.csv.xls")
    Polynominal.training(kernel = 'Laplace', parameter = 2.0)
    print(Polynominal.testing(Polynominal.test_sample, Polynominal.test_label))

    Soft = SoftMarginSVM("data/sample_3.csv.xls", "data/label_3.csv.xls")
    Soft.input_Margin(C = 0.1)
    Soft.training(kernel = 'Gaussian', parameter = 5.0)
    print(Soft.testing(Soft.test_sample, Soft.test_label))
    '''

    I = ISVM("data/sample_3.csv.xls", "data/label_3.csv.xls")
    I.input_Margin(0.1)
    I.online_training(kernel = 'Gaussian', parameter = 5.0)
    print(I.testing(I.test_sample, I.test_label))



import numpy as np
from utilities import prox, get_gradient_sum, huber, rbf, calcK, createX, calcL, get_gradient, update_alpha, update_alpha2, calcK_pred, update_A, prox_box
import sys
import random

class SpHAM:
    T = 0
    start = 0
    end = 0
    interval = 0
    p = 0
    m = 0
    back_t = 0
    y = 0
    y_true = 0
    y_orig = 0
    y_true_orig = 0
    rbf_sigma = 2.
    huber_sigma = 0
    lamb = 0.01
    X = 0

    def __init__(self, y, y_true, huber_sigma, start=0, interval=100, backward=1, rbf_sigma=1., lamb=0.0001):
        self.m = y.shape[1]
        self.p = self.m * backward
        self.y_orig = y
        self.y_true_orig = y_true
        self.y = y[start:start+interval]
        self.y_true = y_true[start:start+interval]
        self.start = start
        self.interval = interval
        self.end = start + interval
        self.back_t = backward
        self.rbf_sigma = rbf_sigma
        self.lamb = lamb
        self.T = interval
        self.huber_sigma = huber_sigma
        '''if square_loss:
            self.huber_sigma = 99999999999
        else:
            self.huber_sigma = np.sqrt(self.T)'''


    def fit(self, max_iter=1000, tol=1e-8, verbose=False, square_loss=False):
        # s: 1/T for stationary problems
        # L: the Lipschitz constant of gradient of f
        # lamb: the regularization parameter
        # max_iter: maxmimum number of iterations
        # tol: the algorithm will stop if difference between two successive alpha is smaller than this value
        # alpha_old = 0.1 * np.random.rand(T*p, 1)

        big_alpha = np.ones((self.T * self.p, self.m))
        X = createX(self.y_orig, self.start, self.interval, self.p, self.m, self.back_t)
        self.X = X
        K = calcK(X, self.T, self.p, self.rbf_sigma)
        L = calcL(K, self.T)
       # print("Training:")
        for i in range(self.m):
            alpha_old = np.ones((self.T * self.p, 1))

            t_old = 1
            iter = 1
            y_old = alpha_old.copy()

            while iter <= max_iter:
                alpha_new = update_alpha(y_old, self.y[:,i,:], L, self.T, self.p, self.lamb, self.huber_sigma, K, square_loss)
                t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old ** 2))
                #print("t_new:",t_new.shape)
                y_new = alpha_new + (t_old - 1) / t_new * (alpha_new - alpha_old)
                #print("y_new.shape:",y_new.shape)

                prediction = np.empty(self.T)
                for t in range(self.T):
                    prediction[t] = K[:, t].T @ alpha_new

                #print("prediction.shape",prediction.shape)
                loss1 = np.linalg.norm(prediction - self.y[:,i,0]) / self.T
                loss2 = np.linalg.norm(prediction - self.y_true[:,i,0]) / self.T


                if verbose and (iter == 1 or (iter >= 10 and iter % 10 == 0)):
                    print("dimension %d, iteration: %d" % (i, iter))
                    print("loss : %.8f" % (loss1))
                    print("loss with ground truth: %.8f\n" % (loss2))

                # check with tolerence
                e = np.linalg.norm((alpha_new - alpha_old), ord=1) / (self.T * self.p)
                if e < tol:
                    break

                # update
                alpha_old = alpha_new
                t_old = t_new
                y_old = y_new
                iter += 1
            big_alpha[:,i] = alpha_old[:,0]
        return big_alpha

    # y and y_true are the entire ys, start is the initial time that we want to predict
    def predict(self, big_alpha, y, y_true, start, repeat, backward=1, verbose=False):
        my_y = y[start - backward:start, :, :]
        my_y2 = my_y.flatten()


        big_prediction = np.empty((repeat, self.m))
        for count in range(repeat):
            prediction = np.empty(self.m)
            for i in range(self.m):

                k = np.empty((self.T * self.p))
                for j in range(self.p):
                    for t in range(self.T):
                        #k[j*self.T + t] = rbf(self.X[t][j], my_y[:,j,:] , self.rbf_sigma)
                        k[j * self.T + t] = rbf(self.X[t][j], my_y2[j], self.rbf_sigma)

                prediction[i] = k @ big_alpha[:,i]

            if verbose:
                print("prediction:",prediction)

            loss1 = np.linalg.norm(prediction - y[start+count, :, 0])
            loss2 = np.linalg.norm(prediction - y_true[start+count, :, 0])

            if verbose:
                print("At time %d dimension %d" % (start+count, i))
                print("loss: %.8f" % loss1)
                print("loss with ground truth: %.8f\n" % loss2)

            big_prediction[count] = prediction

    #        my_y2_temp = my_y2.copy()

    #        for tp in range(len(my_y2) - self.m):
    #            my_y2[tp] = my_y2_temp[tp + self.m]


    #        prediction_idx = 0
    #        for tp in range(len(my_y2) - self.m, len(my_y2)):
    #            my_y2[tp] = prediction[prediction_idx]
    #            prediction_idx += 1

            my_y = y[start + count + 1 - backward:start + count + 1, :, :]
            my_y2 = my_y.flatten()

        return big_prediction




class nonStationary:
    T = 0
    start = 0
    end = 0
    interval = 0
    p = 0
    m = 0
    back_t = 0
    y = 0
    y_true = 0
    y_orig = 0
    y_true_orig = 0
    rbf_sigma = 2.
    huber_sigma = 0
    lamb = 0.01
    X = 0
    q = 0
    s = 0
    lamb1 = 0
    lamb2 = 0
    stepA_cons = 0
    l = 0
    s_iter = 0
    dccp_iter = 0
    dccp_tau = 0

    def __init__(self, y, y_true, l, huber_sigma, start=0, interval=100, backward=1, rbf_sigma=1., lamb=0.001, lamb1=100., lamb2=20., lambA=1000,  s_iter=5, out_iter=5):
        self.m = y.shape[1]
        self.p = self.m * backward
        self.y_orig = y
        self.y_true_orig = y_true
        self.y = y[start:start + interval]
        self.y_true = y_true[start:start + interval]
        self.start = start
        self.interval = interval
        self.end = start + interval
        self.back_t = backward
        self.rbf_sigma = rbf_sigma
        self.lamb = lamb
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.lambA = lambA
        self.c = 1
        self.T = interval
        self.huber_sigma = huber_sigma
        self.q = np.empty(interval)
        self.q[:interval-l] = 0.
        self.q[-l:] = 1./l
        self.l = l
        self.s = np.ones(interval)
        self.s_iter = s_iter
        self.out_iter = out_iter

    def fit(self, lr=0.00005, max_iter=1000, tol=1e-8, verbose=False, square_loss=False):
        # s: 1/T for stationary problems
        # L: the Lipschitz constant of gradient of f
        # lamb: the regularization parameter
        # max_iter: maxmimum number of iterations
        # tol: the algorithm will stop if difference between two successive alpha is smaller than this value

        # alpha_old = 0.1 * np.random.rand(T*p, 1)
        big_alpha = np.ones((self.T * self.p, self.m))
        X = createX(self.y_orig, self.start, self.interval, self.p, self.m, self.back_t)
        self.X = X
        K = calcK(X, self.T, self.p, self.rbf_sigma)
        L = calcL(K, self.T)


        for dim in range(self.m):
            # Step A
            s_iter = 1
            s_old =  np.zeros(self.T)
           # s_old = np.ones(self.T)
            sys.stdout.flush()
            while s_iter <= self.s_iter:
                # alpha
                alpha_old = 0.1*np.ones((self.T * self.p, 1))
                alpha_new = 0.1*np.ones((self.T * self.p, 1))
                iter = 1
                while iter <= self.out_iter:
                    grad = get_gradient_sum(K, alpha_old, self.y, self.huber_sigma, self.T, self.p, range(self.interval), self.q-s_old)
                    temp_alpha = alpha_old + lr * (grad - 2*self.lambA * alpha_old)
                    value = np.linalg.norm(temp_alpha-alpha_old, 2)
                    alpha_old = temp_alpha


                    # check with tolerence
                    e = np.linalg.norm((alpha_new - alpha_old), ord=1) / (self.T * self.p)
                    if e < tol:
                        break
                    # update
                    alpha_new = alpha_old.copy()
                    iter += 1
                s_iter += 1
                # update A
                A = update_A(self.T, self.p, self.X, alpha_new, self.y[:,dim,:], self.rbf_sigma, self.huber_sigma)

                # update s
                s_new = s_old - lr * (-A[:,0] - 2*self.lamb1 * (self.q - s_old) + 2*self.lamb2 * s_old)
                s_new = prox_box(s_new, self.q, self.T)

                sum = 0
                for t in range(self.T):
                    sum += A[t,0] * (s_new[t] - self.q[t])

                sum += self.lamb1  * np.linalg.norm(self.q - s_new, 2) + self.lamb2  * np.linalg.norm(s_new,2)


                sys.stdout.flush()

                # check if s converged
                e = np.linalg.norm((s_new - s_old), ord=1)
                if e < 0.000001:
                    break
                s_old = s_new
                s_iter += 1
                sys.stdout.flush()

            sys.stdout.flush()


            # step B
            alpha_old =  np.ones((self.T * self.p, 1))
      #      print('the value of s', s_old[100:120])
            t_old = 1
            iter = 1
            y_old = alpha_old.copy()


            while iter <= max_iter:
                alpha_new = update_alpha2(y_old, self.y[:,dim,:], L, self.T, self.p, self.lamb, self.huber_sigma, K, square_loss, s_old)
                t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old ** 2))
                y_new = alpha_new + (t_old - 1) / t_new * (alpha_new - alpha_old)

                prediction = np.empty(self.T)
                for t in range(self.T):
                    prediction[t] = K[:, t].T @ alpha_new

                loss1 = np.linalg.norm(prediction - self.y[:,dim,0]) / self.T
                loss2 = np.linalg.norm(prediction - self.y_true[:,dim,0]) / self.T

                if verbose and (iter == 1 or (iter >= 10 and iter % 10 == 0)):
                    print("dimension %d, iteration: %d" % (dim, iter))
                    print("loss : %.8f" % (loss1))
                    print("loss with ground truth: %.8f\n" % (loss2))

                # check with tolerence
                e = np.linalg.norm((alpha_new - alpha_old), ord=1)/(self.T * self.T)
                if e < tol:
                    break

                # update
                alpha_old = alpha_new
                t_old = t_new
                y_old = y_new
                iter += 1
            big_alpha[:,dim] = alpha_old[:,0]
        return big_alpha

    def predict(self, big_alpha, y, y_true, start, repeat, backward=1, verbose=False):
        my_y = y[start - backward:start, :, :]
        my_y2 = my_y.flatten()

        big_prediction = np.empty((repeat, self.m))
        for count in range(repeat):
            prediction = np.empty(self.m)
            for i in range(self.m):

                k = np.empty((self.T * self.p))
                for j in range(self.p):
                    for t in range(self.T):
                        #k[j*self.T + t] = rbf(self.X[t][j], my_y[:,j,:] , self.rbf_sigma)
                        k[j * self.T + t] = rbf(self.X[t][j], my_y2[j], self.rbf_sigma)

                prediction[i] = k @ big_alpha[:,i]

            if verbose:
                print("prediction:",prediction)

            #print("diff.shape",diff.shape)
            loss1 = np.linalg.norm(prediction - y[start+count, :, 0])
            loss2 = np.linalg.norm(prediction - y_true[start+count, :, 0])

            if verbose:
                print("At time %d dimension %d" % (start+count, i))
                print("loss: %.8f" % loss1)
                print("loss with ground truth: %.8f\n" % loss2)

            big_prediction[count] = prediction



            my_y = y[start + count + 1 - backward:start + count + 1, :, :]
            my_y2 = my_y.flatten()


        return big_prediction

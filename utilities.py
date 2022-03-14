import numpy as np

def write_files(filename, method, data1, data2, data3, data4):
    with open(filename, 'a') as f:
        f.write("TD, Loss with true data:\n")
        f.write("%s: %.10f \n" % (method, np.mean(np.array(data1))))
        f.write("ASE, Loss with noise data: \n")
        f.write("%s: %.10f \n" % (method, np.mean(np.array(data2))))
        f.write("TD, minimal:\n")
        f.write("%s: %.10f \n" % (method,  data3))
        f.write("ASE, minimal: \n")
        f.write("%s: %.10f \n" % (method, data4))

def rbf(x, y, sigma):
    return np.exp(-(np.sum((x - y) ** 2)) / (2 * sigma ** 2))

# Calculate the big matrix of K
def calcK(X, T, p, sigma):
    K = np.empty((T * p, T))
    for dimension in range(p):
        for t in range(T):
            for temp_t in range(T):
                K[dimension * T + t][temp_t] = rbf(X[temp_t][dimension], X[t][dimension], sigma)

    return K

# Calculate the big matrix of K
def calcK_group(X, T, sigma):
    K = np.empty((T * 7, T))
    K1 = np.empty((T, T))
    K2 = np.empty((T, T))
    K3 = np.empty((T, T))
    K12 = np.empty((T, T))
    K13 = np.empty((T, T))
    K23 = np.empty((T, T))
    K123 = np.empty((T, T))
    for t in range(T):
        for temp_t in range(T):
            K1[t][temp_t] = rbf(X[temp_t][0], X[t][0], sigma)
            K2[t][temp_t] = rbf(X[temp_t][1], X[t][1], sigma)
            K3[t][temp_t] = rbf(X[temp_t][2], X[t][2], sigma)
            a = np.array([X[temp_t][0], X[temp_t][1]])
            b = np.array([X[t][0], X[t][1]])
            K12[t][temp_t] = rbf(a, b, sigma)
            a = np.array([X[temp_t][0], X[temp_t][2]])
            b = np.array([X[t][0], X[t][2]])
            K13[t][temp_t] = rbf(a, b, sigma)
            a = np.array([X[temp_t][1], X[temp_t][2]])
            b = np.array([X[t][1], X[t][2]])
            K23[t][temp_t] = rbf(a, b, sigma)
            K123[t][temp_t] = rbf(X[temp_t, :], X[t, :], sigma)
            

    K = np.concatenate((K1, K2, K3, K12, K13, K23, K123), axis=0)

    return # Calculate the big matrix of K
def calcK_group(X, T, sigma):
    K = np.empty((T * 7, T))
    K1 = np.empty((T, T))
    K2 = np.empty((T, T))
    K3 = np.empty((T, T))
    K12 = np.empty((T, T))
    K13 = np.empty((T, T))
    K23 = np.empty((T, T))
    K123 = np.empty((T, T))
    for t in range(T):
        for temp_t in range(T):
            K1[t][temp_t] = rbf(X[temp_t][0], X[t][0], sigma)
            K2[t][temp_t] = rbf(X[temp_t][1], X[t][1], sigma)
            K3[t][temp_t] = rbf(X[temp_t][2], X[t][2], sigma)
            a = np.array([X[temp_t][0], X[temp_t][1]])
            b = np.array([X[t][0], X[t][1]])
            K12[t][temp_t] = rbf(a, b, sigma)
            a = np.array([X[temp_t][0], X[temp_t][2]])
            b = np.array([X[t][0], X[t][2]])
            K13[t][temp_t] = rbf(a, b, sigma)
            a = np.array([X[temp_t][1], X[temp_t][2]])
            b = np.array([X[t][1], X[t][2]])
            K23[t][temp_t] = rbf(a, b, sigma)
            K123[t][temp_t] = rbf(X[temp_t, :], X[t, :], sigma)
            

    K = np.concatenate((K1, K2, K3, K12, K13, K23, K123), axis=0)

    return 
# Calculate the big matrix of K
def calcK_group(X, T, sigma):
    K = np.empty((T * 7, T))
    K1 = np.empty((T, T))
    K2 = np.empty((T, T))
    K3 = np.empty((T, T))
    K12 = np.empty((T, T))
    K13 = np.empty((T, T))
    K23 = np.empty((T, T))
    K123 = np.empty((T, T))
    for t in range(T):
        for temp_t in range(T):
            K1[t][temp_t] = rbf(X[temp_t][0], X[t][0], sigma)
            K2[t][temp_t] = rbf(X[temp_t][1], X[t][1], sigma)
            K3[t][temp_t] = rbf(X[temp_t][2], X[t][2], sigma)
            a = np.array([X[temp_t][0], X[temp_t][1]])
            b = np.array([X[t][0], X[t][1]])
            K12[t][temp_t] = rbf(a, b, sigma)
            a = np.array([X[temp_t][0], X[temp_t][2]])
            b = np.array([X[t][0], X[t][2]])
            K13[t][temp_t] = rbf(a, b, sigma)
            a = np.array([X[temp_t][1], X[temp_t][2]])
            b = np.array([X[t][1], X[t][2]])
            K23[t][temp_t] = rbf(a, b, sigma)
            K123[t][temp_t] = rbf(X[temp_t, :], X[t, :], sigma)
            

    K = np.concatenate((K1, K2, K3, K12, K13, K23, K123), axis=0)

    return 
# Calculate the big matrix of K
def calcK_group2(X, T, sigma):
    K = np.empty((T * 7, T))
    K1 = np.empty((T, T))
    K2 = np.empty((T, T))
    K12 = np.empty((T, T))
    for t in range(T):
        for temp_t in range(T):
            K1[t][temp_t] = rbf(X[temp_t][0], X[t][0], sigma)
            K2[t][temp_t] = rbf(X[temp_t][1], X[t][1], sigma)
            a = np.array([X[temp_t][0], X[temp_t][1]])
            b = np.array([X[t][0], X[t][1]])
            K12[t][temp_t] = rbf(a, b, sigma)
            

    K = np.concatenate((K1, K2, K12), axis=0)

    return K

def calcK_pred(X_static,T,y,p,sigma):
    K = np.empty((T*p, 1))
    for dim in range(p):
        for t in range(T):
            K[dim*T+t] = rbf(X_static[t][dim], y[dim], sigma)

# Create X from y and given time interval
def createX(y, start, interval, p, m, back_t):
    X = np.zeros((interval, p))

    for t in range(interval):
        for dim in range(m):
            curr_t = t + start
            for temp_t in range(back_t):
                X[t][dim * back_t + temp_t] = y[curr_t - (back_t - temp_t)][dim]

    return X

    # Calculate the Lipschitz constant of gradient of f
def calcL(K, T):
    max_val = -np.inf
    for t in range(T):
        norm = np.linalg.norm(K[:, t] @ K[:, t].T)
        if norm > max_val:
            max_val = norm

    return max_val

def get_gradient(k, k_tj, alpha, y, sigma, T, square_loss):
    diff = k.T @ alpha - y
    if square_loss:
        ans = 2 * diff * k_tj
        return ans.reshape((T, 1))
    else:
        if np.abs(diff) < sigma:
            #print("1!")
            ans = 2 * diff * k_tj
            return ans.reshape((T, 1))
        elif diff > sigma:
            #print("2!")
            return 2 * sigma * np.ones((T, 1))
        else:
            #print("3!")
            return -2 * sigma * np.ones((T, 1))


def get_gradient_sum(K, alpha, y, sigma, T,p, B,s_hat):
    sum = np.zeros((T*p, 1))
    for i in B:
        temp = get_gradient(K[:, i], K[:,i], alpha, y[i], sigma, T*p, 0)
        sum += s_hat[i] * temp
    return sum

def update_alpha(alpha, y, L, T, p, lamb, sigma, K, square_loss):
    v = np.empty((T * p, 1))
    for dim in range(p):
        sum = np.zeros((T, 1))
        for t in range(T):
            temp = get_gradient(K[:, t], K[dim * T:(dim + 1) * T][t], alpha, y[t], sigma, T, square_loss)
            sum += temp

        vj = alpha[dim * T:(dim + 1) * T] - (1 / (T * L)) * sum
        first = 1 - (lamb / np.linalg.norm(vj))

        v[dim * T:(dim + 1) * T] = max(0, first) * vj

    return v

def update_alpha2(alpha, y, L, T, p, lamb, sigma, K, square_loss, s):
    v = np.empty((T * p, 1))
    for dim in range(p):
        sum = np.zeros((T, 1))
        for t in range(T):
            temp = get_gradient(K[:, t], K[dim * T:(dim + 1) * T][t], alpha, y[t], sigma, T, square_loss)
            sum = s[t] * temp +sum

        vj = alpha[dim * T:(dim + 1) * T] - (1 / L) * sum
        first = 1 - (lamb / np.linalg.norm(vj))

        v[dim * T:(dim + 1) * T] = max(0, first) * vj

    return v

def prox(alpha, T, p, lamb):
    v = np.empty((T * p, 1))
    for dim in range(p):
        vj = alpha[dim * T:(dim + 1) * T]
        first = 1 - (lamb / np.linalg.norm(vj))
        v[dim * T:(dim + 1) * T] = max(0, first) * vj
    return v

def prox_box(s,p,T):
    for dim in range(T):
        if s[dim] < 0:
            s[dim] = 0
    return s

def huber(a, b, sigma):
    if np.abs(a-b) < sigma:
        return (a-b)**2
    else:
        return 2 * sigma * np.abs(a-b) - sigma**2

def update_A(T, p, X, alpha, y, rbf_sigma, huber_sigma):
    A = np.empty((T, 1))
    for t in range(T):
        k = np.empty((T * p))
        for j in range(p):
            for inner_t in range(T):
                k[j * T + inner_t] = rbf(X[inner_t][j], X[t][j], rbf_sigma)

        prediction = k @ alpha
        loss = huber(prediction, y[t], huber_sigma)
        A[t] =  loss

    return A


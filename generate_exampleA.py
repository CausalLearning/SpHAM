import numpy as np
def generate_A(filename1, filename2, noise = 'gau'):
    exp_T = 4000
    big_y_true_gau = []
    big_y_noise_gau = []
    big_y_true_t2 = []
    big_y_noise_t2 = []
    for times in range(100):
        y_true_gau = np.zeros((exp_T, 1, 1))
        y_true_gau[0] = np.random.rand()
        y_true_gau[1] = np.random.rand()
        y_true_t2 = np.zeros((exp_T, 1, 1))
        y_true_t2[0] = np.random.rand()
        y_true_t2[1] = np.random.rand()
        y_noise_gau = y_true_gau.copy()
        y_noise_t2 = y_true_t2.copy()
        e_gau = np.random.normal(0, 0.3, (exp_T, 1))
        e_t2 = np.random.standard_t(2, (exp_T,1))
        y_noise_gau[0] = y_true_gau[0] + e_gau[0]
        y_noise_gau[1] = y_true_gau[1] + e_gau[1]
        y_noise_t2[0] = y_true_t2[0] + e_t2[0]
        y_noise_t2[1] = y_true_t2[1] + e_t2[1]
        for t in range(2, exp_T):
            y_true_gau[t] = (3./2.)*np.sin(np.pi / 2. * y_noise_gau[t - 1]) - np.sin(np.pi / 2. * y_noise_gau[t - 2])
            y_noise_gau[t] = y_true_gau[t] + 2* e_gau[t]

            y_true_t2[t] = np.sin(np.pi / 2. * y_noise_t2[t - 1]) -np.sin(np.pi / 2. * y_noise_t2[t - 2])
            y_noise_t2[t] = y_true_t2[t] + 2* e_t2[t]
        big_y_true_gau.append(y_true_gau)
        big_y_noise_gau.append(y_noise_gau)
        big_y_true_t2.append(y_true_t2)
        big_y_noise_t2.append(y_noise_t2)
    if noise == 'gau':
        with open(filename1, 'wb') as f:
            np.save(f, np.array(big_y_true_gau))
        with open(filename2, 'wb') as f:
            np.save(f, np.array(big_y_noise_gau))
    else:
        with open(filename1, 'wb') as f:
            np.save(f, np.array(big_y_true_t2))
        with open(filename2, 'wb') as f:
            np.save(f, np.array(big_y_noise_t2))

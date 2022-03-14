import numpy as np
def generate_C(filename1, filename2, noise = 'gau'):
    exp_T = 4000
    big_y_true_gau = []
    big_y_noise_gau = []
    big_y_true_t2 = []
    big_y_noise_t2 = []
    for times in range(100):
        e_t2 = np.random.standard_t(2, (exp_T,1))
        e_gau = np.random.normal(0, 0.3, (exp_T, 1))
        y_true_gau = np.zeros((exp_T, 1, 1))
        y_true_t2 = np.zeros((exp_T, 1, 1))
        y_true_gau[0][0] = np.random.rand()
        y_true_t2[0][0] = np.random.rand()
        y_noise_gau = y_true_gau.copy()
        y_noise_t2 = y_true_t2.copy()
        for t in range(1, exp_T):
            y_true_gau[t][0] = (t / 400.) * np.sin(y_noise_gau[t - 1][0])
            y_true_t2[t][0] = (t / 400.) * np.sin(y_noise_t2[t - 1][0])
            y_noise_gau[t][0] = y_true_gau[t][0] + 0.5* e_gau[t][0]
            y_noise_t2[t][0] = y_true_t2[t][0] + 0.5 * e_t2[t][0]

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

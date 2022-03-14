import sys
import numpy as np
from model import SpHAM as spham
from utilities import write_files
from generate_exampleA import generate_A



## data generating
filename1 = '/Users/jacky/Desktop/Paper/2021_ICLR_Wang_SphAM/Source_code_git/result/t_A/exampleA_t_true.npy'
filename2 = '/Users/jacky/Desktop/Paper/2021_ICLR_Wang_SphAM/Source_code_git/result/t_A/exampleA_t_noise.npy'

generate_A(filename1, filename2, noise = 't')


with open(filename1, 'rb') as f:
    big_y_true_t = np.load(f)
with open(filename2, 'rb') as f:
    big_y_noise_t = np.load(f)


## parameter
exp_T = 4000
start = 1500
interval = 400
back = 2
end = start + interval

max_iteration = 2000
num_pred = 100

loss_huber_true = []
loss_huber_noise = []


for experiment in range(50):
    print("Experiment %d" % experiment)
    y_true = big_y_true_t[experiment]
    y_noise = big_y_noise_t[experiment]
    sys.stdout.flush()

    model_spham = spham(y_noise, y_true, huber_sigma=np.power(interval, 1./48.), start=start, interval=interval, backward=back, rbf_sigma=0.5, lamb=1./interval)
    alpha = model_spham.fit(max_iter=max_iteration)
    prediction = model_spham.predict(alpha, y_noise, y_true, start=end, repeat=num_pred, backward=back)
    loss_huber_true.append(np.linalg.norm(y_true[end:end+num_pred, :, 0] - prediction, 2) / num_pred)
    loss_huber_noise.append(np.linalg.norm(y_noise[end:end+num_pred, :, 0] - prediction, 2) / num_pred)

    loss_huber_true_print = np.mean(loss_huber_true)
    loss_huber_noise_print = np.mean(loss_huber_noise)

    loss_huber_true_print_std=np.std(loss_huber_true)
    loss_huber_noise_print_std=np.std(loss_huber_noise)

    print("SpHAM: ASE:%.3f, ASE std:%.3f, TD:%.3f, TD std:%.3f\n" % (loss_huber_noise_print,  loss_huber_noise_print_std, loss_huber_true_print, loss_huber_true_print_std))



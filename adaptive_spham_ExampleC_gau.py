import sys
import numpy as np
from utilities import write_files
from model import nonStationary
from model import SpHAM as spham
from generate_exampleC import generate_C

filename1 = '/Users/jacky/Desktop/Paper/2021_ICLR_Wang_SphAM/Source_code_git/result/gau_B/exampleB_gau_true.npy'
filename2 = '/Users/jacky/Desktop/Paper/2021_ICLR_Wang_SphAM/Source_code_git/result/gau_B/exampleB_gau_noise.npy'

generate_C(filename1, filename2,  noise = 'gau')

with open(filename1, 'rb') as f:
    big_y_true_gau = np.load(f)
with open(filename2, 'rb') as f:
    big_y_noise_gau = np.load(f)


exp_T = 4000
start = 1500
interval = 400
back = 1
end = start + interval
max_iteration = 2000
num_pred = 100

loss_huber_true = []
loss_huber_noise = []
loss_adaptive_true = []
loss_adaptive_noise = []

for experiment in range(50):

    print("Experiment %d" % experiment)
    y_true = big_y_true_gau[experiment]
    y_noise = big_y_noise_gau[experiment]
    sys.stdout.flush()


    #spham
    model_spham = spham(y_noise, y_true, huber_sigma=np.power(interval, 1/48), start=start, interval=interval, backward=back, rbf_sigma=0.5, lamb=1/interval)
    alpha = model_spham.fit(max_iter=max_iteration)
    prediction = model_spham.predict(alpha, y_noise, y_true, start=end, repeat=num_pred, backward=back)
    loss_huber_true.append(np.linalg.norm(y_true[end:end+num_pred, :, 0] - prediction, 2) / num_pred)
    loss_huber_noise.append(np.linalg.norm(y_noise[end:end+num_pred, :, 0] - prediction, 2) / num_pred)

    #adaptive spham l=150
    model_aspham = nonStationary(y_noise, y_true, l=300, huber_sigma=np.power(interval, 1/48),
                                 start=start, interval=interval, lamb1=10000, lamb2=100, backward=back,
                                 rbf_sigma=0.5, lamb=1/interval, s_iter=20, out_iter=20)
    alpha = model_aspham.fit(max_iter=max_iteration)
    prediction = model_aspham.predict(alpha, y_noise, y_true, start=end, repeat=num_pred, backward=back)
    loss_adaptive_true.append(np.linalg.norm(y_true[end:end+num_pred, :, 0] - prediction, 2) / num_pred)
    loss_adaptive_noise.append(np.linalg.norm(y_noise[end:end+num_pred, :, 0] - prediction, 2) / num_pred)

    loss_huber_true_print=np.mean(loss_huber_true)
    loss_huber_noise_print=np.mean(loss_huber_noise)
    loss_adaptive_true_print=np.mean(loss_adaptive_true)
    loss_adaptive_noise_print=np.mean(loss_adaptive_noise)

    loss_huber_true_print_std = np.std(loss_huber_true)
    loss_huber_noise_print_std = np.std(loss_huber_noise)
    loss_adaptive_true_print_std=np.std(loss_adaptive_true)
    loss_adaptive_noise_print_std=np.std(loss_adaptive_noise)

    print("SpHAM: ASE:%.3f, ASE std:%.3f, TD:%.3f, TD std:%.3f\n" % (loss_huber_noise_print, loss_huber_noise_print_std, loss_huber_true_print, loss_huber_true_print_std))
    print("ASpHAM l=300:  ASE:%.3f, ASE std:%.3f, TD:%.3f, TD std:%.3f\n" % (loss_adaptive_noise_print, loss_adaptive_noise_print_std, loss_adaptive_true_print, loss_adaptive_true_print_std))



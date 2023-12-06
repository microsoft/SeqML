import os
import argparse
import random
import logging
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
from .utils import *

matplotlib.use('agg')


def get_logger(name):
    logger = logging.getLogger(name)
    filename = f'{name}.log'
    fh = logging.FileHandler(filename, mode='a+', encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=False)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default='./')
parser.add_argument('--model_name', type=str, default='Neural_ODE',
                    choices=['Neural_ODE', 'Contiformer'])
parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--seed', type=int, default=27)
parser.add_argument('--noise_std', type=float, default=.05)
parser.add_argument('--noise_a', type=float, default=0)
parser.add_argument('--cc', type=eval, default=True)

## parameters for Contiformer
parser.add_argument('--atol', type=float, default=0.1)
parser.add_argument('--rtol', type=float, default=0.1)
parser.add_argument('--method', type=str, default='rk4')
parser.add_argument('--dropout', type=float, default=0)

args = parser.parse_args()

if not os.path.exists(args.train_dir):
    os.makedirs(args.train_dir)

log = get_logger(os.path.join(args.train_dir, 'log'))

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      noise_a=.002,
                      a=0.,
                      b=1.):
    """Parametric formula for 2d spiral is `r = a + b * theta`.
    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check
    Returns:
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)  # [ntotal]
    aa = npr.randn(nspiral) * noise_a + a  # [nspiral]
    bb = npr.randn(nspiral) * noise_a + b  # [nspiral]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts  # [ntotal]
    rs_cw = aa.reshape(-1, 1) + bb.reshape(-1, 1) * 50. / zs_cw  # [nspiral, ntotal]
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=-1)  # [nspiral, ntotal, 2]
    orig_traj_cw = np.flip(orig_traj_cw, axis=1)

    zs_cc = orig_ts
    rw_cc = aa.reshape(-1, 1) + bb.reshape(-1, 1) * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=-1)

    # sample starting timestamps
    orig_trajs = []
    for _ in range(nspiral):
        if args.cc == 2:
            cc = bool(npr.rand() > .5)  # uniformly select rotation
        else:
            cc = args.cc
        orig_traj = orig_traj_cc[_] if cc else orig_traj_cw[_]
        orig_trajs.append(orig_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)

    samp_trajs = npr.randn(*orig_trajs.shape) * noise_std + orig_trajs

    return orig_trajs, samp_trajs, orig_ts


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(1, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


class NeuralODE(nn.Module):
    def __init__(self, obs_dim, device, batch_size=200):
        super(NeuralODE, self).__init__()
        self.latent_dim = 8
        self.func = LatentODEfunc(self.latent_dim, 16).to(device)
        self.rec = RecognitionRNN(self.latent_dim, obs_dim + 1, 16, 1).to(device)
        self.dec = Decoder(self.latent_dim, obs_dim, 16).to(device)
        self.batch_size = batch_size

    def forward(self, samples, orig_ts, **kwargs):
        if kwargs.get('is_train', False):
            bs, ls = samples.shape[0], len(orig_ts)
            sample_idx = npr.choice(bs, self.batch_size, replace=False)
            samples = samples[sample_idx, ...]
            h = self.rec.initHidden().to(device).repeat(samples.shape[0], 1)

            for t in reversed(range(samples.size(1))):
                obs = samples[:, t, :]
                out, h = self.rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            pred_z = odeint(self.func, z0, torch.tensor(orig_ts)).permute(1, 0, 2)
            pred_x = self.dec(pred_z)
            return pred_x, qz0_mean, qz0_logvar, sample_idx
        else:
            h = self.rec.initHidden().to(device).repeat(samples.shape[0], 1)

            for t in reversed(range(samples.size(1))):
                obs = samples[:, t, :]
                out, h = self.rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            pred_z = odeint(self.func, z0, torch.tensor(orig_ts)).permute(1, 0, 2)
            pred_x = self.dec(pred_z)
            return pred_x, qz0_mean, qz0_logvar, None

    def calculate_loss(self, out, target):
        pred_x, qz0_mean, qz0_logvar, idx = out
        target_x, pz0_mean, pz0_logvar = target
        if idx is not None:
            noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(
                target_x[idx, ...], pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            return loss
        else:
            noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(
                target_x, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            return loss


class Contiformer(nn.Module):
    def __init__(self, obs_dim, device, batch_size=64):
        super(Contiformer, self).__init__()
        args_ode = {
            'use_ode': True, 'actfn': 'tanh', 'layer_type': 'concat', 'zero_init': True,
            'atol': args.atol, 'rtol': args.rtol, 'method': args.method, 'regularize': False,
            'approximate_method': 'bilinear', 'nlinspace': 1, 'linear_type': 'before',
            'interpolate': 'linear', 'itol': 1e-2
        }
        args_ode = AttrDict(args_ode)

        self.encoder = EncoderLayer(16, 64, 4, 4, 4, args=args_ode, dropout=args.dropout).to(device)
        self.lin_in = nn.Linear(obs_dim, 16).to(device)
        self.lin_out = nn.Linear(16, obs_dim).to(device)

        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / 16) for i in range(16)])
        self.batch_size = batch_size

    def temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def pad_input(self, input, t0, tmax=6 * math.pi):
        input_last = input[:, -1:, :]
        input = torch.cat((input, input_last), dim=1)
        t0 = torch.cat((t0, torch.tensor([tmax]).to(t0.device)), dim=0)
        return input, t0

    def forward(self, samples, orig_ts, **kwargs):
        if kwargs.get('is_train', False):
            bs, ls = samples.shape[0], len(orig_ts)
            sample_idx = npr.choice(bs, self.batch_size, replace=False)
            samples = samples[sample_idx, ...]

            t0 = samples[..., -1]
            input = self.lin_in(samples[..., :-1])
            input = (input + self.temporal_enc(t0)).float()

            _input, _t0 = self.pad_input(input, t0[0])

            X = torchcde.LinearInterpolation(_input, t=_t0)
            input = X.evaluate(orig_ts).float()
            orig_ts = torch.tensor(orig_ts).to(input.device)

            non_pad_mask = torch.ones(self.batch_size, ls, 1).to(input.device)
            out, _ = self.encoder(input, orig_ts.unsqueeze(0).repeat(self.batch_size, 1).float(),
                                  non_pad_mask=non_pad_mask)
            return self.lin_out(out), sample_idx
        else:
            bs, ls = samples.shape[0], len(orig_ts)
            t0 = samples[..., -1]
            input = self.lin_in(samples[..., :-1])
            input = (input + self.temporal_enc(t0)).float()

            _input, _t0 = self.pad_input(input, t0[0])

            X = torchcde.LinearInterpolation(_input, t=_t0)
            input = X.evaluate(orig_ts).float()
            orig_ts = torch.tensor(orig_ts).to(input.device)

            non_pad_mask = torch.ones(bs, ls, 1).to(input.device)
            out, _ = self.encoder(input, orig_ts.unsqueeze(0).repeat(bs, 1).float(), non_pad_mask=non_pad_mask)
            return self.lin_out(out), None

    def calculate_loss(self, out, target):
        pred_x, idx = out
        target_x, _, _ = target
        if idx is not None:
            return ((pred_x - target_x[idx, ...]) ** 2).sum()
        else:
            return ((pred_x - target_x) ** 2).sum()


if __name__ == '__main__':
    np.random.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    obs_dim = 2
    nspiral = 300
    start = 0.
    stop = 6 * np.pi
    noise_std = args.noise_std
    noise_a = args.noise_a
    a = 0.
    b = .3
    ntotal = 150
    nsample = 50
    ntrain = 200
    ntest = 100
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    best_val = np.inf
    best_model = None

    # generate toy spiral data
    orig_trajs, samp_traj, orig_ts = generate_spiral2d(
        nspiral=nspiral,
        ntotal=ntotal,
        start=start,
        stop=stop,
        noise_std=noise_std,
        noise_a=noise_a,
        a=a, b=b
    )
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_traj = torch.from_numpy(samp_traj).float().to(device)

    # normalize traj
    trajs_min_x, trajs_min_y = torch.min(orig_trajs[:, :, 0]), torch.min(orig_trajs[:, :, 1])
    trajs_max_x, trajs_max_y = torch.max(orig_trajs[:, :, 0]), torch.max(orig_trajs[:, :, 1])
    orig_trajs[:, :, 0] = (orig_trajs[:, :, 0] - trajs_min_x) / (trajs_max_x - trajs_min_x)
    orig_trajs[:, :, 1] = (orig_trajs[:, :, 1] - trajs_min_y) / (trajs_max_y - trajs_min_y)
    samp_traj[:, :, 0] = (samp_traj[:, :, 0] - trajs_min_x) / (trajs_max_x - trajs_min_x)
    samp_traj[:, :, 1] = (samp_traj[:, :, 1] - trajs_min_y) / (trajs_max_y - trajs_min_y)

    test_idx = npr.choice(int(ntotal * 0.5), nsample, replace=False)
    test_idx = sorted(test_idx.tolist())

    train_trajs = samp_traj[:ntrain]
    test_trajs = samp_traj[ntrain:]
    train_target = orig_trajs[:ntrain]
    test_target = orig_trajs[ntrain:]

    # model
    if args.model_name == 'Neural_ODE':
        model = NeuralODE(obs_dim, device)
    elif args.model_name == 'Contiformer':
        model = Contiformer(obs_dim, device)
    else:
        raise NotImplementedError

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_meter = RunningAverageMeter()

    st = 0

    if args.train_dir is not None:
        ckpt_path = os.path.join(args.train_dir, f'ckpt_{args.model_name}.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model = checkpoint['model']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            orig_trajs = checkpoint['orig_trajs']
            orig_ts = checkpoint['orig_ts']
            test_idx = checkpoint['test_idx']
            train_trajs = checkpoint['train_trajs']
            test_trajs = checkpoint['test_trajs']
            test_target = checkpoint['test_target']
            st = checkpoint['itr']
            log.info('Loaded ckpt from {}'.format(ckpt_path))

    for itr in range(st + 1, args.niters + 1):
        # train one iteration

        optimizer.zero_grad()
        # backward in time to infer q(z_0)

        idx = npr.choice(int(ntotal * 0.5), nsample, replace=False)
        idx = sorted(idx.tolist())

        samp_trajs = train_trajs[:, idx, :]
        samp_ts = torch.tensor(orig_ts[idx]).to(samp_trajs.device)
        samp_ts = samp_ts.reshape(1, -1, 1).repeat(ntrain, 1, 1)
        samp_trajs = torch.cat((samp_trajs, samp_ts), dim=-1).float()

        out = model(samp_trajs, orig_ts, idx=idx, is_train=True)
        try:
            pz0_mean = pz0_logvar = torch.zeros(out[1].size()).to(device)
        except:
            pz0_mean = pz0_logvar = None
        loss = model.calculate_loss(out, (train_target, pz0_mean, pz0_logvar))
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

        log.info('Iter: {}, running loss: {:.4f}'.format(itr, loss_meter.avg))

        ckpt_path = os.path.join(args.train_dir, f'ckpt_{args.model_name}.pth')
        torch.save({
            'model': model,
            'optimizer_state_dict': optimizer.state_dict(),
            'orig_trajs': orig_trajs,
            'orig_ts': orig_ts,
            'test_idx': test_idx,
            'train_trajs': train_trajs,
            'test_trajs': test_trajs,
            'test_target': test_target,
            'itr': itr,
        }, ckpt_path)
        log.info('Stored ckpt at {}'.format(ckpt_path))

        # test one iteration
        with torch.no_grad():
            samp_trajs = test_trajs[:, test_idx, :]
            samp_ts = torch.tensor(orig_ts[test_idx]).to(samp_trajs.device)
            samp_ts = samp_ts.reshape(1, -1, 1).repeat(ntest, 1, 1)
            samp_trajs = torch.cat((samp_trajs, samp_ts), dim=-1).float()

            pred_x = model(samp_trajs, orig_ts, idx=test_idx)[0]
            mae = torch.abs(pred_x - test_target).sum(dim=-1).mean()
            rmse = torch.sqrt(((pred_x - test_target) ** 2).sum(dim=-1).mean())
            log.info('Iter: {}, MAE: {:.4f}, RMSE: {:.4f}'.format(itr, mae.item(), rmse.item()))

            if mae.item() < best_val:
                best_val = mae.item()

                with torch.no_grad():
                    # sample from trajectorys' approx. posterior
                    model_vis = torch.load(ckpt_path)['model']
                    samp_trajs = test_trajs[:, test_idx, :]
                    samp_ts = torch.tensor(orig_ts[test_idx]).to(samp_trajs.device)
                    samp_ts = samp_ts.reshape(1, -1, 1).repeat(ntest, 1, 1)
                    samp_trajs = torch.cat((samp_trajs, samp_ts), dim=-1).float()

                    pred_x = model_vis(samp_trajs, orig_ts, idx=test_idx)[0]

                    xs_pos = pred_x[0][:pred_x.shape[1] // 2, :]
                    xs_neg = pred_x[0][pred_x.shape[1] // 2 - 1:, :]

                save_path = os.path.join(args.train_dir, f'pred.pkl')
                torch.save({
                    'pred': pred_x,
                    'target': test_target,
                    'samp': samp_trajs
                }, save_path)

                ckpt_path = os.path.join(args.train_dir, f'ckpt_{args.model_name}_best.pth')
                torch.save({
                    'model': model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'orig_trajs': orig_trajs,
                    'orig_ts': orig_ts,
                    'test_idx': test_idx,
                    'train_trajs': train_trajs,
                    'test_trajs': test_trajs,
                    'test_target': test_target,
                    'itr': itr,
                }, ckpt_path)
                log.info('Stored ckpt at {}'.format(ckpt_path))

        if args.visualize and itr % args.log_step == 0:
            with torch.no_grad():
                # sample from trajectorys' approx. posterior
                ckpt_path = os.path.join(args.train_dir, f'ckpt_{args.model_name}_best.pth')
                model_vis = torch.load(ckpt_path)['model']
                samp_trajs = test_trajs[:, test_idx, :]
                samp_ts = torch.tensor(orig_ts[test_idx]).to(samp_trajs.device)
                samp_ts = samp_ts.reshape(1, -1, 1).repeat(ntest, 1, 1)
                samp_trajs = torch.cat((samp_trajs, samp_ts), dim=-1).float()

                pred_x = model_vis(samp_trajs, orig_ts, idx=test_idx)[0]

                xs_pos = pred_x[0][:pred_x.shape[1] // 2, :]
                xs_neg = pred_x[0][pred_x.shape[1] // 2 - 1:, :]

                xs_pos = xs_pos.cpu().numpy()
                xs_neg = xs_neg.cpu().numpy()

                orig_traj = test_target[0].cpu().numpy()
                samp_traj = samp_trajs[0].cpu().numpy()


                def tohex(rgb):
                    hex_r = hex(rgb[0])[2:].upper()  # 10进制转16进制，并去掉16进制前面的“0x”，再把得出的结果转为大写
                    hex_g = hex(rgb[1])[2:].upper()
                    hex_b = hex(rgb[2])[2:].upper()
                    hex_r0 = hex_r.zfill(2)  # 位数不足2位时补“0”
                    hex_g0 = hex_g.zfill(2)
                    hex_b0 = hex_b.zfill(2)
                    return '#' + hex_r0 + hex_g0 + hex_b0  # 打印最终结果（格式如“#ff0402”）


                color = {
                    'g': tohex((95, 206, 64)),
                    'r': tohex((234, 60, 51)),
                    'b': tohex((48, 111, 215))
                }
                plt.figure()

                plt.plot(orig_traj[:, 0], orig_traj[:, 1],
                         color['g'], label='True Trajectory', linewidth=1.5)
                plt.plot(xs_pos[:, 0], xs_pos[:, 1], color['b'],
                         label='Interpolation', linewidth=1.5)
                plt.plot(xs_neg[:, 0], xs_neg[:, 1], color['r'],
                         label='Extrapolation', linewidth=1.5)
                plt.scatter(samp_traj[:, 0], samp_traj[:, 1], color=color['g'],
                            label='Sampled Data', s=10)
                plt.scatter(xs_pos[:, 0], xs_pos[:, 1], color=color['b'],
                            label='Prediction', s=10)
                plt.axis('off')
                save_path = os.path.join(args.train_dir, f'vis_{itr}.png')
                plt.savefig(save_path, dpi=500)
                log.info('Saved visualization figure at {}'.format(save_path))

                save_path = os.path.join(args.train_dir, f'pred_{itr}.pkl')
                torch.save({
                    'pred': pred_x,
                    'target': test_target,
                    'samp': samp_trajs
                }, save_path)

                log.info('Saved predict file at {}'.format(save_path))

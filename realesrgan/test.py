# flake8: noqa
import logging
import torch
import os.path as osp
import numpy as np

import realesrgan.archs
import realesrgan.data
import realesrgan.models
import matplotlib.pyplot as plt
import sklearn.preprocessing

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    model.is_train = False

    current_count = 0
    num_code = 128
    count = 5

    for test_loader in test_loaders:
        for idx, val_data in enumerate(test_loader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            model.feed_data(val_data)

            gens = []
            for z in range(num_code):
                model.test()
                visuals = model.get_current_visuals()
                gens.append(visuals['result'])

            gens = torch.stack(gens, dim=0)

            for j in range(gens.shape[1]):
                if current_count >= count:
                    exit()

                single_samps = np.zeros((num_code, 3, gens[:, j, :, :, :].shape[-2], gens[:, j, :, :, :].shape[-1]))
                np_avg = torch.mean(gens[:, j, :, :, :], dim=0).cpu().numpy()
                for z in range(num_code):
                    single_samps[z, :, :, :] = gens[z, j, :, :, :].cpu().numpy()

                single_samps = single_samps - np_avg[None, :, :, :]

                cov_mat = np.zeros((num_code, 3*np_avg.shape[-1] * np_avg.shape[-2]))

                for z in range(num_code):
                    cov_mat[z, :] = single_samps[z].flatten()

                u, s, vh = np.linalg.svd(cov_mat, full_matrices=False)

                plt.figure()
                plt.scatter(range(len(s)), sklearn.preprocessing.normalize(s.reshape((1, -1))))
                plt.savefig(f'sv_test/test_sv_{current_count}.png')
                plt.close()

                for l in range(5):
                    v_re = vh[l].reshape((3, gens[:, j, :, :, :].shape[-2], gens[:, j, :, :, :].shape[-1]))
                    v_re = (v_re - np.min(v_re)) / (np.max(v_re) - np.min(v_re))
                    plt.figure()
                    plt.imshow(v_re.transpose(1, 2, 0))
                    plt.savefig(f'sv_test/test_sv_v_{current_count}_{l}.png')
                    plt.close()

                current_count += 1


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)

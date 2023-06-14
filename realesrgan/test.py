# flake8: noqa
import logging
import torch
import os.path as osp

import realesrgan.archs
import realesrgan.data
import realesrgan.models

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

    for test_loader in test_loaders:
        for idx, val_data in enumerate(test_loader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            model.feed_data(val_data)

            gens = []
            for z in range(128):
                model.test()
                visuals = model.get_current_visuals()
                gens.append(visuals['result'])

            P_8_avg = torch.mean(torch.stack(gens, dim=0), dim=0)
            P_1_avg = visuals['result']

            P_8_avg = tensor2img([P_8_avg])
            P_1_avg = tensor2img([P_1_avg])

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                P_vals = [1, 8]
                P_avgs = [P_1_avg, P_8_avg]
                for i in range(2):
                    if P_vals[i] != 8:
                        continue

                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}_P={P_vals[i]}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}_P={P_vals[i]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}_P={P_vals[i]}.png')
                    imwrite(P_avgs[i], save_img_path)

                gif_ims = []
                gens = torch.stack(gens, dim=0)
                for z in range(self.opt['num_z_val']):
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_samp_{z}.png')
                    imwrite(tensor2img([gens[z]]), save_img_path)
                    gif_ims.append(iio.imread(save_img_path))

                save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                         f'{img_name}_{current_iter}_samp_gif.gif')
                iio.mimsave(save_img_path, gif_ims, duration=0.25)

                for z in range(self.opt['num_z_val']):
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_samp_{z}.png')
                    os.remove(save_img_path)

            for name, opt_ in self.opt['val']['metrics'].items():
                metric_data['img'] = P_1_avg if name == 'psnr_1' else P_8_avg

                self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        for metric in self.metric_results.keys():
            self.metric_results[metric] /= (idx + 1)
            self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

        psnr_8 = self.metric_results['psnr_8']
        psnr_1 = self.metric_results['psnr_1']

        mu_0 = 2e-2
        self.betastd += mu_0 * ((psnr_1 + 2.5) - psnr_8)
        self.cri_pix.update_loss_weight(self.betastd)

        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)

import time
import torch
import numpy as np
from options.train_options import TrainOptions
from options.test_options import TestOptions
import sys, traceback
import h5py
from data.data_loader import CreateDataLoader
from models.models import create_model
# from data.data_loader import CreateDataLoader_TEST
from data.data_loader import CreateDataLoaderIIWTest
from util import util_extra


def test_iiw(model, list_name, full_root='./', display_process=True, visualize_dir=None):
    total_loss =0.0
    total_loss_eq =0.0
    total_loss_ineq =0.0
    total_count = 0.0
    # print("============================= Validation ============================")
    model.switch_to_eval()

    # for 3 different orientation
    for j in range(0,3):
        # print("============================= Testing EVAL MODE ============================", j)
        test_list_dir = full_root + '/CGIntrinsics/IIW/' + list_name
        print(test_list_dir)
        data_loader_IIW_TEST = CreateDataLoaderIIWTest(full_root, test_list_dir, j)
        dataset_iiw_test = data_loader_IIW_TEST.load_data()

        for i, data in enumerate(dataset_iiw_test):
            stacked_img = data['img_1']
            targets = data['target_1']
            total_whdr, total_whdr_eq, total_whdr_ineq, count = model.evlaute_iiw(stacked_img, targets)
            total_loss += total_whdr
            total_loss_eq += total_whdr_eq
            total_loss_ineq += total_whdr_ineq

            total_count += count
            if display_process:
                print("Testing WHDR error ",j, i , total_loss/total_count)

            if visualize_dir is not None:
                pred_R, pred_S = model.predict_images(stacked_img)
                util_extra.visualize_results(visualize_dir, pred_R[0].cpu(), pred_S[0].cpu(),
                                             stacked_img[0].cpu(), targets['chromaticity'][0].cpu())

    return total_loss/(total_count), total_loss_eq/total_count, total_loss_ineq/total_count


if __name__ == '__main__':
    # opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    root = ""
    full_root = root + './'

    model = create_model(opt)

    print("WE ARE IN TESTING PHASE!!!!")
    WHDR, WHDR_EQ, WHDR_INEQ = test_iiw(model, 'test_list/', full_root, True, opt.results_dir+'/test_iiw')
    print('WHDR %f' % WHDR)

    print("We are done")

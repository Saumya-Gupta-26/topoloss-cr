#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Ali Arjomandbigdeli (get basecode from https://github.com/HuXiaoling/TopoLoss/blob/master/topoloss_pytorch.py)
# =============================================================================

# =============
# Modified by Saumya to test intermediate outputs
# =============

import time
import numpy as np
#import gudhi as gd
from pylab import *
import torch
import torch.nn.functional as F

import cripser as cr
import SimpleITK as sitk
import os

savedir = "/scr/saumgupta/kidney-vessel/topo-viz/uncropped"

def savefile(arr, outdir, outname):
    arr = np.squeeze(arr)
    new_sitkimage = sitk.GetImageFromArray(arr)
    sitk.WriteImage(new_sitkimage, os.path.join(outdir, outname))


def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0.03, pers_thresh_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighouboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thresh_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process

    """
    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if (gt_dgm.shape[0] == 0):
        gt_pers = None;
        gt_n_holes = 0;
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size  # number of holes in gt

    if (gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = list();
        idx_holes_to_remove = list(set(range(lh_pers.size)))
        idx_holes_perfect = list();
    else:
        # check to ensure that all gt dots have persistence 1
        tmp = gt_pers > pers_thresh_perfect

        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1; it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values
        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        if np.sum(tmp) >= 1:
            lh_n_holes_perfect = tmp.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect];
        else:
            idx_holes_perfect = list();

        # find top gt_n_holes indices
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes];

        # the difference is holes to be fixed to perfect
        idx_holes_to_fix = list(
            set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        # remaining holes are all to be removed
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:];

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    pers_thd = pers_thresh
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)
    
    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)

    if (do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove

def getCriticalPoints(likelihood):
    """
    Compute the critical points of the image (Value range from 0 -> 1)

    Args:
        likelihood: Likelihood image from the output of the neural networks

    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.

    """
    lh = 1 - likelihood
    lh_vector = np.asarray(lh).flatten()

    lh_cubic = gd.CubicalComplex(
        dimensions=[lh.shape[0], lh.shape[1]],
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()

    # If the paris is 0, return False to skip
    if (len(pairs_lh[0])==0): return 0, 0, 0, False

    # return persistence diagram, birth/death critical points
    pd_lh = numpy.array([[lh_vector[pairs_lh[0][0][i][0]], lh_vector[pairs_lh[0][0][i][1]]] for i in range(len(pairs_lh[0][0]))])
    bcp_lh = numpy.array([[pairs_lh[0][0][i][0]//lh.shape[1], pairs_lh[0][0][i][0]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))]) # [[cp_y, cp_y], ...]
    dcp_lh = numpy.array([[pairs_lh[0][0][i][1]//lh.shape[1], pairs_lh[0][0][i][1]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])

    return pd_lh, bcp_lh, dcp_lh, True

# 2D
def getTopoLoss(likelihood_tensor, gt_tensor, topo_size=100):
    """
    Calculate the topology loss of the predicted image and ground truth image 
    Warning: To make sure the topology loss is able to back-propagation, likelihood 
    tensor requires to clone before detach from GPUs. In the end, you can hook the
    likelihood tensor to GPUs device.

    Args:
        likelihood_tensor:   The likelihood pytorch tensor.
        gt_tensor        :   The groundtruth of pytorch tensor.
        topo_size        :   The size of the patch is used. Default: 100

    Returns:
        loss_topo        :   The topology loss value (tensor)

    """

    likelihood = torch.sigmoid(likelihood_tensor).clone()
    gt = gt_tensor.clone()

    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()

    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    for y in range(0, likelihood.shape[0], topo_size):
        for x in range(0, likelihood.shape[1], topo_size):

            lh_patch = likelihood[y:min(y + topo_size, likelihood.shape[0]),
                         x:min(x + topo_size, likelihood.shape[1])]
            gt_patch = gt[y:min(y + topo_size, gt.shape[0]),
                         x:min(x + topo_size, gt.shape[1])]

            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue

            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints(lh_patch)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints(gt_patch)

            # If the pairs not exist, continue for the next loop
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue

            if pd_lh.shape[0] == 0: continue

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thresh=0.03)

            if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                for hole_indx in idx_holes_to_fix:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                            bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 0
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                        topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                for hole_indx in idx_holes_to_remove:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push birth to death  # push to diagonal
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                likelihood[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push death to birth # push to diagonal
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                                likelihood[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0

    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((likelihood_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()
    return loss_topo


# 3D
def getTopoLoss3d(likelihood_tensor, gt_tensor, topo_size=100, loss_mode="bce"):
    """
    Calculate the topology loss of the predicted image and ground truth image 
    Warning: To make sure the topology loss is able to back-propagation, likelihood 
    tensor requires to clone before detach from GPUs. In the end, you can hook the
    likelihood tensor to GPUs device.

    Args:
        likelihood_tensor:   The likelihood pytorch tensor.
        gt_tensor        :   The groundtruth of pytorch tensor.
        topo_size        :   The size of the patch is used. Default: 100

    Returns:
        loss_topo        :   The topology loss value (tensor)

    """
    if likelihood_tensor.ndim != 3:
        print("incorrct dimension")
    
    # likelihood = torch.sigmoid(likelihood_tensor).clone()
    likelihood = likelihood_tensor.clone()
    # likelihood = torch.softmax(torch.unsqueeze(likelihood_tensor, 0), dim=0).clone()
    gt = gt_tensor.clone()

    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()

    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    #print("Loop y: {}; Loop x: {}".format(likelihood.shape[1]/topo_size, likelihood.shape[2]/topo_size))
    for y in range(0, likelihood.shape[1], topo_size):
        for x in range(0, likelihood.shape[2], topo_size):
            #print("Loop itr (x,y) = {},{}".format(x,y))
            lh_patch = likelihood[:, y:min(y + topo_size, likelihood.shape[1]),
                         x:min(x + topo_size, likelihood.shape[2])]
            gt_patch = gt[:, y:min(y + topo_size, gt.shape[1]),
                         x:min(x + topo_size, gt.shape[2])]

            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue

            # Get the critical points of predictions and ground truth

            ##### IMPORTANT - SAUMYA
            pd_arr_lh = cr.computePH(1 - lh_patch) # flipped/negated
            # pd_arr_lh = cr.computePH(lh_patch) 

            ##### IMPORTANT - SAUMYA
            pd_arr_lh = pd_arr_lh[pd_arr_lh[:, 0] == 0] # 0-dim topology (component) selected
            #pd_arr_lh = pd_arr_lh[pd_arr_lh[:, 0] == 2] # 2-dim topology (component) selected
            
            pd_arr_lh = pd_arr_lh[pd_arr_lh[:, 2] <= 1]
            pd_lh = pd_arr_lh[:, 1:3]
            # pd_lh = np.clip(pd_arr_lh[:, 1:3], 0, 1)
            
            bcp_lh = pd_arr_lh[:, 3:6]
            dcp_lh = pd_arr_lh[:, 6:]
            pairs_lh_pa = pd_arr_lh.shape[0] != 0 and pd_arr_lh is not None

            ##### IMPORTANT - SAUMYA
            pd_arr_gt = cr.computePH(1 - gt_patch) # flipped/negated
            # pd_arr_gt = cr.computePH(gt_patch)
            
            ##### IMPORTANT - SAUMYA
            pd_arr_gt = pd_arr_gt[pd_arr_gt[:, 0] == 0] # 0-dim topology (component) selected
            #pd_arr_gt = pd_arr_gt[pd_arr_gt[:, 0] == 2] # 2-dim topology (component) selected
            
            # pd_arr_gt = pd_arr_gt[pd_arr_gt[:, 2] <= 1]
            # pd_gt = pd_arr_gt[:, 1:3]
            pd_gt = np.clip(pd_arr_gt[:, 1:3], 0, 1)
            
            bcp_gt = pd_arr_gt[:, 3:6]
            dcp_gt = pd_arr_gt[:, 6:]
            pairs_lh_gt = pd_arr_gt.shape[0] != 0 and pd_arr_gt is not None

            # If the pairs not exist, continue for the next loop
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue


            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thresh=0.03)

            if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                for hole_indx in idx_holes_to_fix:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and
                        int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1] and 
                        int(bcp_lh[hole_indx][2]) >= 0 and int(bcp_lh[hole_indx][2]) < likelihood.shape[2]):
                        topo_cp_weight_map[int(bcp_lh[hole_indx][0]), y + int(bcp_lh[hole_indx][1]), x + int(
                            bcp_lh[hole_indx][2])] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[int(bcp_lh[hole_indx][0]), y + int(bcp_lh[hole_indx][1]), x + int(bcp_lh[hole_indx][2])] = 0
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[0] and
                        int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) < likelihood.shape[1] and 
                        int(dcp_lh[hole_indx][2]) >= 0 and int(dcp_lh[hole_indx][2]) < likelihood.shape[2]):
                        topo_cp_weight_map[int(dcp_lh[hole_indx][0]), y + int(dcp_lh[hole_indx][1]), x + int(
                            dcp_lh[hole_indx][2])] = 1  # push death to 1 i.e. max death prob or likelihood
                        topo_cp_ref_map[int(dcp_lh[hole_indx][0]), y + int(dcp_lh[hole_indx][1]), x + int(dcp_lh[hole_indx][2])] = 1
                for hole_indx in idx_holes_to_remove:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and
                        int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1] and 
                        int(bcp_lh[hole_indx][2]) >= 0 and int(bcp_lh[hole_indx][2]) < likelihood.shape[2]):
                        topo_cp_weight_map[int(bcp_lh[hole_indx][0]), y + int(bcp_lh[hole_indx][1]), x + int(
                            bcp_lh[hole_indx][2])] = 1  # push birth to death  # push to diagonal
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[0] and
                            int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) < likelihood.shape[1] and 
                            int(dcp_lh[hole_indx][2]) >= 0 and int(dcp_lh[hole_indx][2]) < likelihood.shape[2]):
                            topo_cp_ref_map[int(bcp_lh[hole_indx][0]), y + int(bcp_lh[hole_indx][1]), x + int(bcp_lh[hole_indx][2])] = \
                                likelihood[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1]), int(dcp_lh[hole_indx][2])]
                        else:
                            topo_cp_ref_map[int(bcp_lh[hole_indx][0]), y + int(bcp_lh[hole_indx][1]), x + int(bcp_lh[hole_indx][2])] = 1
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[0] and
                        int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) < likelihood.shape[1] and 
                        int(dcp_lh[hole_indx][2]) >= 0 and int(dcp_lh[hole_indx][2]) < likelihood.shape[2]):
                        topo_cp_weight_map[int(dcp_lh[hole_indx][0]), y + int(dcp_lh[hole_indx][1]), x + int(
                            dcp_lh[hole_indx][2])] = 1  # push death to birth # push to diagonal
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and
                            int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1] and 
                            int(bcp_lh[hole_indx][2]) >= 0 and int(bcp_lh[hole_indx][2]) < likelihood.shape[2]):
                            topo_cp_ref_map[int(dcp_lh[hole_indx][0]), y + int(dcp_lh[hole_indx][1]), x + int(dcp_lh[hole_indx][2])] = \
                                likelihood[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1]), int(bcp_lh[hole_indx][2])]
                        else:
                            topo_cp_ref_map[int(dcp_lh[hole_indx][0]), y + int(dcp_lh[hole_indx][1]), x + int(dcp_lh[hole_indx][2])] = 0


    savefile(topo_cp_weight_map,savedir,"topo_weight_map.nii.gz")
    savefile(topo_cp_ref_map,savedir,"topo_ref_map.nii.gz")


    if loss_mode == 'bce_cp': indexes = np.nonzero(topo_cp_weight_map)
    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float)#.cuda()
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float)#.cuda()

    # Measuring the MSE loss between predicted critical points and reference critical points
    if loss_mode == "mse":
        # loss_topo = (((likelihood_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum() 
        loss_topo = (((likelihood_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).mean() # it become too low 
    elif loss_mode == "bce":
        loss_topo = F.binary_cross_entropy((likelihood_tensor * topo_cp_weight_map), topo_cp_ref_map)
    elif loss_mode == 'bce_cp':
        if len(indexes[0]) > 0 :
            loss_topo = F.binary_cross_entropy((likelihood_tensor * topo_cp_weight_map), topo_cp_ref_map, reduction='sum')/len(indexes[0])
        else:
            loss_topo = 0.0
    else:
        print("wrong loss mode")
    return loss_topo

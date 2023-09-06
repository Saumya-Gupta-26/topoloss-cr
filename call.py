from __future__ import print_function, division

import torch
from topoloss_saum import getTopoLoss2d, getTopoLoss3d
import SimpleITK as sitk

class TopoLossMSE3D(torch.nn.Module):
    """Weighted Topological loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight_mask=None):
        # pred.size() : [2, 1, 17, 255, 255] # NCDHW
        loss = 0.

        for idx in range(pred.size()[0]): # batchsize=N
            for ch in range(pred.size()[1]): # n_channel=C ; See if we want to perform topoloss on all channels (multi-class), or, only on foreground (binary problem)
                loss += getTopoLoss3d(pred[idx, ch, :, :, : ], target[idx, ch, :, :, : ], 100, 'mse') 
        return loss


class TopoLossMSE2D(torch.nn.Module):
    """Weighted Topological loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight_mask=None):
        # pred.size() : [2, 1, 255, 255] # NCHW
        loss = 0.

        for idx in range(pred.size()[0]): # batchsize=N
            for ch in range(pred.size()[1]): # n_channel=C ; See if we want to perform topoloss on all channels (multi-class), or, only on foreground (binary problem)
                loss += getTopoLoss2d(pred[idx, ch, :, : ], target[idx, ch, :, : ], 100, 'mse') 
        return loss


def savefile(arr, outdir, outname):
    new_sitkimage = sitk.GetImageFromArray(arr)
    sitk.WriteImage(new_sitkimage, os.path.join(outdir, outname))


if __name__ == "__main__":

    savedir = "/scr/saumgupta/kidney-vessel/topo-viz/uncropped"
    #patchsize = [50,100,100]

    predpath = "/scr/saumgupta/kidney-vessel/data/test-results/nnunet_ver2/3d/fold0/50H2_umcropped.npz"
    gtpath = "/scr/saumgupta/kidney-vessel/50H2_umcropped.nii.gz"

    gtsitk = sitk.ReadImage(gtpath)
    gtarray = sitk.GetArrayFromImage(gtsitk)

    patchsize = gtarray.shape # Loss value: 2.2814265321358107e-05
    #print(gtarray.shape) # DHW
    #print(np.min(gtarray), np.max(gtarray)) #0.0, 1.0

    predobj = np.load(predpath, allow_pickle=True)
    for key, myarray in predobj.items():
        predarray = myarray
        #print(predarray.shape) #CDHW ; C=2 because two classes 
        #print(np.min(predarray), np.max(predarray)) #0.0, 1.0

    predcrop = predarray[1,0:patchsize[0],0:patchsize[1],0:patchsize[2]] # taking channel 1 for foreground
    gtcrop = gtarray[0:patchsize[0],0:patchsize[1],0:patchsize[2]]

    assert gtcrop.shape == predcrop.shape # DHW

    predtorch = torch.from_numpy(np.expand_dims(np.expand_dims(predcrop,axis=0), axis=0)) # NCDHW
    gttorch = torch.from_numpy(np.expand_dims(np.expand_dims(gtcrop,axis=0), axis=0)) # NCDHW

    lossclass = TopoLossMSE()
    lossval = lossclass(predtorch, gttorch)
    print("Loss value: {}".format(lossval))

    savefile(predcrop.astype('float32'),savedir,"pred_likelihood.nii.gz")
    savefile(gtcrop,savedir,"gt.nii.gz")

    imgpath = "/scr/saumgupta/kidney-vessel/data/nnunet_ver2/nnUNet_raw_data_base/nnUNet_raw_data/Task270_KIDNEY/imagesTr/50H2_um_0000.nii.gz"
    imgsitk = sitk.ReadImage(imgpath)
    imgarray = sitk.GetArrayFromImage(imgsitk)
    imgcrop = imgarray[0:patchsize[0],0:patchsize[1],0:patchsize[2]]
    savefile(imgcrop,savedir,"img.nii.gz")

    predbinary = sitk.GetArrayFromImage(sitk.ReadImage("/scr/saumgupta/kidney-vessel/data/test-results/nnunet_ver2/3d/fold0/50H2_umcropped.nii.gz"))
    savefile(predbinary[0:patchsize[0],0:patchsize[1],0:patchsize[2]],savedir,"pred_binary.nii.gz")
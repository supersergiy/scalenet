import copy
import torch

def upsample(x, is_res=False, is_pix_res=True):
    if is_res:
        x = x.permute(0, 3, 1, 2)

    result = torch.nn.functional.interpolate(x, scale_factor=2)

    if is_res:
        result = result.permute(0, 2, 3, 1)

    if is_res and is_pix_res:
        result *= 2
    return result

def downsample(res, is_res=False, is_pix_res=True):
    downsampler = torch.nn.AvgPool2d(2)
    result = downsampler(res)
    if is_res and is_pix_res:
        result /= 2
    return result


def res_warp_res(res_a, res_b, is_pix_res=True, rollback=0):
    if len(res_a.shape) == 4:
        res_a_img = res_a.permute(0, 3, 1, 2)
    elif len(res_a.shape) == 3:
        res_a_img = res_a.permute(2, 0, 1)
    else:
        raise Exception("Residual warping requires BxHxWx2 or HxWx2 format.")
    result_perm = res_warp_img(res_a_img, res_b, is_pix_res, rollback)

    if len(res_a.shape) == 4:
        result = result_perm.permute(0, 2, 3, 1)
    elif len(res_a.shape) == 3:
        result = result_perm.permute(1, 2, 0)

    return result


def res_warp_img(img, res_in, is_pix_res=True, rollback=0):
    if is_pix_res:
        res = 2 * res_in / (img.shape[-1])
    else:
        res = res_in
    original_shape = copy.deepcopy(img.shape)

    if len(img.shape) == 4:
        img_unsq = img
        res_unsq = res
    elif len(img.shape) == 3:
        img_unsq = img.unsqueeze(0)
        res_unsq = res.unsqueeze(0)
    elif len(img.shape) == 2:
        img_unsq = img.unsqueeze(0).unsqueeze(0)
        res_unsq = res.unsqueeze(0)
    else:
        raise Exception("Image warping requires BxCxHxW or CxHxW format." +
                        "Recieved dimensions: {}".format(len(img.shape)))

    img_unsq_rollb = img_unsq
    res_unsq_rollb = res_unsq
    for i in range(rollback):
        img_unsq_rollb = upsample(img_unsq_rollb)
        res_unsq_rollb = upsample(res_unsq_rollb, is_res=True)
    result_unsq_rollb = gridsample_residual(img_unsq_rollb, res_unsq_rollb, padding_mode='zeros')

    result_unsq = result_unsq_rollb
    for i in range(rollback):
        result_unsq = downsample(result_unsq)

    result = result_unsq
    while len(result.shape) > len(original_shape):
        result = result.squeeze(0)

    return result


def combine_residuals(a, b, is_pix_res=True, rollback=0):
    return res_warp_res(a, b, is_pix_res=is_pix_res, rollback=rollback) + b


upsampler = torch.nn.UpsamplingBilinear2d(scale_factor=2)
def upsample_residuals(residuals):
    result = upsampler(residuals.permute(
                                     0, 3, 1, 2)).permute(0, 2, 3, 1)
    result *= 2
    return result

def gridsample(source, field, padding_mode):
    if source.shape[2] != source.shape[3]:
        raise NotImplementedError('Grid sample is not impolemented for non-square tensors.')
    scaled_field = field * source.shape[2] / (source.shape[2] - 1)
    return torch.nn.functional.grid_sample(source, scaled_field, mode="bilinear",
                                            padding_mode=padding_mode)

def get_identity_grid(size):
    with torch.no_grad():
        id_theta = torch.cuda.FloatTensor([[[1,0,0],[0,1,0]]]) # identity affine transform
        I = torch.nn.functional.affine_grid(id_theta,torch.Size((1,1,size,size)))
        I *= (size - 1) / size # rescale the identity provided by PyTorch
        return I

def gridsample_residual(source, res, padding_mode):
    size = source.size()[-1]
    field = res + get_identity_grid(size)
    return gridsample(source, field, padding_mode)

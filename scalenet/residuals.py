import torch

def res_warp_res(res_a, res_b, is_pix_res=True):
    if is_pix_res:
        res_b = 2 * res_b / (res_b.shape[-2])

    if len(res_a.shape) == 4:
        result = gridsample_residual(
                        res_a.permute(0, 3, 1, 2),
                        res_b,
                        padding_mode='border').permute(0, 2, 3, 1)
    elif len(res_a.shape) == 3:
        result = gridsample_residual(
                        res_a.permute(2, 0, 1).unsqueeze(0),
                        res_b.unsqueeze(0),
                        padding_mode='border')[0].permute(1, 2, 0)
    else:
        raise Exception("Residual warping requires BxHxWx2 or HxWx2 format.")

    return result


def res_warp_img(img, res_in, is_pix_res=True):

    if is_pix_res:
        res = 2 * res_in / (img.shape[-1])
    else:
        res = res_in

    if len(img.shape) == 4:
        result = gridsample_residual(img, res, padding_mode='zeros')
    elif len(img.shape) == 3:
        result = gridsample_residual(img.unsqueeze(0),
                                     res.unsqueeze(0), padding_mode='zeros')[0]
    elif len(img.shape) == 2:
        result = gridsample_residual(img.unsqueeze(0).unsqueeze(0),
                                     res.unsqueeze(0),
                                     padding_mode='zeros')[0, 0]
    else:
        raise Exception("Image warping requires BxCxHxW or CxHxW format." +
                        "Recieved dimensions: {}".format(len(img.shape)))

    return result


def combine_residuals(a, b, is_pix_res=True):
    return b + res_warp_res(a, b, is_pix_res=is_pix_res)

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

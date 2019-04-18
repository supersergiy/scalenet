import os
import torch

def dict_to_str(d, separator='_'):
    result = ''
    if isinstance(d, dict):
        for k in sorted(d.keys()):
            if result != '':
                result += separator
            result += 'start' + separator + k
            result += separator
            result += dict_to_str(d[k])
            result += separator
            result += 'end' + separator + k
    else:
        result = str(d)
        if separator in result:
            raise ValueError('Separator ({}) cannot occur within transformed dictionary.\
                              Violation: {}'.format(separator, result))
        if d in ['start', 'end']:
            raise ValueError('Keywords "start" and "end" cannot occur within transformed dictionary')

    return result

def upsample_residuals(residuals):
    result = nn.functional.interpolate(residuals.permute(
                                     0, 3, 1, 2), scale_factor=2).permute(0, 2, 3, 1)
    result *= 2

    return result

def create_model_nonportable(model_constructor, constructor_params={},
                             name=None, source_model_path=None,
                             checkpoint_folder='./checkpoints/', cuda=True):
    model_path = os.path.join(checkpoint_folder, "{}.pth.tar".format(name))
    if (name is not None) and (os.path.isfile(model_path)):
        print("Loading from checkpoint")
        model = torch.load(model_path)['model']
    elif source_model_path is not None:
        print("Loading weights from {}".format(init_path))
        model = torch.load(source_model_path)['model']
        model.best_val = sys.maxsize
    else:
        print("Initializing new model {}".format(name))
        model = model_constructor(**constructor_params)

    model.name = name
    if cuda:
        return model.cuda()
    else:
        return model

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
        if separator in d:
            raise ValueError('Separator ({}) cannot occur within transformed dictionary')
        if d in ['start', 'end']:
            raise ValueError('Keywords "start" and "end" cannot occur within transformed dictionary')

    return result

def create_model_nonportable(model_constructor, constructor_params={},
                             name=None, source_model_path=None,
                             checkpoint_dir='./checkpoints/', cuda=True)
    model_path = os.path.join(dheckpoint_dir, "{}.tar.gz")
    if name is not None and os.path.isfile(model_path):
        print("Loading from checkpoint")
        model = torch.load(model_path)['model']
    elif source_model_path is not None:
        print("Loading weights from {}".format(init_path))
        model = torch.load(source_model_path)['model']
        model.best_val = sys.maxsize
    else:
        model = model_constructor(*constructor_params)

    model.name = name
    if cuda:
        return model.cuda()
    else:
        return model

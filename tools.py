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

#!/usr/bin/env python3
import json
import itertools
import os
import time

def generate_args(hyper_path):
    data = None
    with open(hyper_path, 'r') as f:
        data = json.load(f)

    ids2names = [name for name in data.keys()]
    hyperparams = [data[name] for name in ids2names]

    all_combinations = list(itertools.product(*hyperparams))
    print(len(all_combinations))

    # os.mkdir('args.nosync')
    #, ["title", "desc"], ["title", "desc", "narr"]],
    current_doc = all_combinations[0][:9]
    current_config = 0
    os.mkdir('args.nosync')
    os.mkdir(os.path.join('args.nosync', str(current_config)))
    i = 0
    for comb in all_combinations:
        if comb[:9] != current_doc:
            # print(comb[:9])
            current_config +=1
            i = 0
            current_doc = comb[:9]
            os.mkdir(os.path.join('args.nosync', str(current_config)))
            # time.sleep(0.5)
        data = {}
        for hyper, selected in zip(ids2names, comb):
            data[hyper] = selected

        with open(os.path.join('args.nosync', str(current_config), 'args-{}-{}.json'.format(current_config, i)), 'w') as f:
            json.dump(data, f, indent=4)
        i += 1

if __name__=='__main__':
    generate_args('hyper.json')

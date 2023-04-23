import json


def load_summary(dataset):
    dataset = dataset.strip()
    with open('csir/' + dataset + '/' + dataset + '_summary.json') as f:
        current_summary = json.load(f)
    return current_summary


def load_datasets(choose = None):
    with open('csir/summary.json') as f:
        datasets = json.load(f).keys()
        if choose:
            return list(filter(lambda x: x[7] == choose, datasets))
    return datasets

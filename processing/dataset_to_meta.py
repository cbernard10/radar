from dataset_to_summary import load_summary
import numpy as np

def dataset_to_range_times(dataset):
    currSummary = load_summary(dataset)
    rangeStart = currSummary["PCI"]["Range"]
    ranges = np.array(currSummary["PCI"]["RangeOffset"]) + rangeStart
    times = np.array(currSummary["PCI"]["Time"])
    return {
        'ranges': ranges,
        'times': times
    }

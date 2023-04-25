from .dataset_to_summary import load_summary, load_datasets
import numpy as np

def get_trajectory(ds):

    summ = load_summary(ds)

    # proclat = summ['GPS']['Track']['Proc']['Lat']
    # proclong = summ['GPS']['Track']['Proc']['Long']
    rawlong = np.array(summ['GPS']['Track']['Raw']['Long']).squeeze()
    rawlat = np.array(summ['GPS']['Track']['Raw']['Lat']).squeeze()
    times = np.array(summ['GPS']['Track']['Time']).squeeze()
    dtimes = [times[i+1] - times[i] for i in range(len(times) - 1)]

    xydt = np.vstack((rawlong, rawlat, [0] + dtimes))

    xydt_ = []
    for i in range(xydt.shape[-1]):
        if xydt[-1, i] != 0 or i == 0:
            xydt_.append(xydt[:, i])

    xydt = np.array(xydt_).T
    dlong = [ (xydt[0, i+1] - xydt[0, i]) / xydt[-1, i+1] for i in range(xydt.shape[-1]-1) ]
    dlat = [(xydt[1, i + 1] - xydt[1, i]) / xydt[-1, i + 1] for i in range(xydt.shape[-1] - 1)]
    speed = [np.sqrt(dlat[i] ** 2 + dlong[i] ** 2) for i in range(len(dlat))]

    return xydt[0], xydt[1], speed

def get_radial_trajectory(ds):
    summ = load_summary(ds)
    r, az = np.array(summ['GPS']['Track']['Raw']['Range']).squeeze(), np.array(summ['GPS']['Track']['Raw']['Azimuth']).squeeze()
    times = np.array(summ['GPS']['Track']['Time']).squeeze()
    or_ = summ['PCI']['Range']
    ranges = or_ + np.array(summ['PCI']['RangeOffset'])
    return r, az, times, ranges

def get_ds_with_GPS_data():

    dss = load_datasets()
    arr = []
    for ds in dss:
        summ = load_summary(ds)

        if 'GPS' in summ.keys():
            arr.append(ds)

    return arr
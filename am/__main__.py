# 3rd party
import numpy as np

# builtin
import os

# local
import am

DATADIR_RAW = "/home/shared/netfabb_ti64_hires_raw/"
DATADIR_OUT = "/home/shared/netfabb_ti64_hires_out/"


if __name__ == "__main__":

    # am.extract(DATADIR_RAW, DATADIR_OUT)
    datafile = os.path.join(DATADIR_OUT, r"data_0-100", "101635_11b839a3_5.npz")
    data = np.load(datafile)

    pass
#

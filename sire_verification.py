import os
import sys
import shutil
from datetime import datetime
import platform
import numpy as np
import pandas as pd

# import openpyxl
font = {'size': 8}
import matplotlib

matplotlib.rc("font", **font)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker

from shapely.geometry import Point, LineString, Polygon

import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature

import shapefile
import flopy
import pyemu

sys.path.append("sire")
import hauraki_sire

verf_d = "sire_verification"
master_d = os.path.join("..",hauraki_sire.master_d)
pst_name = "hauraki_resp.pst"

def setup_dir():
    if os.path.exists(verf_d):
        shutil.rmtree(verf_d)
    #os.mkdir(verf_d)
    files = os.listdir(master_d)
    # for f in files:
    #     if not f.endswith(".jcb"):
    #         shutil.copy2(os.path.join(master_d,f),os.path.join(verf_d,f))
    shutil.copytree(master_d,verf_d)
    os.remove(os.path.join(verf_d,pst_name.replace(".pst",".jcb")))
    shutil.copytree("pyemu",os.path.join(verf_d,"pyemu"))


def random_test():
    pst = pyemu.Pst(os.path.join(verf_d, pst_name))
    par = pst.parameter_data
    load_pars = par.loc[par.pargp == "kg_load", "parnme"]
    par.loc[par.pargp!="kg_load","partrans"] = "fixed"
    #print(load_pars)

    pe = pyemu.ParameterEnsemble.from_uniform_draw(pst=pst,num_reals=1)
    par.loc[pe.columns,"loading_change"] = pe.iloc[0,:]
    print(par.loading_change)
    resp_vec = hauraki_sire.sire(par)
    print(resp_vec)
    par.loc[:,"parval1"] = par.loading_change
    pst.control_data.noptmax = 0
    case_name = "hauraki_sire_random_verf.pst"
    pst.write(os.path.join(verf_d,case_name))
    pyemu.os_utils.run("pestpp {0}".format(case_name),cwd=verf_d)
    return case_name



def compare_test(case_name):
    pst = pyemu.Pst(os.path.join(verf_d,case_name))
    par = pst.parameter_data
    par.loc[:, "loading_change"] = par.parval1
    print(par.loading_change)
    resp_vec = hauraki_sire.sire(par)
    diff = (resp_vec.obsval - pst.res.modelled).dropna()
    print(diff)
    print(resp_vec.loc[diff.index,"obsval"])
    print(pst.res.loc[diff.index,"modelled"])

if __name__ == "__main__":
    #setup_dir()
    #random_test()
    compare_test("hauraki_sire_random_verf.pst")






import os
import shutil
import platform
from datetime import datetime
import numpy as np
import pandas as pd
import flopy
import pyemu

font = {'size'   : 8}
import matplotlib as mpl
mpl.rc("font",**font)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker



base_pst_file = "restart.pst"
base_dir = os.path.join("ouu","base")
scen_pst_file = "scenario.pst"
scen_dir = os.path.join("ouu","scenario")
exe_name = "pestpp-opt"
dv_group = "k1"
par_file = os.path.join(scen_dir,scen_pst_file.replace(".pst",".1.par"))


if "linux" in platform.platform().lower():
    bin_dir = os.path.join("ouu","bin","linux")
    ext = ''
elif "darwin" in platform.platform().lower():
    bin_dir = os.path.join("ouu","bin","mac")
    ext = ''
else:
    bin_dir = os.path.join("ouu","bin","win")
    ext = ".exe"
    

def get_const_dict():
    pst = load_pst()
    const_names = {name:1.0 for name in pst.nnz_obs_names if pst.observation_data.loc[name,"obgnme"].startswith("less_")}
    return const_names

def load_pst():
    pst = pyemu.Pst(os.path.join(base_dir,base_pst_file))
    pst.control_data.noptmax = 1
    pst.pestpp_options = {}
    pst.pestpp_options["opt_dec_var_groups"] = dv_group
    pst.pestpp_options["base_jacobian"] = "restart.jcb"
    pst.pestpp_options["hotstart_resfile"] = "restart.rei"
    pst.pestpp_options["opt_skip_final"] = True
    pst.pestpp_options["opt_std_weights"] = True
    pst.pestpp_options["opt_direction"] = "max"
    return pst

def scrape_recfile(recfile):
    infeas = False
    with open(recfile,'r') as f:
        for line in f:
            if "best objective function value" in line:
                phi = float(line.strip().split(':')[-1])
                #break

            if "warning: primal solution infeasible" in line:
                infeas = True
    return infeas, phi


def run_pestpp_opt(const_dict,risk=0.5):
    if os.path.exists(scen_dir):
        shutil.rmtree(scen_dir)
    shutil.copytree(base_dir,scen_dir)
    shutil.copy2(os.path.join(bin_dir,exe_name+ext),os.path.join(scen_dir,exe_name+ext))
    pst_scen = load_pst()
    pst_scen.pestpp_options["opt_risk"] = risk
    obs = pst_scen.observation_data
    for const_name,const_percent_change in const_dict.items():
        const_name = const_name.lower()
        assert const_name in obs.obsnme
        obs.loc[const_name,"obsval"] *= const_percent_change
    pst_scen.write(os.path.join(scen_dir,scen_pst_file))
    pyemu.os_utils.run("{0} {1}".format(exe_name,scen_pst_file),cwd=scen_dir)
    infeas,phi = scrape_recfile(os.path.join(scen_dir,scen_pst_file.replace(".pst",".rec")))
    
    return infeas,phi

def plot_scenario_dv(infeas=False):
    assert os.path.exists(par_file)
    m = flopy.modflow.Modflow.load("freyberg.truth.nam",model_ws=scen_dir,load_only=[])
    ib = m.bas6.ibound[0].array
    pst = load_pst()
    obs = pst.observation_data.loc[pst.nnz_obs_names,:]



    gw_obs = obs.loc[obs.obsnme.apply(lambda x: x.startswith("ucn")),:].copy()
    gw_obs.loc[:,"i"] = gw_obs.obsnme.apply(lambda x: int(x.split('_')[2]))
    gw_obs.loc[:,"j"] = gw_obs.obsnme.apply(lambda x: int(x.split('_')[3]))
    gw_obs.loc[:,"x"] = gw_obs.apply(lambda x: m.sr.xcentergrid[x.i,x.j],axis=1)
    gw_obs.loc[:,"y"] = gw_obs.apply(lambda x: m.sr.ycentergrid[x.i,x.j],axis=1)
    
    sw_x = m.sr.xcentergrid[30,14]
    sw_y = m.sr.ycentergrid[30,14]
    sw_arr = np.zeros(ib.shape)
    sw_arr[:,14] = 1.0
    sw_arr = np.ma.masked_where(sw_arr==0,sw_arr)
    
    fig = plt.figure(figsize=(5,7))
    ax = plt.subplot(111)
    if infeas:
        ax.imshow(sw_arr,extent=m.sr.get_extent())
        ax.scatter(gw_obs.x,gw_obs.y,marker='*',color='r',s=100,label="gw constraint")
        ax.scatter([sw_x],[sw_y],marker='^',color='r',s=100,label="sw constraint")
        ax.text(0.5,0.5,"INFEASIBLE\n#SAD",fontsize=30,
            ha="center",va="center",color="0.5",transform=ax.transAxes)

        return fig,ax

    df = pyemu.pst_utils.read_parfile(par_file)
    df.loc[:,"i"] = df.parnme.apply(lambda x: int(x.split('_')[1]))
    df.loc[:,"j"] = df.parnme.apply(lambda x: int(x.split('_')[2]))
    arr = np.zeros(ib.shape)
    arr[df.i,df.j] = df.parval1
    arr = np.ma.masked_where(ib==0,arr)
    
    cb = ax.imshow(arr,alpha=0.75,extent=m.sr.get_extent(),cmap="plasma")
    ax.imshow(sw_arr,extent=m.sr.get_extent())
    cb = plt.colorbar(cb)
    cb.set_label("$\\frac{kg}{d}$")
    ax.scatter(gw_obs.x,gw_obs.y,marker='*',color='r',s=100,label="gw constraint")
    ax.scatter([sw_x],[sw_y],marker='^',color='r',s=100,label="sw constraint")
    return fig,ax

def run_scenario(const_dict={},risk=0.5):
    infeas,phi = run_pestpp_opt(const_dict,risk=risk)
    fig,ax = plot_scenario_dv(infeas)
    ax.set_title("phi: {0:<15.3G}, risk: {1:<15.3G}".format(phi,risk))
    return fig,ax

if __name__ == "__main__":
    #run_pestpp_opt({})
    #fig,ax = plot_scenario_dv(True)
    fig,ax = run_scenario({})
    plt.show()





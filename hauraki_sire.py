import os
import shutil
from datetime import datetime
import json
import scipy
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

# import cartopy
# import cartopy.crs as ccrs
# import cartopy.io.img_tiles as cimgt
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# import cartopy.feature

import shapefile

master_d = "master_resp"

sire_d = "sire"

try:
    mprj = ccrs.epsg(2193)
except:
    print("no mprj")

def prep_lu_df():
    lu_cats = ["dairy", "forestry", "hort", "other", "snb"]
    arr = np.loadtxt(os.path.join("_data","dnz_processed_kg_m2_d.dat"))
    r = shapefile.Reader(os.path.join("shapes", "lc_grid_join.shp"))
    fnames = [i[0].lower() for i in r.fields[1:]]
    shp_df = pd.DataFrame(r.records(), columns=fnames)
    print(shp_df.columns)
    columns = [c for c in shp_df.columns if c.split('_')[0] in lu_cats and "_f" in c]
    print(columns)
    df = shp_df.loc[:,columns]
    df.columns = df.columns.map(lambda x: x.split('_')[0])
    df.loc[:,'i'] = shp_df.row - 1
    df.loc[:,'j'] = shp_df.column - 1
    df.loc[:,"parnme"] = shp_df.apply(lambda x: "kg_{0:03.0f}_{1:03.0f}".format(x.row-1,x.column-1),axis=1)
    df.index = df.parnme
    df.loc[:,"base_load"] = df.apply(lambda x: arr[x.i,x.j],axis=1)
    df.to_csv(os.path.join(sire_d,"lu_fracs.csv"))

def prep_numerics():
    if os.path.exists(sire_d):
        shutil.rmtree(sire_d)
    os.mkdir(sire_d)
    jco_file = os.path.join(sire_d,"hauraki_resp.jcb")
    pst_file = os.path.join(sire_d, "hauraki_resp.pst")
    shutil.copy2(os.path.join(master_d,"hauraki_resp.jcb"),jco_file)
    shutil.copy2(os.path.join(master_d, "hauraki_resp.pst"), pst_file)

    pst = pyemu.Pst(pst_file)
    obs = pst.observation_data
    obs.loc[:,"weight"] = 0.0
    forecasts = list(obs.loc[obs.apply(lambda x: x.obgnme=="ucn1_0" and x.obsnme.endswith("_010"),axis=1),"obsnme"])
    #print(forecasts)
    sfrc = list(obs.loc[obs.apply(lambda x: x.obgnme=="sfrc" and "_1_" in x.obsnme and "3650.0" in x.obsnme, axis=1),"obsnme"])
    forecasts.extend(sfrc)
    print(len(sfrc),len(forecasts))
    #forecasts = sfrc
    #print(forecasts)
    par = pst.parameter_data
    par_names = list(par.loc[par.pargp != "kg_load","parnme"])
    dv_names = list(par.loc[par.pargp == "kg_load","parnme"])

    pst.parameter_data = pst.parameter_data.loc[par_names,:]
    cov = pyemu.Cov.from_parameter_data(pst,sigma_range=6.0)
    jco = pyemu.Jco.from_binary(jco_file)

    fore_jco = jco.get(row_names=forecasts,col_names=par_names)
    resp_mat = jco.get(row_names=forecasts,col_names=dv_names)
    resp_mat.to_binary(os.path.join(sire_d,"resp_mat.jcb"))
    la = pyemu.LinearAnalysis(pst=pst,parcov=cov,forecasts=fore_jco.T,verbose=True)
    prior_fore = la.prior_forecast
    with open(os.path.join(sire_d,"forecast_std.csv"),'w') as f:
        f.write("obsnme,obsval\n")
        for n,v in prior_fore.items():
            f.write("{0},{1}\n".format(n,np.sqrt(v)))


def spike_test(spike_dict,risk=0.5):

    std_df = pd.read_csv(os.path.join(sire_d,"forecast_std.csv"))
    std_df.index = std_df.obsnme
    std_df.loc[:,"totim"] = std_df.obsnme.apply(lambda x: float(x.split('_')[-1]))
    std_df.loc[:,"reach"] = std_df.obsnme.apply(lambda x: int(x.split('_')[0].replace("sfrc",'')))

    resp_mat = pyemu.Jco.from_binary(os.path.join(sire_d,"resp_mat.jcb"))

    par_df = pd.DataFrame({"parnme":resp_mat.col_names},index=resp_mat.col_names)
    par_df.loc[:,"i"] = par_df.parnme.apply(lambda x: int(x.split('_')[1]))
    par_df.loc[:, "j"] = par_df.parnme.apply(lambda x: int(x.split('_')[2]))
    par_df.loc[:,"ij"] = par_df.apply(lambda x: (x.i,x.j),axis=1)

    std_df = std_df.loc[resp_mat.row_names,:]

    par_df.loc[:,"loading_change"] = 0.0
    par_df.index = par_df.ij
    for ij,load_change in spike_dict.items():
        #par_df.loc[par_df.apply(lambda x: x.i in spike_i and x.j in spike_j,axis=1),"load_change"] = 1.0
        par_df.loc[ij,"loading_change"] = load_change
    print(par_df.loading_change.sum())

    # resp_vec = resp_mat * par_df.loading_change.values
    # resp_vec = list(resp_vec.x[0,:])
    # std_df.loc[:,"response"] = resp_vec
    #
    # if risk != 0.5:
    #     assert risk > 0.0
    #     assert risk < 1.0
    #     std_df.loc[:,"response_org"] = std_df.response
    #     std_df.loc[:,"offset"] = (risk - 0.5) * std_df.obsval
    #     std_df.loc[:,"response"] += std_df.offset
    # std_df.to_csv(os.path.join(sire_d,"sire_results.csv"))

    std_df = sire(par_df,risk=risk)
    return std_df



def _load_reach_dict():
    seg_dict = {}
    with open(os.path.join(sire_d,"reach_verts.dat"),'r') as f:
        for line in f:
            raw = line.strip().split()
            seg = int(raw[0])
            if seg in seg_dict:
                raise Exception()
            xs,ys = [],[]
            for xy in raw[1:]:
                x,y = xy.split(',')
                xs.append(float(x))
                ys.append(float(y))
            seg_dict[seg] = [xs,ys]
    return seg_dict

def plot_sire(df,loading_df,spike_dict=None,tol=1.0e-4,show=False):
    m = flopy.modflow.Modflow.load("BH.nam", model_ws="template", forgive=False, verbose=True, check=False,
                                   load_only=[])
    if spike_dict is not None:

        spike_x = [m.sr.xcentergrid[k[0],k[1]] for k in spike_dict.keys()]
        spike_y = [m.sr.ycentergrid[k[0], k[1]] for k in spike_dict.keys()]

    plt_dir = "plot_sire"
    if os.path.exists(plt_dir):
        shutil.rmtree(plt_dir)
    os.mkdir(plt_dir)
    df.loc[:,"response"] *= 1000.0 # kg/m3 to mg/l
    # if totims is None:
    #     totims = df.totim.unique()
    #     show = False
    # else:
    #     all_totims = df.totim.unique()
    #     for totim in totims:
    #         assert totim in totims
    #     df = df.loc[df.totim.apply(lambda x: x in totims)]

    df_sfr = df.loc[df.index.map(lambda x: x.startswith("sfr")), :].copy()
    df_sfr.loc[:,"reach"] = df_sfr.obsnme.apply(lambda x: int(x.split('_')[0][4:]))
    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=df_sfr.response.min(), vmax=df_sfr.response.max())
    colors = plt.cm.plasma(df_sfr.response.values / df_sfr.response.max())
    df_sfr.loc[:,"c_idx"] = np.arange(df_sfr.shape[0])
    reach_dict = _load_reach_dict()


    df_ucn = df.loc[df.index.map(lambda x: x.startswith("ucn")),:].copy()
    df_ucn.loc[:, "i"] = df_ucn.obsnme.apply(lambda x: int(x.split('_')[2]))
    df_ucn.loc[:, "j"] = df_ucn.obsnme.apply(lambda x: int(x.split('_')[3]))
    resp_arr = np.zeros((m.nrow,m.ncol))
    resp_arr[df_ucn.i,df_ucn.j] = df_ucn.response
    resp_arr = np.ma.masked_where(np.abs(resp_arr)<tol,resp_arr)

    # for seg,(x,y) in seg_dict.items():
    #   ax.plot(x,y)

    load_arr = np.zeros((m.nrow,m.ncol))
    load_arr[loading_df.i,loading_df.j] = 100.0 * (loading_df.loading_change / loading_df.base_load)
    #load_arr = np.ma.masked_where(np.abs(resp_arr)<tol,resp_arr)
    cb = plt.imshow(load_arr)
    plt.colorbar(cb)
    plt.show()
    return

    fig = plt.figure(figsize=(8.5,11))
    ax = plt.subplot(111,aspect="equal")
    ax2 = plt.axes((0.8, 0.1, 0.025, 0.8))
    ax3 = plt.axes((0.9, 0.1, 0.025, 0.8))
    c = ax.pcolormesh(m.sr.xcentergrid,m.sr.ycentergrid,resp_arr,alpha=0.5)
    plt.colorbar(c, cax=ax3,orientation="vertical")
    for i,reach in enumerate(df_sfr.reach):
        if np.abs(df_sfr.response.iloc[i]) < tol:
            color="0.5"
            lw=0.5
        else:
            color = colors[df_sfr.c_idx.iloc[i]]
            lw=1.5
        x,y = reach_dict[reach]
        ax.plot(x,y,color=color,lw=lw)
    if spike_dict is not None:
        ax.scatter(spike_x,spike_y,marker='x',s=40)


    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label("change N concen ($\\Delta\\frac{mg}{l})$")
    if not show:
        plt.savefig(os.path.join(plt_dir,"sire_{0:03d}.png".format(itotim)),dpi=300)
        plt.close(fig)

    return fig


def _get_base_ax(ax=None):
    terrain = cimgt.StamenTerrain()

    ax = plt.subplot(1, 1, 1, projection=mprj)

    ax.set_extent([175.15, 176.2, -36.95, -38.25])

    # Add the Stamen data at zoom level 8.
    ax.add_image(terrain, 9, cmap="Greys")
    return ax


def prep_plotting():
    r = shapefile.Reader(os.path.join("shapes","sfr_link.shp"))
    fields = [i[0].lower() for i in r.fields[1:]]
    df = pd.DataFrame(r.records(),columns=fields)
    pts = [shp.points for shp in r.shapes()]
    df.loc[:,"pts"] = pts
    print(df)
    with open(os.path.join(sire_d,"reach_verts.dat"),'w') as f:
        #f.write("reach,x,y\n")
        for reach,pts in zip(df.reachid,df.pts):
            f.write("{0} ".format(reach))
            for (x,y) in pts:
                f.write(" {0},{1} ".format(x,y))
            f.write("\n")



def sire(loading_df, risk=0.5):
    std_df = pd.read_csv(os.path.join(sire_d, "forecast_std.csv"))
    std_df.index = std_df.obsnme
    #std_df.loc[:, "totim"] = std_df.obsnme.apply(lambda x: float(x.split('_')[-1]))
    #std_df.loc[:, "reach"] = std_df.obsnme.apply(lambda x: int(x.split('_')[0].replace("sfrc", '')))

    assert "loading_change" in loading_df.columns
    resp_mat = pyemu.Jco.from_binary(os.path.join(sire_d, "resp_mat.jcb"))

    # align
    loading_df = loading_df.loc[resp_mat.col_names, :]
    std_df = std_df.loc[resp_mat.row_names,:]

    # mat-vec mult
    resp_vec = resp_mat * loading_df.loading_change.values

    std_df.loc[:, "response"] = resp_vec.x.flatten()

    if risk != 0.5:
        assert risk > 0.0
        assert risk < 1.0
        probit = -np.sqrt(2.0) * scipy.special.erfcinv((2.0 * risk))
        std_df.loc[:, "response_org"] = std_df.response
        #std_df.loc[:,"logit"] = np.log(risk/(1.0 - risk))
        #std_df.loc[:, "offset"] = (risk - 0.5) * std_df.obsval
        std_df.loc[:,"probit"] = probit
        std_df.loc[:,"offset"] = std_df.probit * std_df.obsval
        std_df.loc[:, "response"] += std_df.offset
    else:
        std_df.loc[:, "response_org"] = std_df.response
        std_df.loc[:, "probit"] = 0.0
        std_df.loc[:, "offset"] = 0.0
        #std_df.loc[:, "response"] += std_df.offset
    std_df.to_csv(os.path.join(sire_d, "sire_results.csv"))
    return std_df


def sire_lu_scenario(lu_change_dict,risk=0.5):
    std_df = pd.read_csv(os.path.join(sire_d, "forecast_std.csv"))
    std_df.index = std_df.obsnme

    #std_df.loc[:, "totim"] = std_df.obsnme.apply(lambda x: float(x.split('_')[-1]))
    #std_df.loc[:, "reach"] = std_df.obsnme.apply(lambda x: int(x.split('_')[0].replace("sfrc", '')))

    lu_df = pd.read_csv(os.path.join(sire_d,"lu_fracs.csv"),index_col=0)
    lu_df.loc[:,"parnme"] = lu_df.index
    loading_vec = pd.DataFrame({"parnme":lu_df.parnme,"loading_change":0.0})
    loading_vec.index = lu_df.index

    for lu,change in lu_change_dict.items():
        if lu not in lu_df.columns:
            raise Exception(lu+" not in lu_df.columns")
        loading_vec.loc[:,"loading_change"] += lu_df.loc[:,lu].values * lu_df.base_load.values * (change / 100.0)

    #resp_mat = pyemu.Jco.from_binary(os.path.join(sire_d, "resp_mat.jcb"))
    #loading_vec = loading_vec.loc[resp_mat.col_names,:]
    #resp_vec = resp_mat * loading_vec.loading_change.values
    loading_vec.loc[:,"base_load"] = lu_df.base_load.values
    loading_vec.loc[:,"i"] = lu_df.i.values
    loading_vec.loc[:,"j"] = lu_df.j.values
    std_df = sire(loading_vec,risk=risk)
    return loading_vec,std_df


def sire_lu_scenario_json(lu_change_dict,risk=0.5):
    tol = 1.0e-3
    loading_df,df = sire_lu_scenario(lu_change_dict,risk=risk)
    with open(os.path.join("sire", "sire_sfr.json"), "r") as f:
        sfr_data = json.load(f)

    with open(os.path.join("sire", "sire_grid.json"), "r") as f:
        ucn_data = json.load(f)

    with open(os.path.join("sire", "sire_grid.json"), "r") as f:
        load_data = json.load(f)
    loading_df.loc[:, "rowcol"] = loading_df.apply(lambda x: "{0:03d}_{1:03d}".format(x.i + 1, x.j + 1), axis=1)
    loading_df.index = loading_df.rowcol
    #loading_df.loc[:, "percent_change"] = (100.0 * (loading_df.loading_change / loading_df.base_load))
    loading_df.loc[:, "normed"] = mpl.colors.Normalize()(loading_df.loading_change.values)
    rdict = loading_df.loading_change.to_dict()
    colormap = mpl.cm.jet
    loading_df.loc[:, "color"] = [mpl.colors.rgb2hex(d[0:3]) for d in colormap(loading_df.normed.values)]
    cdict = loading_df.color.to_dict()
    for i, feature in enumerate(ucn_data["features"]):
        rowcol = feature["properties"]["rowcol"]
        if pd.isnull(loading_df.loc[rowcol, "loading_change"]):
            rdict[rowcol] = 0.0
            color = "#676565"
        elif rowcol not in rdict:
            rdict[rowcol] = 0.0
            color = "#676565"
        elif np.abs(loading_df.loc[rowcol, "loading_change"]) < tol:
            color = "#676565"
        else:
            color = cdict[rowcol]
        load_data["features"][i]["properties"]["style"]["fillColor"] = color
        load_data["features"][i]["properties"]["style"]["color"] = color
        load_data["features"][i]["properties"]["style"]["weight"] = 2.0
        load_data["features"][i]["properties"]["response"] = rdict[rowcol]

    df.loc[:, "response"] *= 1000
    df.loc[:,"response_org"] *= 1000
    df.loc[:,"obsval"] *= 1000

    df_sfr = df.loc[df.obsnme.apply(lambda x: x.startswith('sfr')), :].copy()
    df_sfr.loc[:, "reach"] = df_sfr.obsnme.apply(lambda x: int(x.split('_')[0][4:]))
    df_sfr.index = df_sfr.reach
    df_sfr.loc[:, "normed"] = mpl.colors.Normalize()(df_sfr.response.values)
    rdict = df_sfr.response.to_dict()
    rdict_org = df_sfr.response_org.to_dict()
    std_dict = df_sfr.obsval.to_dict()
    logit_dict = df_sfr.probit.to_dict()
    off_dict = df_sfr.offset.to_dict()
    colormap = mpl.cm.jet
    df_sfr.loc[:, "color"] = [mpl.colors.rgb2hex(d[0:3]) for d in colormap(df_sfr.normed.values)]
    cdict = df_sfr.color.to_dict()
    for i, feature in enumerate(sfr_data["features"]):
        reachid = feature["properties"]["reachID"]
        if np.abs(df_sfr.loc[reachid, "response"]) < tol:
            color = "#676565"
        else:
            color = cdict[reachid]
        sfr_data["features"][i]["properties"]["style"]["fillColor"] = color
        sfr_data["features"][i]["properties"]["style"]["color"] = color
        sfr_data["features"][i]["properties"]["style"]["weight"] = 2.0
        sfr_data["features"][i]["properties"]["response"] = rdict[reachid]
        sfr_data["features"][i]["properties"]["response_org"] = rdict_org[reachid]
        sfr_data["features"][i]["properties"]["std"] = std_dict[reachid]
        sfr_data["features"][i]["properties"]["logit"] = logit_dict[reachid]
        sfr_data["features"][i]["properties"]["offset"] = off_dict[reachid]





    df_ucn = df.loc[df.index.map(lambda x: x.startswith("ucn")),:].copy()
    df_ucn = df_ucn.loc[df_ucn.response.apply(np.abs)<1.0e+10,:]
    df_ucn.loc[:, "i"] = df_ucn.obsnme.apply(lambda x: int(x.split('_')[2]))
    df_ucn.loc[:, "j"] = df_ucn.obsnme.apply(lambda x: int(x.split('_')[3]))
    df_ucn.loc[:,"rowcol"] = df_ucn.apply(lambda x: "{0:03d}_{1:03d}".format(x.i+1,x.j+1),axis=1)
    df_ucn.index = df_ucn.rowcol
    df_ucn.loc[:,"normed"] = mpl.colors.Normalize()(df_ucn.response.values)
    rdict = df_ucn.response.to_dict()
    rdict_org = df_ucn.response_org.to_dict()
    std_dict = df_ucn.obsval.to_dict()
    colormap=mpl.cm.jet
    df_ucn.loc[:,"color"] = [mpl.colors.rgb2hex(d[0:3]) for d in colormap(df_ucn.normed.values)]
    cdict = df_ucn.color.to_dict()
    for i,feature in enumerate(ucn_data["features"]):
        rowcol = feature["properties"]["rowcol"]
        if rowcol not in rdict:
            rdict[rowcol] = 0.0
            rdict_org[rowcol] = 0.0
            std_dict[rowcol] = 0.0
            color = "#676565"
        elif np.abs(df_ucn.loc[rowcol,"response"]) < tol:
            color="#676565"
        else:
            color = cdict[rowcol]
        ucn_data["features"][i]["properties"]["style"]["fillColor"] = color
        ucn_data["features"][i]["properties"]["style"]["color"] = color
        ucn_data["features"][i]["properties"]["style"]["weight"] = 2.0
        ucn_data["features"][i]["properties"]["response"] = rdict[rowcol]
        ucn_data["features"][i]["properties"]["response_org"] = rdict_org[rowcol]
        ucn_data["features"][i]["properties"]["std"] = std_dict[rowcol]


    return sfr_data,ucn_data,load_data


def filter_resp_mat():
	mat = pyemu.Matrix.from_binary(os.path.join(sire_d,"resp_mat.jcb"))
	#x = mat.x
	#x[x<1.0e-15] = 0.0
	mat.to_binary(os.path.join(sire_d,"test.jcb"),droptol=1.0e-13)



	x = mat.x.flatten()
	x = x[x>0.0]
	x = np.log10(x)
	x = x[~np.isnan(x)]
	print(x.min(),x.max())
	plt.hist(x)
	plt.show()


if __name__ == "__main__":
	filter_resp_mat()
    #prep_numerics()
    #prep_plotting()
    #prep_lu_df()

    # spike_dict = {(69, 34): 1000.0}  # ,(56,38):10000}
    # risk = 0.95
    # df = spike_test(spike_dict,risk=risk)
    # plot_sire(df,spike_dict=spike_dict)
    #

    # change is increase or decrease of N loading (kg/day) for a given land use sector
    #lu_change_dict = {"dairy":-1.0,"snb":+1}
    # start = datetime.now()
    # loading_df,result_df = sire_lu_scenario(lu_change_dict=lu_change_dict,risk=0.5)
    # sire_end = datetime.now()
    #
    # plot_sire(result_df,loading_df,show=True)#,18615.0])
    # plt.show()
    # plot_end = datetime.now()
    # sire_duration = (sire_end - start).total_seconds()
    # plot_duration = (plot_end - sire_end).total_seconds()
    # print("sire: {0}, plot: {1}".format(sire_duration,plot_duration))


    #j1,j2,j3 = sire_lu_scenario_json(lu_change_dict=lu_change_dict, risk=0.5)


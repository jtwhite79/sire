
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import flopy
import pyemu
model_ws = "den_exp"


def setup_mf_model():
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    m = flopy.modflow.Modflow("den",model_ws=model_ws,version="mfnwt",exe_name="mfnwt")
    d = flopy.modflow.ModflowDis(m,nrow=1,nlay=2,ncol=10,nper=1,delr=100,delc=100,top=10,
                                 botm=[-5,-100],perlen=365.25*20,steady=True)
    d = flopy.modflow.ModflowBas(m,strt=10)
    d = flopy.modflow.ModflowLpf(m,laytyp=1,hk=10.0,vka=1.0)
    d = flopy.modflow.ModflowPcg(m)
    d = flopy.modflow.ModflowOc(m)#,chedfm="({0}E15.6)".format(m.ncol))
    d = flopy.modflow.ModflowLmt(m,output_file_format="formatted")
    ghb_data = [[0,0,0,-4.0,100.0],[0,0,m.ncol-1,10.0,100.0]]
    ghb = flopy.modflow.ModflowGhb(m,stress_period_data={0:ghb_data})
    m.write_input()
    m.run_model()
    hds = flopy.utils.HeadFile(os.path.join(m.model_ws,m.name+".hds"))


def setup_mt_model(mws = model_ws):

    m = flopy.modflow.Modflow.load("den.nam",model_ws=mws,check=False)
    mt = flopy.mt3d.Mt3dms("den_mt3d",model_ws=mws,modflowmodel=m,exe_name="mt3dusgs",external_path='.')
    nper = 20
    perlen = np.zeros((nper)) + 365.25
    d = flopy.mt3d.Mt3dBtn(mt,nper=nper,sconc=0.0,prsity=0.1,
                           perlen=perlen,nstp=1,tsmult=1.0)
    d = flopy.mt3d.Mt3dGcg(mt)
    d = flopy.mt3d.Mt3dAdv(mt,mixelm=0)
    d = flopy.mt3d.Mt3dDsp(mt,al=0.01,trpt=0.1)
    d = flopy.mt3d.Mt3dRct(mt,isothm=0,ireact=1,igetsc=0,rc1=0.01)
    id = flopy.mt3d.Mt3dSsm.itype_dict()
    ssm_data_load = [[0,0,m.ncol-1,100.0,15],[0,0,0,0.0,id["GHB"]],[0,0,m.ncol-1,0.0,id["GHB"]]]
    ssm_data = {kper:ssm_data_load for kper in range(mt.nper)}
    # ssm_data_off = ssm_data_load.copy()
    # ssm_data_off[0][-2] = 0.0
    # for kper in range(int(mt.nper/2),mt.nper):
    #     ssm_data[kper] = ssm_data_off

    d = flopy.mt3d.Mt3dSsm(mt,stress_period_data=ssm_data,mxss=3)
    mt.write_input()
    mt.run_model()

    hds = flopy.utils.HeadFile(os.path.join(mws, m.name + ".hds"))
    ucn = flopy.utils.UcnFile(os.path.join(mt.model_ws,"MT3D001.UCN"),model=mt)
    #fig = plt.figure(figsize=(10,2))
    data = ucn.get_data()[0,0,:]
    ucn = flopy.utils.UcnFile(os.path.join(mt.model_ws, "MT3D001.UCN"))
    fig = plt.figure(figsize=(10, 2))
    #df = pd.DataFrame({"ucn": ucn.get_data()[0, 0, :], "hds": hds.get_data()[0, 0, :]})
    #df.plot(kind="bar", color='b', subplots=True)
    #plt.show()
    return mt


def write_ssm_tpl(mws):
    fname = os.path.join(mws,"den_mt3d.ssm")
    f_in = open(fname,'r')
    f_tpl = open(fname+".tpl",'w')
    f_tpl.write("ptf ~\n")
    c = 0
    vals,names = [],[]
    while True:
        line = f_in.readline()
        if line == '':
            break

        if "stress" in line:

            f_tpl.write(line)
            line = f_in.readline()
            name = "kg_{0:02d}".format(c)
            val = float(line[31:41])
            names.append(name)
            vals.append(val)
            line = line[:31] + "~ {0} ~ ".format(name) + line[41:]
            c += 1
        f_tpl.write(line)

    f_in.close()
    f_tpl.close()
    df = pd.DataFrame({"parnme":names,"parval1":vals},index=names)
    return df


def setup_pest():

    pst_helper = pyemu.helpers.PstFromFlopyModel("den.nam", org_model_ws=model_ws, new_model_ws="template",
                                         remove_existing=True,
                                         hds_kperk=[0, 0],
                                         grid_props=[["extra.rc11", 0], ["extra.rc11", 1]],
                                         model_exe_name="mfnwt",extra_post_cmds="mt3dusgs den_mt3d.nam")
    #mm = flopy.modflow.Modflow.load("den.nam", model_ws="template",check=False)
    pst_helper.m.run_model(silent=True)
    mt = setup_mt_model(mws='template')
    pst = pst_helper.pst

    [shutil.copy2(os.path.join("template", f), os.path.join("template", 'arr_org', f)) for f in
     os.listdir("template") if "rc11" in f.lower()]

    bdir = os.getcwd()
    os.chdir("template")

    df = write_ssm_tpl(".")
    pst.add_parameters("den_mt3d.ssm.tpl","den_mt3d.ssm")
    pst.parameter_data.loc[df.parnme,"parval1"] = df.parval1
    pst.parameter_data.loc[df.parnme,"pargp"] = "load"

    ucn_kperk = []
    for kper in np.arange(mt.nper):
        ucn_kperk.append([kper, 0])

    fline, df = pyemu.gw_utils.setup_hds_obs("mt3d001.ucn", skip=1.0e+30, kperk_pairs=ucn_kperk,                                             prefix="ucn1")
    pst_helper.frun_post_lines.append(fline)
    pst_helper.tmp_files.append("mt3d001.ucn")
    ucn1 = pst.add_observations(os.path.join("mt3d001.ucn.dat.ins"), os.path.join("mt3d001.ucn.dat"))

    os.chdir(bdir)

    pst_helper.write_forward_run()

    par = pst.parameter_data
    par.loc[:,"partrans"] = "none"
    par.loc[:,"parlbnd"] = 0.0
    par.loc[:, "parubnd"] = 1.0e+10


    pst.control_data.noptmax = 0
    pst_helper.pst.write(os.path.join("template","den.pst"))
    shutil.copy2(os.path.join("pestpp.exe"),os.path.join("template","pestpp.exe"))
    shutil.copy2(os.path.join("mfnwt.exe"),os.path.join("template","mfnwt.exe"))
    shutil.copy2(os.path.join("mt3dusgs.exe"),os.path.join("template","mt3dusgs.exe"))
    pyemu.os_utils.run("pestpp den.pst",cwd="template")



def run_test1():
    pst = pyemu.Pst(os.path.join("template","den.pst"))
    base_res = pst.res
    par = pst.parameter_data
    par.loc[:,"partrans"] = "none"
    load_pars = par.loc[par.pargp=="load","parnme"]
    rc_pars = par.loc[par.pargp!="load","parnme"]

    par.loc[load_pars[1:],"partrans"] = "tied"
    par.loc[load_pars[1:], "partied"] = load_pars[0]
    #par.loc[rc_pars[1:], "partrans"] = "tied"
    #par.loc[rc_pars[1:], "partied"] = rc_pars[0]
    par.loc[rc_pars,"partrans"] = "fixed"

    par.loc[rc_pars,"parval1"] = 0.0
    pst.control_data.noptmax = -1
    dfs = []

    for load in [1,1000]:
        par.loc[load_pars,"parval1"] = load
        pst_name = "test_load_{0}.pst".format(load)

        pst.write(os.path.join("template",pst_name))
        shutil.copy2(os.path.join("pestpp.exe"),os.path.join("template","pestpp.exe"))
        shutil.copy2(os.path.join("mfnwt.exe"),os.path.join("template","mfnwt.exe"))
        shutil.copy2(os.path.join("mt3dusgs.exe"),os.path.join("template","mt3dusgs.exe"))
        pyemu.os_utils.run("pestpp {0}".format(pst_name),cwd="template")

        df = pyemu.Jco.from_binary(os.path.join("template",pst_name.replace(".pst",".jcb"))).to_dataframe()
        dfs.append(df)
        #print(df)
        #break
    df_noden = pd.concat(dfs,axis=1)
    df_noden.to_csv(os.path.join("template","no_den.csv"))

    par.loc[rc_pars, "parval1"] = 1.0
    pst.control_data.noptmax = -1
    dfs = []

    for load in [1, 1000]:
        par.loc[load_pars, "parval1"] = load
        pst_name = "test_load_{0}.pst".format(load)

        pst.write(os.path.join("template", pst_name))
        shutil.copy2(os.path.join("pestpp.exe"),os.path.join("template","pestpp.exe"))
        pyemu.os_utils.run("pestpp {0}".format(pst_name), cwd="template")

        df = pyemu.Jco.from_binary(os.path.join("template", pst_name.replace(".pst", ".jcb"))).to_dataframe()
        dfs.append(df)
        # print(df)
        # break

    df_den = pd.concat(dfs, axis=1)
    df_den.to_csv(os.path.join("template", "den.csv"))

def run_test2():
    pst = pyemu.Pst(os.path.join("template","den.pst"))
    base_res = pst.res
    par = pst.parameter_data
    par.loc[:,"partrans"] = "none"
    load_pars = par.loc[par.pargp=="load","parnme"]
    rc_pars = par.loc[par.pargp!="load","parnme"]

    par.loc[rc_pars,"parval1"] = 0.0
    pst.control_data.noptmax = 0 # fwd only
    cout = []

    for load in [0.1,1,10,100]:
        par.loc[load_pars,"parval1"] = load
        pst_name = "test_load_{0}.pst".format(load)

        pst.write(os.path.join("template",pst_name))
        shutil.copy2(os.path.join("pestpp.exe"),os.path.join("template","pestpp.exe"))
        shutil.copy2(os.path.join("mfnwt.exe"),os.path.join("template","mfnwt.exe"))
        shutil.copy2(os.path.join("mt3dusgs.exe"),os.path.join("template","mt3dusgs.exe"))
        pyemu.os_utils.run("pestpp {0}".format(pst_name),cwd="template")

        df = pd.read_table(os.path.join("template","mt3d001.ucn.dat"),index_col=0,sep=' ')
        co = df.loc["ucn1_00_000_000_009",:] # just downstream-most cell
        cout.append(co)
    couts = pd.concat(cout)
    print(couts)
    couts.to_csv(os.path.join("template","no_den_conc.csv"))

    par.loc[rc_pars, "parval1"] = 1.0
    pst.control_data.noptmax = 0
    cout = []

    for load in [0.1,1,10,100]:
        par.loc[load_pars, "parval1"] = load
        pst_name = "test_load_{0}.pst".format(load)

        pst.write(os.path.join("template", pst_name))
        shutil.copy2(os.path.join("pestpp.exe"),os.path.join("template","pestpp.exe"))
        pyemu.os_utils.run("pestpp {0}".format(pst_name), cwd="template")

        df = pd.read_table(os.path.join("template","mt3d001.ucn.dat"),index_col=0,sep=' ')
        co = df.loc["ucn1_00_000_000_009",:] # just downstream-most cell
        cout.append(co)
    couts = pd.concat(cout)
    print(couts)
    couts.to_csv(os.path.join("template","den_conc.csv"))

def plot_test1():
    pst = pyemu.Pst(os.path.join("template", "den.pst"))
    obs = pst.observation_data
    onames = obs.loc[obs.obsnme.apply(lambda x: x.startswith("ucn1_00") and x.endswith("_009")),"obsnme"]
    df_noden = pd.read_csv(os.path.join("template","no_den.csv"),index_col=0).loc[onames,:]
    print(df_noden)

    df_den = pd.read_csv(os.path.join("template", "den.csv"), index_col=0).loc[onames,:]
    print(df_den)

    fig = plt.figure(figsize=(10,10))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    df_noden.columns = ["1kg","1000kg"]
    df_den.columns = ["1kg", "1000kg"]

    df_noden.plot(kind="bar",ax=ax1)
    df_den.plot(kind="bar", ax=ax2)

    ax1.set_title("sensitivity to load, no de-N")
    ax2.set_title("sensitivity to load, with de-N")

    ax1.set_xticklabels([])


    plt.savefig(os.path.join("template","test1.pdf"))
    plt.close(fig)

def plot_test2():
    df_noden = pd.read_csv(os.path.join("template","no_den_conc.csv"),index_col=0,header=None)
    print(df_noden)

    df_den = pd.read_csv(os.path.join("template", "den_conc.csv"),index_col=0,header=None)
    print(df_den)

    dfs = pd.concat((df_noden,df_den),axis=1,)
    dfs.columns = ["first-order de-N rate = 0.0 $d^{-1}$","first-order de-N rate = 0.01 $d^{-1}$"]
    dfs.index = ["0.1kg","1kg","10kg","100kg"]

    fig = plt.figure()
    ax = plt.subplot(111)

    dfs = np.log10(dfs)
    dfs.plot(kind="line",ax=ax,)

    ax.set_ylabel("$log_{10}$ conc @ outflow")
    ax.set_xlabel("$log_{10}$ load")
    ax.set(xticks=range(len(dfs.index)), xticklabels=(dfs.index))

    plt.savefig(os.path.join("template","test2.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    #setup_mf_model()
    #setup_mt_model()
    #setup_pest()
    #write_ssm_tpl("template")
    #run_test1()
    #plot_test1()
    run_test2()
    plot_test2()
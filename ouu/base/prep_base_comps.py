import numpy as np
import pyemu

pst = pyemu.Pst("restart_org.pst")
jco = pyemu.Jco.from_binary("restart_org.jcb")

# get the less than constraints and set to zero weight
obs = pst.observation_data
forecast_names = list(obs.loc[obs.apply(lambda x: x.weight > 0 and x.obgnme.startswith("less_"),axis=1), "obsnme"].values)
obs.loc[forecast_names, "weight"] = 0.0

sc = pyemu.Schur(pst=pst, jco=jco, forecasts=forecast_names)
prior_fore = sc.prior_forecast.items()

# reload since the sc extract the forecast rows
pst = pyemu.Pst("restart_org.pst")
jco = pyemu.Jco.from_binary("restart_org.jcb")
rei = pyemu.pst_utils.read_resfile("restart_org.rei")

for n,v in prior_fore:
    pst.observation_data.loc[n,"weight"] = np.sqrt(v)
#sqrt then set weight

keep_obs = pst.nnz_obs_names
keep_pars = list(pst.parameter_data.loc[pst.parameter_data.pargp == "k1", "parnme"].values)

pst.observation_data = pst.observation_data.loc[keep_obs, :]
pst.parameter_data = pst.parameter_data.loc[keep_pars,:]
rei = rei.loc[keep_obs, :]
jco = jco.get(row_names=keep_obs, col_names=keep_pars)

jco.to_binary("restart.jcb")
pst.write("restart.pst")
with open("restart.rei", 'w') as f:
    f.write(" MODEL OUTPUTS AT END OF OPTIMISATION ITERATION NO. -999:-\n\n\n")
    f.flush()
    rei.to_csv(f, sep=" ", mode='a')

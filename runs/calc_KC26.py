import sys
import os
import shutil
sys.path.append("../reqlocal")
from pathlib import Path
from yaml import safe_load

from reqlocal import REQSolver, SolverParameter
from reqlocal.constants import C, EV, M_ATOMIC, M_SUN, DAY
from reqlocal.ion import IonStateSolver

import numpy as np
import numpy.ma as ma

def config_initalize(params_ej: dict[str, float], solver_mode: str = "all_Tarumi", r_uv: float = 1.0,
                     norm_rr: bool =  False, use_Sobolev_method: bool = True, use_Hotokezaka_SF_res: bool = True,
                     cbbt_data: str = "modified", SrIonEnergy_data: str= "extended", without_thermal_col: bool = False, 
                     check_lte: bool = False, xtol: float | None = None) -> dict:
    
    config = {
        "setup":{
            "solver_mode": solver_mode,
            "norm_rr": norm_rr,
            "use_Sobolev_method": use_Sobolev_method,
            "use_Hotokezaka_SF_res": use_Hotokezaka_SF_res,
            "cbbt_data": cbbt_data,
            "SrIonEnergy_data": SrIonEnergy_data,
            "without_thermal_col": without_thermal_col,
            "check_lte": check_lte,
            "xtol": xtol
        },
        "ejecta":{}
    }

    config["ejecta"] = dict(**params_ej)
    config["ejecta"]["r_uv"] = r_uv

    return config

def get_rho_ph(vel_c, t_expl_day, m_ej_msun, alpha = 3, vel_in_c = 0.05, vel_out_c = 0.35):
    r_out_in = vel_out_c/vel_in_c
    beta = alpha - 3

    term1 = 1/(4.0*np.pi*(vel_c*C*t_expl_day*DAY)**3)
    term2 = (vel_c/vel_in_c)**(-beta)

    if np.isclose(beta, 0.0):
        term3 = 1/np.log(r_out_in)
    else:
        term3 = beta/(1.0-(r_out_in)**(-beta))

    rho_ph = term1 * term2 * term3 * m_ej_msun * M_SUN

    return rho_ph

mu_ion = lambda X_He, X_Sr, A_env: (X_He/4.0 + X_Sr/88.0 + (1.0-X_He-X_Sr)/A_env)**-1
qdot_beta_erg = lambda t_expl_day, X_He, X_Sr, A_env: 1e10 * (t_expl_day)**-1.3 * (1.0-X_He) * mu_ion(X_He, X_Sr, A_env) * M_ATOMIC


if __name__ == '__main__':
    params_file_path = Path("params_kn17gfo_KC26.yml")

    with open(params_file_path) as fp:
        params = safe_load(fp)

    params_ej = params["ejecta_parameters"]
    fac_tem_e = params["fac_tem_e"]

    vel_in_c = params["vel_in_c"]
    vel_out_c = params["vel_out_c"]
    alpha = params["alpha"]
    m_ej_msun = params["m_ej_msun"]

    A_env = params["A_env"]
    qdot_scale = params["qdot_scale"]
    X_He_seq = np.logspace(*params["X_He_seq"].values())
    X_Sr_seq = np.logspace(*params["X_Sr_seq"].values())

    vel_ph_c = params_ej.pop("vel_ph_c")
    params_ej["tem_e"] = fac_tem_e * params_ej["tem_r"]

    # make save directory
    t_expl_day = params_ej["t_expl"]
    rho_ej = get_rho_ph(vel_ph_c, t_expl_day, m_ej_msun, alpha=alpha, vel_in_c=vel_in_c, vel_out_c=vel_out_c)
    tem_e = params_ej["tem_e"]
    tem_r = params_ej["tem_r"]
    pr_esc_rr_gr = params_ej["pr_esc_rr_gr"]
    ion_Sr = params_ej["with_Sr_pop"]
    w_geo = params_ej["w_geo"]
    save_base_name = f"{t_expl_day:.2f}d_{tem_e:.2e}K_{tem_r:.2e}K_{rho_ej:.2e}gpcc_{pr_esc_rr_gr:.2f}drr_{qdot_scale:.2f}_{w_geo:.2f}_Sr-{ion_Sr}"

    # day to second
    params_ej["t_expl"] *= DAY

    print(save_base_name)
    save_dir = Path("res_kn17gfo_KC26", save_base_name)
    # os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir)

    shutil.copy(params_file_path, save_dir)

    logfpath = Path(save_dir, "log.txt")
    with open(logfpath, mode='w') as logf:
        # nd_levels + free electron
        num_levels = 21
        nd_res_He_pop = np.zeros((num_levels+1, len(X_Sr_seq), len(X_He_seq)))
        mask_nd_res_He_pop = np.zeros_like(nd_res_He_pop, dtype=int)

        ISsolver_Sr = IonStateSolver(SrIonEnergy_data="extended")
        nd_res_Sr_pop = np.empty((ISsolver_Sr.ion_deg_seq.size, len(X_Sr_seq), len(X_He_seq)))
        mask_nd_res_Sr_pop = np.zeros_like(nd_res_Sr_pop, dtype=int)


        r_heat_array = np.empty((len(X_Sr_seq), len(X_He_seq)))

        for idx_X_Sr, X_Sr in enumerate(X_Sr_seq):
            for idx_X_He, X_He in enumerate(X_He_seq):
                if (X_He + X_Sr) >= 1.0:
                    mask_nd_res_He_pop[:, idx_X_Sr, idx_X_He] = 1
                    mask_nd_res_Sr_pop[:, idx_X_Sr, idx_X_He] = 1
                    
                    # It is expected that "dict_rate_mat_each_res_array" is already defined in this condition.
                    for process in dict_rate_mat_each_res_array.keys():
                        dict_rate_mat_each_res_array[process]["_mask"][..., idx_X_Sr, idx_X_He] = 1

                    print("X_He + X_Sr >= 1.0", file=logf)
                    print("X_He + X_Sr >= 1.0")
                    print(f"X_He: {idx_X_He} loop is finished", end='\n\n', file=logf)
                    print(f"X_He: {idx_X_He} loop is finished", end='\n\n')
                    continue
                
                r_heat = qdot_scale * qdot_beta_erg(t_expl_day, X_He, X_Sr, A_env)
                r_heat_array[idx_X_Sr, idx_X_He] = r_heat
                params_ej["r_heat"] = r_heat
                params_ej["nd_He"] = X_He*rho_ej/(4.0*M_ATOMIC)
                params_ej["nd_Sr"] = X_Sr*rho_ej/(88.0*M_ATOMIC)
                params_ej["nd_env"] = (1.0-X_He-X_Sr)*rho_ej/(A_env*M_ATOMIC)

                config = config_initalize(params_ej)
                print(config, file=logf)
                print(config)                
                reqsolver = REQSolver.set_config(config)
                reqsolver.solve(anounce=False)
                print(reqsolver.sol.message, file=logf)
                print(reqsolver.sol.message)
                nd_res_He_pop[:, idx_X_Sr, idx_X_He] = reqsolver.nd_vec
                nd_res_Sr_pop[:, idx_X_Sr, idx_X_He] = reqsolver.ISsolver_Sr.frac_ion * params_ej["nd_Sr"]

                dict_rate_mat_each_res = reqsolver.rate_mat_each_res
                if (idx_X_He == 0) & (idx_X_Sr == 0):
                    dict_rate_mat_each_res_array = {
                        k: ma.array(np.zeros((num_levels, num_levels, X_Sr_seq.size, X_He_seq.size)), 
                                    mask=np.zeros((num_levels, num_levels, X_Sr_seq.size, X_He_seq.size), dtype=int)
                                    ).torecords() 
                        for k in dict_rate_mat_each_res.keys()
                        }
                for process in dict_rate_mat_each_res_array.keys():
                    dict_rate_mat_each_res_array[process]["_data"][..., idx_X_Sr, idx_X_He] = dict_rate_mat_each_res[process]
                
                if reqsolver.sol.status != 1:
                    mask_nd_res_He_pop[:, idx_X_Sr, idx_X_He] = 1
                    mask_nd_res_Sr_pop[:, idx_X_Sr, idx_X_He] = 1
                    for process in dict_rate_mat_each_res_array.keys():
                        dict_rate_mat_each_res_array[process]["_mask"][..., idx_X_Sr, idx_X_He] = 1

                elif not np.all(reqsolver.nd_vec >= 0.0):
                    print("negative values exist", file=logf)
                    print("negative values exist")
                    print(reqsolver.nd_vec, file=logf)
                    print(reqsolver.nd_vec)
                    mask_nd_res_He_pop[:, idx_X_Sr, idx_X_He] = 1
                    mask_nd_res_Sr_pop[:, idx_X_Sr, idx_X_He] = 1  
                    for process in dict_rate_mat_each_res_array.keys():
                        dict_rate_mat_each_res_array[process]["_mask"][..., idx_X_Sr, idx_X_He] = 1
                          
                print(f"X_He: {idx_X_He} loop is finished", end='\n\n', file=logf)
                print(f"X_He: {idx_X_He} loop is finished", end='\n\n')
            
            print(f"X_Sr: {idx_X_Sr} loop is finished", end='\n\n', file=logf)
            print(f"X_Sr: {idx_X_Sr} loop is finished", end='\n\n\n')

    nd_res_He_pop = ma.array(nd_res_He_pop, mask=mask_nd_res_He_pop)
    nd_res_Sr_pop = ma.array(nd_res_Sr_pop, mask=mask_nd_res_Sr_pop)

    # save results
    fpath_nd = Path(save_dir, f"nd_{save_base_name}.npz")
    np.savez(fpath_nd, X_He_seq=X_He_seq, X_Sr_seq=X_Sr_seq, r_heat_array=r_heat_array, 
             nd_res_He_pop=nd_res_He_pop.torecords(), nd_res_Sr_pop=nd_res_Sr_pop.torecords())

    # fpath_process = Path(save_dir, f"process_{save_base_name}.npz")
    # np.savez(fpath_process, X_He_seq=X_He_seq, X_Sr_seq=X_Sr_seq, r_heat_array=r_heat_array, **dict_rate_mat_each_res_array) 
 
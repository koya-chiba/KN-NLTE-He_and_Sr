from pathlib import Path
import numpy.typing as npt

import numpy as np
from scipy import optimize
import pandas as pd

from reqlocal.constants import C, H, K_B, EV, DAY
from reqlocal.base import DATA_HE_DIR, SolverParameter
from reqlocal.environment import EjectaParameter
from reqlocal.transition import PhotoIonization, RadiativeRecombination, ThermalCollisionBFFB,\
                       RadiativeBB, ThermalCollisionBB, NonThermalCollisionIonization,\
                       dep_frac_ion
from reqlocal.ion import IonStateSolver


class REQSolver:
    def __init__(self, setup: SolverParameter, ejecta: EjectaParameter):
        self._solved = False

        self.setup = setup
        self.ejecta = ejecta

        if self.ejecta.with_Sr_pop is not None:
            match self.ejecta.with_Sr_pop:
                case "LTE":
                    self.ISsolver_Sr = IonStateSolver(SrIonEnergy_data=setup.SrIonEnergy_data, tem_e=ejecta.tem_e, w_geo=ejecta.w_geo)
                case "NLTE":
                    self.ISsolver_Sr = IonStateSolver(SrIonEnergy_data=setup.SrIonEnergy_data, tem_e=ejecta.tem_e, w_geo=ejecta.w_geo, qdot_nt=ejecta.r_heat, flag_nt=True)

        self.dim_rate_mat = setup.dim_rate_mat
        self.nd_levels = None
        self.nd_e = None
        self.nd_vec = None
        self.frac_e_env = self.ejecta.frac_e_env_init

        # for mk_rate_eq
        self.idx_replace = 1

        # metadata of use state 
        self.use_id = setup.use_id

        if (np.log10(ejecta.tem_r) < self.setup.log_tem_range[0]) | (np.log10(ejecta.tem_r) > self.setup.log_tem_range[1]):
            raise ValueError(f"Ejecta temperature must take a value between {10**self.setup.log_tem_range[0]:.3f} K and {10**self.setup.log_tem_range[1]:.3f} K.")

        if self.setup.check_lte:
            self.processes = (
                ThermalCollisionBFFB(setup, ejecta),
                ThermalCollisionBB(setup, ejecta),
            )

        else:
            if setup.without_thermal_col:
                self.processes = (
                    PhotoIonization(setup, ejecta),
                    RadiativeRecombination(setup, ejecta),
                    RadiativeBB(setup, ejecta),
                    NonThermalCollisionIonization(setup, ejecta)
                )
            else:
                self.processes = (
                    PhotoIonization(setup, ejecta),
                    RadiativeRecombination(setup, ejecta),
                    ThermalCollisionBFFB(setup, ejecta),
                    RadiativeBB(setup, ejecta),
                    ThermalCollisionBB(setup, ejecta),
                    NonThermalCollisionIonization(setup, ejecta)
                )

        self.mk_rate_mat_array = tuple(process.mk_rate_mat for process in self.processes)
        self.mk_rate_mat_init_array = tuple(process.mk_rate_mat_init for process in self.processes)

        self.nd_initialize()

    @classmethod
    def set_config(cls, config: dict):
        setup = SolverParameter(**config["setup"])
        ejecta = EjectaParameter.set_ejecta_parameter(**config["ejecta"])
                
        return cls(setup, ejecta)
    
    def nd_initialize(self) -> None:
        self.nd_levels = np.zeros(self.dim_rate_mat) 
        self.nd_vec = np.zeros(self.dim_rate_mat+1)
        
        self.nd_e = self.ejecta.nd_He
        if self.ejecta.nd_env is not None:
            self.nd_e += self.frac_e_env*self.ejecta.nd_env
        if self.ejecta.with_Sr_pop is not None:
            self.nd_e += self.frac_e_env*self.ejecta.nd_Sr

        if self.setup.check_lte:
            ISsolver_He = IonStateSolver(atom_name='He', nd_e=self.nd_e, tem_e=self.ejecta.tem_e, w_geo=self.ejecta.w_geo)
        else:
            ISsolver_He = IonStateSolver(atom_name='He', nd_e=self.nd_e, tem_e=self.ejecta.tem_e, w_geo=self.ejecta.w_geo,
                                         qdot_nt=self.ejecta.r_heat, flag_nt=True)
        ISsolver_He.solve()
        frac_He_I, frac_He_II, frac_He_III = ISsolver_He.frac_ion

        self.nd_levels[0] = frac_He_I * self.ejecta.nd_He
        if self.setup.solver_mode in ['seven', 'n3_Tarumi', 'all_Tarumi']:
            self.nd_levels[-2] = frac_He_II * self.ejecta.nd_He
            self.nd_levels[-1] = frac_He_III * self.ejecta.nd_He
        else:
            self.nd_levels[-1] = (1.0 - frac_He_I) * self.ejecta.nd_He

        rate_mat_init = self.mk_rate_mat_init(self.nd_e)
        rate_mat_init[1] = np.ones(self.dim_rate_mat)
        b = np.zeros(self.dim_rate_mat)
        b[1] = self.ejecta.nd_He

        self.nd_levels = np.linalg.inv(rate_mat_init) @ b
        self.nd_vec[0] = self.nd_e
        self.nd_vec[1:] = self.nd_levels

    def mk_rate_mat(self, nd_levels: npt.NDArray[np.float64], nd_e: float):
        # sum over each matrix
        rate_mat = sum([mk_rate_mat_each(nd_levels, nd_e) for mk_rate_mat_each in self.mk_rate_mat_array])

        rate_mat -= np.diag(np.sum(rate_mat, axis=0))

        return rate_mat
    
    def mk_rate_mat_init(self, nd_e: float):
        # sum over each matrix
        rate_mat = sum([mk_rate_mat_init_each(nd_e) for mk_rate_mat_init_each in self.mk_rate_mat_init_array])

        rate_mat -= np.diag(np.sum(rate_mat, axis=0))

        return rate_mat
    
    def mk_rate_eq(self, nd_vec: npt.NDArray[np.float64]):
        nd_e = nd_vec[0]
        if nd_e <= 0.0:
            nd_e = self.nd_e
        nd_levels = nd_vec[1:]
        
        nd_e_new = 0.0
        if self.setup.solver_mode in ['seven', 'n3_Tarumi', 'all_Tarumi']:
            nd_e_new += nd_levels[-2] + 2*nd_levels[-1]
        else:
            nd_e_new += nd_levels[-1] 
        if self.ejecta.with_Sr_pop is not None:
            self.ISsolver_Sr.nd_e = nd_e
            self.ISsolver_Sr.solve()
            # frac_e_env is forced to match with frac_e of Sr
            self.frac_e_env = self.ISsolver_Sr.frac_e
            nd_e_new += self.frac_e_env * self.ejecta.nd_Sr
        if self.ejecta.nd_env is not None:
            nd_e_new += self.frac_e_env * self.ejecta.nd_env
        
        rate_mat = self.mk_rate_mat(nd_levels, nd_e_new)
        # due to make the matrix full rank
        rate_mat[self.idx_replace] = np.ones(self.dim_rate_mat)

        rate_mat_aug = np.zeros((self.dim_rate_mat+1, self.dim_rate_mat+1))
        rate_mat_aug[0, 0] = 1.0
        rate_mat_aug[1:, 1:] = rate_mat

        b = np.zeros(self.dim_rate_mat+1)
        b[0] = nd_e_new
        b[self.idx_replace+1] = self.ejecta.nd_He

        rate_eq = rate_mat_aug @ nd_vec - b

        return rate_eq
    
    def solve(self, anounce=True):
        assert not self._solved, "rate equation is already solved."

        if self.setup.xtol is None:
            self.sol = optimize.root(self.mk_rate_eq, self.nd_vec)
        else:
            self.sol = optimize.root(self.mk_rate_eq, self.nd_vec, options={"xtol":self.setup.xtol})
    
        if anounce:
            print(self.sol.message, end='\n\n')

        self.nd_vec = self.sol.x

        self.nd_e = self.nd_vec[0]
        self.nd_levels = self.nd_vec[1:]
            
        self._solved = True

    @property
    def rate_mat_each_res(self):
        assert self._solved, "rate equation is not solved yet."

        return {process.name: mk_rate_mat_each(self.nd_levels, self.nd_e) for process, mk_rate_mat_each in zip(self.processes, self.mk_rate_mat_array)}
    
    @property
    def rate_mat_res(self):
        assert self._solved, "rate equation is not solved yet."

        return self.mk_rate_mat(self.nd_levels, self.nd_e)
    
    @property
    def rate_eq_res(self):
        assert self._solved, "rate equation is not solved yet."

        nd_e = self.nd_vec[0]
        nd_levels = self.nd_vec[1:]
                
        rate_mat = self.mk_rate_mat(nd_levels, nd_e)
        # due to make the matrix full rank
        rate_mat[self.idx_replace] = np.ones(self.dim_rate_mat)

        rate_mat_aug = np.zeros((self.dim_rate_mat+1, self.dim_rate_mat+1))
        rate_mat_aug[0, 0] = 1.0
        rate_mat_aug[1:, 1:] = rate_mat

        b = np.zeros(self.dim_rate_mat+1)
        b[0] = nd_e
        b[self.idx_replace+1] = self.ejecta.nd_He

        rate_eq = rate_mat_aug @ self.nd_vec - b

        return rate_eq


    
    def output_result(self):
        assert self._solved, "rate equation is not solved yet."

        rate_mat_each_res = self.rate_mat_each_res
        rate_mat_res = self.rate_mat_res

        np.set_printoptions(formatter={'float': '{:.3e}'.format})

        print("calculation configuration")
        print("----------------------------------------------------------")
        print(self.setup)
        print(self.ejecta)
        print()

        print("main physical parameter")
        print("----------------------------------------------------------")
        print(f"{self.ejecta.t_expl/DAY:.3e} days after the explosion")
        print(f"ejecta radiation temperature = {self.ejecta.tem_r:.3e} K")
        print(f"ejecta electron temperature = {self.ejecta.tem_e:.3e} K")
        print()
        print(f"geometric dilution factor = {self.ejecta.w_geo:.3e}")
        print(f"He total number density = {self.ejecta.nd_He:.3e} cm^-3")
        print()
        wpi_I_ev = self.setup.wpi_He_ev[0]
        wpi_II_ev = self.setup.wpi_He_ev[1]
        print(f"work per ion pair for He taken form Tarumi+ (2023) = {int(wpi_I_ev)} eV (He I), {int(wpi_II_ev)} eV (He II)")
        if self.setup.use_Hotokezaka_SF_res:
            print("* Non-thermal ionization rate is calculated with work per ion pair values above,")
            print("* so work per ion pair values below are just formal values.")
        if self.setup.solver_mode in ['seven', 'n3_Tarumi', 'all_Tarumi']:
            x_e_He = (self.nd_levels[-2]+2*self.nd_levels[-1])/self.ejecta.nd_He
            print(f"electron fraction (He ionization degree) = {x_e_He:.3e}")
            dfi_I = dep_frac_ion(x_e_He, 'I')
            dfi_II = dep_frac_ion(x_e_He, 'II')
            e_ion_I_ev = self.setup.use_id["level (eV)"].iat[-2]
            e_ion_II_ev = self.setup.use_id["level (eV)"].iat[-1]
            print(f"non-thermal energy fraction going into He I ionization = {dfi_I:.3e}")
            print(f"work per ion pair for He I (ionization potential) = {e_ion_I_ev/dfi_I:.3e} eV ({e_ion_I_ev:.3e} eV)")
            print(f"non-thermal energy fraction going into He II ionization = {dfi_II:.3e}")
            print(f"work per ion pair for He II (ionization potential) = {e_ion_II_ev/dfi_II:.3e} eV ({e_ion_II_ev:.3e} eV)")

        else:
            x_e_He = (self.nd_levels[-1])/self.ejecta.nd_He
            print(f"electron fraction (He ionization degree) = {x_e_He:.3e}")
            dfi_I = dep_frac_ion(x_e_He, 'I')
            e_ion_I_ev = self.setup.use_id["level (eV)"].iat[-2]
            print(f"non-thermal energy fraction going into He I ionization = {dfi_I:.3e}")
            print(f"work per ion pair for He I (ionization potential) = {e_ion_I_ev/dfi_I:.3e} eV ({e_ion_I_ev:.3e} eV)")


        print(f"non-thermal heating rate = {self.ejecta.r_heat/EV:.3e} eV s^-1 ion^-1")
        print()
        print(f"Threshold photon energy for UV line blanketing = {self.ejecta.uv_bk_prm.e_ph_thr:.3e} eV ({1e8*H*C/(self.ejecta.uv_bk_prm.e_ph_thr*EV):.3e} A)")
        print(f"UV photon rate compared with blackbody radiation = {self.ejecta.uv_bk_prm.r_uv:.3e}", end="\n\n")

        print("He level population [cm^-3]")
        for i in range(self.nd_levels.size):
            print(f"{i} {self.use_id["term"].iat[i]}:{self.nd_levels[i]:.3e}")

        if self.setup.solver_mode in ['seven', 'n3_Tarumi', 'all_Tarumi']:
            print('\n'+f"electron number density from He = {self.nd_levels[-2] + 2*self.nd_levels[-1]:.3e} cm^-3")
        else:
            print('\n'+f"electron number density from He = {self.nd_levels[-1]:.3e} cm^-3")

        if self.ejecta.with_Sr_pop is not None:
            print()
            print(f"Sr {self.ejecta.with_Sr_pop} ionization model")
            print(f"Sr total number density = {self.ejecta.nd_Sr:.3e} cm^-3")
            print(f"Sr average ionization degree = {self.ISsolver_Sr.average_ion_deg:.3f}")
            print(f"electron fraction of Sr = {self.ISsolver_Sr.frac_e:.3f}")
            print(f"electron number density from Sr = {self.frac_e_env * self.ejecta.nd_Sr:.3e} cm^-3")

            print("degree | fraction | number density [cm^-3]")
            for ion_deg, frac_ion in zip(self.ISsolver_Sr.ion_deg_seq, self.ISsolver_Sr.frac_ion):
                print(f"{ion_deg} | {frac_ion:.3e} | {frac_ion * self.ejecta.nd_Sr:.3e}")

        if self.ejecta.nd_env is not None:
            print('\n'+f"number density of environmental nuclei = {self.ejecta.nd_env:.3e} cm^-3")
            print(f"electron fraction of environmental nuclei = {self.frac_e_env:.3f}")
            print(f"electron number density from environmental nuclei = {self.frac_e_env*self.ejecta.nd_env:.3e} cm^-3")

        print('\n'+f"total free electron number density = {self.nd_e:.3e} cm^-3", end='\n\n')

        print("final residuals of rate equation")
        print(f"the location where the conservation of numbers is incorporated: {self.idx_replace+1:d}")
        flow_max_seq = np.max(np.abs(self.rate_mat_res*self.nd_levels), axis=-1)
        weighted_residual = np.empty_like(self.nd_vec)
        weighted_residual[0] = self.rate_eq_res[0] / self.nd_e # must be zero by definition
        weighted_residual[1:] = self.rate_eq_res[1:] / flow_max_seq
        print("Unweighted:")
        print(self.rate_eq_res, '\n')
        print("Weighted (1/max(abs(flow))):")
        print(weighted_residual, end='\n\n')
        
        print("transition matrix [s^-1]")
        print("----------------------------------------------------------")
        for name_each, rate_mat_each in rate_mat_each_res.items():
            print(name_each)
            print(rate_mat_each, end='\n\n')
        print("----------------------------------------------------------", end='\n\n')

        print("transition flow [cm^-3 s^-1] (transion rate [s^-1])")
        print("----------------------------------------------------------")
        
        # for detect major trasition(0)
        threshold_major_trans = 1e2
        convert_name = {
            "Photoionization": "pi",
            "Radiative Recombination": "rr",
            "Thermal Collision Bound-Free / Free-Bound": "cbffbt",
            "Radiative Bound-Bound": "rbbt",
            "Thermal Collision Bound-Bound": "cbbt",
            "Non-Thermal Collision Ionization":"ntci"
        }
        
        for i in range(self.dim_rate_mat):
            if self.setup.check_lte:
                print(f"flow (rate) : cbffbt + cbbt = total [major transition (threshold:{threshold_major_trans:.1e} times difference)]")
            else:
                if self.setup.without_thermal_col:
                    print(f"flow (rate) : pi + rr + rbbt + ntci = total [major transition (threshold:{threshold_major_trans:.1e} times difference)]")
                else:
                    print(f"flow (rate) : pi + rr + cbffbt + rbbt + cbbt + ntci = total [major transition (threshold:{threshold_major_trans:.1e} times difference)]")
            
            for j in range(self.dim_rate_mat):
                # for detect major trasition(1)
                rate_mat_each_ij = {}
                if i == j:
                    continue

                line_ji = f"{j} -> {i} : "
                for name_each, rate_mat_each in rate_mat_each_res.items():
                    # for detect major trasition(2)
                    rate_mat_each_ij[name_each] = rate_mat_each[i,j]
                    sorted_rate_mat_each_ij = dict(sorted(rate_mat_each_ij.items(), key=lambda item: item[1], reverse=True))

                    line_ji += f"{rate_mat_each[i,j]*self.nd_levels[j]:.3e} ({rate_mat_each[i,j]:.3e})"
                    line_ji += " + "
                line_ji = line_ji.rstrip(" + ")
                line_ji += f" = {rate_mat_res[i,j]*self.nd_levels[j]:.3e} ({rate_mat_res[i,j]:.3e})"

                val_sorted_rate_mat_each_ij = list(sorted_rate_mat_each_ij.values())
                key_sorted_rate_mat_each_ij = list(sorted_rate_mat_each_ij.keys())
                
                if val_sorted_rate_mat_each_ij[1] == 0.0:
                    line_ji += f" [{convert_name[key_sorted_rate_mat_each_ij[0]]}]"
                elif val_sorted_rate_mat_each_ij[0]/val_sorted_rate_mat_each_ij[1] >= threshold_major_trans:
                    line_ji += f" [{convert_name[key_sorted_rate_mat_each_ij[0]]}]"
                else:
                    line_ji += f" [{convert_name[key_sorted_rate_mat_each_ij[0]]}, {convert_name[key_sorted_rate_mat_each_ij[1]]}]"

                print(line_ji)
            
            print()

            for k in range(self.dim_rate_mat):
                # for detect major trasition(1)
                rate_mat_each_ki = {}
                if i == k:
                    continue
                line_ik = f"{i} -> {k} : "
                for name_each, rate_mat_each in rate_mat_each_res.items():
                    # for detect major trasition(2)
                    rate_mat_each_ki[name_each] = rate_mat_each[k,i]
                    sorted_rate_mat_each_ki = dict(sorted(rate_mat_each_ki.items(), key=lambda item: item[1], reverse=True))

                    line_ik += f"{rate_mat_each[k,i]*self.nd_levels[i]:.3e} ({rate_mat_each[k,i]:.3e})"
                    line_ik += " + "
                line_ik = line_ik.rstrip(" + ")
                line_ik += f" = {rate_mat_res[k,i]*self.nd_levels[i]:.3e} ({rate_mat_res[k,i]:.3e})"
                
                val_sorted_rate_mat_each_ki = list(sorted_rate_mat_each_ki.values())
                key_sorted_rate_mat_each_ki = list(sorted_rate_mat_each_ki.keys())
                
                if val_sorted_rate_mat_each_ki[1] == 0.0:
                    line_ik += f" [{convert_name[key_sorted_rate_mat_each_ki[0]]}]"
                elif val_sorted_rate_mat_each_ki[0]/val_sorted_rate_mat_each_ki[1] >= threshold_major_trans:
                    line_ik += f" [{convert_name[key_sorted_rate_mat_each_ki[0]]}]"
                else:
                    line_ik += f" [{convert_name[key_sorted_rate_mat_each_ki[0]]}, {convert_name[key_sorted_rate_mat_each_ki[1]]}]"

                print(line_ik)                    

            print()
            
            print("net flow [in: +, out: -]")
            netflow_i_seq = np.zeros(self.dim_rate_mat)
            for l in range(self.dim_rate_mat):
                if i == l:
                    continue
                line_il = f"{i} <-> {l} (+ <-> -) : "
                inflow_i = rate_mat_res[i,l]*self.nd_levels[l]
                outflow_i = rate_mat_res[l,i]*self.nd_levels[i]
                netflow_i = inflow_i - outflow_i
                netflow_i_seq[l] = netflow_i
                line_il += f" = {netflow_i:.3e}"
                                
                print(line_il)
            
            print(f"total : {netflow_i_seq.sum():.3e}")
            print(f"major net flows (top three)")
            sorted_arg = np.argsort(np.abs(netflow_i_seq))[::-1]
            for m in range(3):
                print(f"{i} <-> {sorted_arg[m]} (+ <-> -) : {netflow_i_seq[sorted_arg[m]]:.3e}")
            print()
     
    def compare_with_lte(self):
        fname_cbbt = Path(DATA_HE_DIR, "He_I_eie_ecs.csv")
        df_cbbt = pd.read_csv(fname_cbbt, comment='#')

        print("He level population [cm^-3]")
        for i in range(self.nd_levels.size):
            print(f"{i} {self.use_id["term"].iat[i]}:{self.nd_levels[i]:.3e}")
        print('\n'+f"electron number density = {self.nd_e:.3e} cm^-3", end='\n\n')

        print("log(n_u/n_l)")
        print("----------------------------------------------------------")
        print("transition : input - LTE (ratio of collisional transition rate) = departure from LTE")
        
        cbbt = ThermalCollisionBB(self.setup, self.ejecta).mk_rate_mat(self.nd_levels, self.nd_e)

        term_names = self.setup.use_id["term"].to_list()
        for _, lower, upper, E_ul, *_ in df_cbbt.itertuples():
            if not (lower in term_names and upper in term_names):
                continue

            idx_lower = term_names.index(lower)
            idx_upper = term_names.index(upper)
            
            g_l = self.setup.use_id.at[idx_lower, "g"]
            g_u = self.setup.use_id.at[idx_upper, "g"]

            n_l_input = self.nd_levels[idx_lower]
            n_u_input = self.nd_levels[idx_upper]

            log_r_input = np.log10(n_u_input/n_l_input)
            log_r_lte = np.log10(g_u/g_l) - np.log10(np.exp(1))*E_ul*EV/(K_B*self.ejecta.tem_e)

            r_bbt_rate = np.log10(cbbt[idx_upper, idx_lower]) - np.log10(cbbt[idx_lower, idx_upper])

            print(f"{idx_lower} -- {idx_upper} : {log_r_input:.3e} - {log_r_lte:.3e} ({r_bbt_rate:.3e}) = {log_r_input - log_r_lte:.3e}")
        
        print()
        print("log(n+_0 * nd_e / n_0)")
        print("----------------------------------------------------------")
        print("transition : input - LTE (ratio of collisional transition rate) = departure from LTE")
        
        ISsolver_He = IonStateSolver(atom_name='He', tem_e=self.ejecta.tem_e, w_geo=self.ejecta.w_geo)
        log_r_lte_first, log_r_lte_second = ISsolver_He.logSaha 

        if self.setup.solver_mode in ['seven', 'n3_Tarumi', 'all_Tarumi']:
            n_gr_input = self.nd_levels[[0,-2,-1]]
            log_r_input_first = np.log10(n_gr_input[1] * self.nd_e / n_gr_input[0])
            log_r_input_second = np.log10(n_gr_input[2] * self.nd_e / n_gr_input[1])

            cbffbt = ThermalCollisionBFFB(self.setup, self.ejecta).mk_rate_mat(self.nd_levels, self.nd_e)
            r_bffb_rate = np.log10(self.nd_e) + np.log10(cbffbt[-2, 0]) - np.log10(cbffbt[0, -2])

            print(f"{0} -- {self.nd_levels.size-2} : {log_r_input_first:.3e} - {log_r_lte_first:.3e} ({r_bffb_rate:.3e}) = {log_r_input_first - log_r_lte_first:.3e}")
            print(f"{0} -- {self.nd_levels.size-1} : {log_r_input_second:.3e} - {log_r_lte_second:.3e} = {log_r_input_second - log_r_lte_second:.3e}")

        else:
            n_gr_input = self.nd_levels[[0,-1]]
            log_r_input_first = np.log10(n_gr_input[1] * self.nd_e / n_gr_input[0])

            cbffbt = ThermalCollisionBFFB(self.setup, self.ejecta).mk_rate_mat(self.nd_levels, self.nd_e)
            r_bffb_rate = np.log10(self.nd_e) + np.log10(cbffbt[-1, 0]) - np.log10(cbffbt[0, -1])

            print(f"{0} -- {self.nd_levels.size-1} : {log_r_input_first:.3e} - {log_r_lte_first:.3e} ({r_bffb_rate:.3e}) = {log_r_input_first - log_r_lte_first:.3e}")

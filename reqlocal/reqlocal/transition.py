from dataclasses import dataclass
from abc import ABC, abstractmethod

from pathlib import Path
from functools import partial
from typing import Callable
import numpy.typing as npt

import numpy as np
from scipy import integrate
import pandas as pd

from reqlocal.constants import H, K_B, M_ELEC, EC, EV, RY
from reqlocal.base import DATA_HE_DIR, SolverParameter
from reqlocal.environment import EjectaParameter


@dataclass(frozen=True, slots=True)
class Transition(ABC):
    setup: SolverParameter

    @abstractmethod
    def mk_rate_mat(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        mat_trans = np.empty((self.setup.dim_rate_mat, self.setup.dim_rate_mat))
        
        return lambda nd_levels, nd_e: mat_trans

@dataclass(frozen=True, slots=True)
class PhotoIonization(Transition):
    ejecta: EjectaParameter

    name: str = "Photoionization"

    @property
    def mk_rate_mat(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        # photoionization
        mat_pi = self._get_pi_data()
        
        return lambda nd_levels, nd_e: mat_pi
    
    @property
    def mk_rate_mat_init(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        # photoionization
        mat_pi = self._get_pi_data()
        
        return lambda nd_e: mat_pi

    def _get_pi_data(self) -> npt.NDArray[np.float64]:
        data_pi = np.zeros((self.setup.dim_rate_mat, self.setup.dim_rate_mat))

        match self.ejecta.input_j_spec:
            case 'DiluteBlackBody':
                flux_ph: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = lambda freq_ph: 4.0*np.pi * self.ejecta.dbb_spec(freq_ph) / (H * freq_ph)
            case 'numerical':
                flux_ph: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = lambda freq_ph: 4.0*np.pi * self.ejecta.J_in_numerical(freq_ph) / (H * freq_ph)

        if self.setup.solver_mode in ['seven', 'n3_Tarumi','all_Tarumi']:
            e_level_I = self.setup.use_id["level (eV)"].iloc[:-2]
            e_ion_I = self.setup.use_id["level (eV)"].iat[-2] - e_level_I
            sup_rate_I = np.ones_like(e_ion_I)
            sup_rate_I[e_ion_I >= self.ejecta.uv_bk_prm.e_ph_thr] = self.ejecta.uv_bk_prm.r_uv

            for idx, term in enumerate(self.setup.use_id["term"][:-2]):
                e_ph, sig = read_sigma(term)
                freq_ph = e_ph / H
                 
                data_pi[-2, idx] = integrate.simpson(flux_ph(freq_ph)*sig, x=freq_ph) * 1e-18

            data_pi[-2, :-2] *= sup_rate_I

            e_level_II = self.setup.use_id["level (eV)"].iloc[-2]
            e_ion_II = self.setup.use_id["level (eV)"].iat[-1] - e_level_II
            sup_rate_II = 1.0
            if e_ion_II >= self.ejecta.uv_bk_prm.e_ph_thr:
                sup_rate_II = self.ejecta.uv_bk_prm.r_uv

            e_ph, sig = read_sigma_HeII()
            freq_ph = e_ph / H
                
            data_pi[-1, -2] = integrate.simpson(flux_ph(freq_ph)*sig, x=freq_ph) * 1e-18

            data_pi[-1, -2] *= sup_rate_II
        else:
            e_level_I = self.setup.use_id["level (eV)"].iloc[:-1]
            e_ion_I = self.setup.use_id["level (eV)"].iat[-1] - e_level_I
            sup_rate_I = np.ones_like(e_ion_I)
            sup_rate_I[e_ion_I >= self.ejecta.uv_bk_prm.e_ph_thr] = self.ejecta.uv_bk_prm.r_uv

            for idx, term in enumerate(self.setup.use_id["term"][:-1]):
                e_ph, sig = read_sigma(term)
                freq_ph = e_ph / H

                data_pi[-1, idx] = integrate.simpson(flux_ph(freq_ph)*sig, x=freq_ph) * 1e-18

            data_pi[-1, :-1] *= sup_rate_I
        
        return data_pi
    
def read_sigma(term: str):
    fname = Path(DATA_HE_DIR, "He_I_ppcs", f"{term}_ppcs.txt")
    
    df = pd.read_table(fname, sep=r'\s+', comment='#', header=None)
    e_ph = (df[0] * RY * EV).to_numpy(copy=True) # in erg
    sig = df[1].to_numpy(copy=True) # in Mb = 1e-18 cm^2 

    return e_ph, sig

def read_sigma_HeII():
    fname = Path(DATA_HE_DIR, "He_II_pcs", "ground_pcs.txt")

    df = pd.read_table(fname, sep=r'\s+', comment='#', header=None)
    e_ph = (df[0] * RY * EV).to_numpy(copy=True) # in erg
    sig = df[2].to_numpy(copy=True) # in Mb = 1e-18 cm^2 

    return e_ph, sig


@dataclass(frozen=True, slots=True)
class RadiativeRecombination(Transition):
    ejecta: EjectaParameter

    name: str = "Radiative Recombination"

    @property
    def mk_rate_mat(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        # radiative recombination
        mat_rr_spn = self._get_rr_spn_data()
        mat_rr_stm = self._get_rr_stm_data()
        
        return lambda nd_levels, nd_e: nd_e*(mat_rr_spn+mat_rr_stm)
    
    @property
    def mk_rate_mat_init(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        # radiative recombination
        mat_rr_spn = self._get_rr_spn_data()
        mat_rr_stm = self._get_rr_stm_data()
        
        return lambda nd_e: nd_e*(mat_rr_spn+mat_rr_stm)
    
    def _get_rr_spn_data(self) -> npt.NDArray[np.float64]:
        fname_rr = Path(DATA_HE_DIR, "He_ssrrc.csv")
        df_rr = pd.read_csv(fname_rr, comment='#')
        
        data_rr_spn = np.zeros((self.setup.dim_rate_mat, self.setup.dim_rate_mat))

        if self.setup.solver_mode in ['seven', 'n3_Tarumi','all_Tarumi']:
            if self.setup.norm_rr:
                rr_gr = df_rr["11S"]
                rr_tot = df_rr["HeI_total"]
                rr_use_tot = df_rr[self.setup.use_id["term"][1:-2]].sum(axis=1)
                norm_coeff = (rr_tot - rr_gr) / rr_use_tot
                data_rr_spn[1:-2, -2] = 10**(((df_rr[self.setup.use_id["term"][1:-2]].apply(lambda x: x*norm_coeff)).apply(np.log10)).apply(partial(np.interp, np.log10(self.ejecta.tem_e), df_rr["log10(T/[K])"])))
                data_rr_spn[0, -2] = 10**(np.interp(np.log10(self.ejecta.tem_e), df_rr["log10(T/[K])"], np.log10(rr_gr)))
            else:
                data_rr_spn[:-2, -2] = 10**((df_rr[self.setup.use_id["term"][:-2]].apply(np.log10)).apply(partial(np.interp, np.log10(self.ejecta.tem_e), df_rr["log10(T/[K])"])))
            data_rr_spn[-2, -1] = 10**(np.interp(np.log10(self.ejecta.tem_e), df_rr["log10(T/[K])"], np.log10(df_rr["HeII_total"])))
            data_rr_spn[0, -2] *= self.ejecta.pr_esc_rr_gr
        else:
            if self.setup.norm_rr:
                rr_gr = df_rr["11S"]
                rr_tot = df_rr["HeI_total"]
                rr_use_tot = df_rr[self.setup.use_id["term"][1:-1]].sum(axis=1)
                norm_coeff = (rr_tot - rr_gr) / rr_use_tot
                data_rr_spn[1:-1, -1] = 10**(((df_rr[self.setup.use_id["term"][1:-1]].apply(lambda x: x*norm_coeff)).apply(np.log10)).apply(partial(np.interp, np.log10(self.ejecta.tem_e), df_rr["log10(T/[K])"])))
                data_rr_spn[0, -1] = 10**(np.interp(np.log10(self.ejecta.tem_e), df_rr["log10(T/[K])"], np.log10(rr_gr)))
            else:
                data_rr_spn[:-1, -1] = 10**((df_rr[self.setup.use_id["term"][:-1]].apply(np.log10)).apply(partial(np.interp, np.log10(self.ejecta.tem_e), df_rr["log10(T/[K])"])))
                data_rr_spn[0, -1] *= self.ejecta.pr_esc_rr_gr
        
        return data_rr_spn
    
    def get_rr_spn_tot(self) -> dict[str, float]:
        fname_rr = Path(DATA_HE_DIR, "He_ssrrc.csv")
        df_rr = pd.read_csv(fname_rr, comment='#')
        ds_rr_gr = df_rr["11S"]
        ds_rr_tot_I = df_rr["HeI_total"]
        ds_rr_tot_II = df_rr["HeII_total"]

        log_rr_gr = np.interp(np.log10(self.ejecta.tem_e), df_rr["log10(T/[K])"], np.log10(ds_rr_gr))
        log_rr_tot_I = np.interp(np.log10(self.ejecta.tem_e), df_rr["log10(T/[K])"], np.log10(ds_rr_tot_I))
        log_rr_tot_II = np.interp(np.log10(self.ejecta.tem_e), df_rr["log10(T/[K])"], np.log10(ds_rr_tot_II))

        rr_spn = {
            "to_HeI_gr_actual": 10**log_rr_gr,
            "to_HeI_tot_actual": 10**log_rr_tot_I,
            "to_HeII_tot_actual": 10**log_rr_tot_II
        }

        mat_rr_spn = self._get_rr_spn_data()
        rr_spn["to_HeI_gr_model"] = mat_rr_spn[0, -2]

        if self.setup.solver_mode in ['seven', 'n3_Tarumi','all_Tarumi']:
            rr_spn["to_HeI_tot_model"] = mat_rr_spn[:-2, -2].sum()
            rr_spn["to_HeII_tot_model"] = mat_rr_spn[-2, -1]
        else:
            rr_spn["to_HeI_tot_model"] = mat_rr_spn[:-1, -1].sum()
            rr_spn["to_HeII_tot_model"] = 0.0

        return rr_spn

    def _get_rr_stm_data(self) -> npt.NDArray[np.float64]:
        data_rr_stm = np.zeros((self.setup.dim_rate_mat, self.setup.dim_rate_mat))

        match self.ejecta.input_j_spec:
            case 'DiluteBlackBody':
                flux_ph: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = lambda freq_ph: 4.0*np.pi * self.ejecta.dbb_spec(freq_ph) / (H * freq_ph)
            case 'numerical':
                flux_ph: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = lambda freq_ph: 4.0*np.pi * self.ejecta.J_in_numerical(freq_ph) / (H * freq_ph)
        
        # (h^2/(2\pi m_e k T_e))^(3/2)*(1/g_e)
        coeff_const = 0.5*(H**2/(2.0*np.pi*M_ELEC*K_B*self.ejecta.tem_e))**1.5

        if self.setup.solver_mode in ['seven', 'n3_Tarumi','all_Tarumi']:
            e_level_I = self.setup.use_id["level (eV)"].iloc[:-2].to_numpy(copy=True) * EV
            e_level_II = self.setup.use_id["level (eV)"].iat[-2] * EV
            e_level_III = self.setup.use_id["level (eV)"].iat[-1] * EV
            e_ion_I = e_level_II - e_level_I
            e_ion_II = e_level_III - e_level_II
            g_I = self.setup.use_id["g"].iloc[:-2].to_numpy(copy=True)
            g_II = self.setup.use_id["g"].iat[-2]
            g_III = self.setup.use_id["g"].iat[-1]
            sup_rate_I = np.ones_like(e_ion_I)
            sup_rate_I[e_ion_I >= self.ejecta.uv_bk_prm.e_ph_thr] = self.ejecta.uv_bk_prm.r_uv

            for idx, term in enumerate(self.setup.use_id["term"][:-2]):
                e_ph, sig = read_sigma(term)
                freq_ph = e_ph / H

                sig *= coeff_const * (g_I[idx]/g_II) * np.exp(-(e_ph-e_ion_I[idx])/(K_B*self.ejecta.tem_e))

                data_rr_stm[idx, -2] = integrate.simpson(flux_ph(freq_ph)*sig, x=freq_ph) * 1e-18

            data_rr_stm[:-2, -2] *= sup_rate_I

            sup_rate_II = 1.0
            if e_ion_II >= self.ejecta.uv_bk_prm.e_ph_thr:
                sup_rate_II = self.ejecta.uv_bk_prm.r_uv

            e_ph, sig = read_sigma_HeII()
            freq_ph = e_ph / H
            sig *= coeff_const * (g_II/g_III) * np.exp(-(e_ph-e_ion_II)/(K_B*self.ejecta.tem_e))
                
            data_rr_stm[-2, -1] = integrate.simpson(flux_ph(freq_ph)*sig, x=freq_ph) * 1e-18

            data_rr_stm[-2, -1] *= sup_rate_II
        else:
            e_level_I = self.setup.use_id["level (eV)"].iloc[:-1].to_numpy(copy=True) * EV
            e_level_II = self.setup.use_id["level (eV)"].iat[-1] * EV
            e_ion_I = e_level_II - e_level_I
            g_I = (self.setup.use_id["g"].iloc[:-1]).to_numpy(copy=True)
            g_II = self.setup.use_id["g"].iat[-1]
            sup_rate_I = np.ones_like(e_ion_I)
            sup_rate_I[e_ion_I >= self.ejecta.uv_bk_prm.e_ph_thr] = self.ejecta.uv_bk_prm.r_uv

            for idx, term in enumerate(self.setup.use_id["term"][:-1]):
                e_ph, sig = read_sigma(term)
                freq_ph = e_ph / H

                sig *= coeff_const * (g_I[idx]/g_II) * np.exp(-(e_ph-e_ion_I[idx])/(K_B*self.ejecta.tem_e))

                data_rr_stm[idx, -1] = integrate.simpson(flux_ph(freq_ph)*sig, x=freq_ph) * 1e-18

            data_rr_stm[:-1, -1] *= sup_rate_I
        
        return data_rr_stm


@dataclass(frozen=True, slots=True)
class ThermalCollisionBFFB(Transition):
    ejecta: EjectaParameter

    name: str = "Thermal Collision Bound-Free / Free-Bound"

    @property
    def mk_rate_mat(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        # electron-impact ionization / three-body recombination
        mat_eii, mat_tbr = self._get_cbffbt_data()

        EII_CONST = 1.15506e-15
        mat_eii *= EII_CONST / np.sqrt(self.ejecta.tem_e)
        TBR_CONST = 2.39174e-31
        mat_tbr *= TBR_CONST / (self.ejecta.tem_e**2)
        

        return lambda nd_levels, nd_e: nd_e*mat_eii + nd_e**2*mat_tbr
    
    @property
    def mk_rate_mat_init(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        # electron-impact ionization / three-body recombination
        mat_eii, mat_tbr = self._get_cbffbt_data()

        EII_CONST = 1.15506e-15
        mat_eii *= EII_CONST / np.sqrt(self.ejecta.tem_e)
        TBR_CONST = 2.39174e-31
        mat_tbr *= TBR_CONST / (self.ejecta.tem_e**2)
        

        return lambda nd_e: nd_e*mat_eii + nd_e**2*mat_tbr
    
    def _get_cbffbt_data(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # collisional bound-free / free-bound transion
        fname_cbffbt = Path(DATA_HE_DIR, "He_I_eii_ecs.csv")
        df_cbffbt = pd.read_csv(fname_cbffbt, comment='#')

        data_eii = np.zeros((self.setup.dim_rate_mat, self.setup.dim_rate_mat))
        data_tbr = np.zeros((self.setup.dim_rate_mat, self.setup.dim_rate_mat))

        if self.setup.solver_mode in ['seven', 'n3_Tarumi','all_Tarumi']:
            col_strength = 10**((df_cbffbt[self.setup.use_id["term"][:-2]].apply(np.log10)).apply(partial(np.interp, np.log10(self.ejecta.tem_e), df_cbffbt["log10(T/[K])"])))
            e_level_I = self.setup.use_id["level (eV)"].iloc[:-2].to_numpy(copy=True) * EV
            e_level_II = self.setup.use_id["level (eV)"].iat[-2] * EV
            e_ion_I = e_level_II - e_level_I
            g_I = self.setup.use_id["g"].iloc[:-2].to_numpy(copy=True)
            g_II = self.setup.use_id["g"].iat[-2]
            data_eii[-2, :-2] = col_strength.to_numpy(copy=True) * np.exp(-e_ion_I / (K_B*self.ejecta.tem_e)) / e_ion_I
            data_tbr[:-2, -2] = col_strength.to_numpy(copy=True) * (g_I/g_II) * np.exp(-e_level_I / (K_B*self.ejecta.tem_e)) / e_ion_I
        else:
            col_strength = 10**((df_cbffbt[self.setup.use_id["term"][:-1]].apply(np.log10)).apply(partial(np.interp, np.log10(self.ejecta.tem_e), df_cbffbt["log10(T/[K])"])))
            e_level_I = self.setup.use_id["level (eV)"].iloc[:-1].to_numpy(copy=True) * EV
            e_level_II = self.setup.use_id["level (eV)"].iat[-1] * EV
            e_ion_I = e_level_II - e_level_I
            g_I = self.setup.use_id["g"].iloc[:-1].to_numpy(copy=True)
            g_II = self.setup.use_id["g"].iat[-1]
            data_eii[-1, :-1] = col_strength.to_numpy(copy=True) * np.exp(- e_ion_I / (K_B*self.ejecta.tem_e)) / e_ion_I
            data_tbr[:-1, -1] = col_strength.to_numpy(copy=True) * (g_I/g_II) * np.exp(- e_level_I / (K_B*self.ejecta.tem_e)) / e_ion_I

        return data_eii, data_tbr

@dataclass(frozen=True, slots=True)
class RadiativeBB(Transition):
    ejecta: EjectaParameter

    name: str = "Radiative Bound-Bound"

    @property
    def mk_rate_mat(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        mat_rbbt = self._get_rbbt_data()

        if self.setup.use_Sobolev_method:
            pr_esc_mat = partial(mk_pr_esc_mat, dim_mat=self.setup.dim_rate_mat, use_id=self.setup.use_id, t_expl=self.ejecta.t_expl)
            return lambda nd_levels, nd_e: mat_rbbt*pr_esc_mat(nd_levels)
        else:
            return lambda nd_levels, nd_e: mat_rbbt
    
    @property
    def mk_rate_mat_init(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        mat_rbbt = self._get_rbbt_data()

        return lambda nd_e: mat_rbbt
    
    def _get_rbbt_data(self) -> npt.NDArray[np.float64]:
        # radiative bound-bound transition
        fname_rbbt = Path(DATA_HE_DIR, "He_I_rbbt_params.csv")
        df_rbbt = pd.read_csv(fname_rbbt, comment='#')

        data_rbbt = np.zeros((self.setup.dim_rate_mat, self.setup.dim_rate_mat))

        term_names = self.setup.use_id["term"].to_list()

        match self.ejecta.input_j_spec:
            case 'DiluteBlackBody':
                j_in: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = lambda freq_ph: self.ejecta.dbb_spec(freq_ph)
            case 'numerical':
                j_in: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = lambda freq_ph: self.ejecta.J_in_numerical(freq_ph)

        for _, lower, upper, *values in df_rbbt.itertuples():
            if not (lower in term_names and upper in term_names):
                continue
            
            idx_lower = term_names.index(lower)
            idx_upper = term_names.index(upper)
            
            g_l = self.setup.use_id.at[idx_lower, "g"]
            g_u = self.setup.use_id.at[idx_upper, "g"]
            E_ul, A_ul, _, B_lu = values
            freq_ph = E_ul * EV / H

            if self.ejecta.uv_bk_prm.photoion_only:
                r_uv = 1.0
            else:
                if E_ul >= self.ejecta.uv_bk_prm.e_ph_thr:
                    r_uv = self.ejecta.uv_bk_prm.r_uv
                else:
                    r_uv = 1.0

            data_rbbt[idx_lower, idx_upper] = A_ul + (g_l/g_u)*B_lu * r_uv*j_in(freq_ph)
            data_rbbt[idx_upper, idx_lower] = B_lu * r_uv*j_in(freq_ph)
    
        return data_rbbt
    
def mk_pr_esc_mat(nd_levels: npt.NDArray[np.float64], dim_mat: int, use_id: pd.DataFrame, t_expl: float) -> npt.NDArray[np.float64]:
    fname_rbbt = Path(DATA_HE_DIR, "He_I_rbbt_params.csv")
    df_rbbt = pd.read_csv(fname_rbbt, comment='#')

    mat_pr_esc = np.zeros((dim_mat, dim_mat))

    term_names = use_id["term"].to_list()

    for _, lower, upper, *values in df_rbbt.itertuples():
        if not (lower in term_names and upper in term_names):
            continue
        
        idx_lower = term_names.index(lower)
        idx_upper = term_names.index(upper)
        
        g_l = use_id.at[idx_lower, "g"]
        g_u = use_id.at[idx_upper, "g"]
        E_ul, _, f_lu, _ = values

        tau_l = (np.pi*EC*EC*H/M_ELEC) * f_lu * (nd_levels[idx_lower] - (g_l/g_u) * nd_levels[idx_upper]) * t_expl / (E_ul*EV)
        if tau_l > 0.01:
            pr_esc = (1.0-np.exp(-tau_l))/tau_l
        else:
            pr_esc = 1.0 + 0.5 * tau_l

        mat_pr_esc[idx_lower, idx_upper] = pr_esc
        mat_pr_esc[idx_upper, idx_lower] = pr_esc

    return mat_pr_esc

@dataclass(frozen=True, slots=True)
class ThermalCollisionBB(Transition):
    ejecta: EjectaParameter


    name: str = "Thermal Collision Bound-Bound"

    @property
    def mk_rate_mat(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        mat_cbbt = self._get_cbbt_data()
        
        CBBT_CONST = 8.629e-6
        mat_cbbt *= CBBT_CONST / np.sqrt(self.ejecta.tem_e)

        return lambda nd_levels, nd_e: nd_e*mat_cbbt
    
    @property
    def mk_rate_mat_init(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        mat_cbbt = self._get_cbbt_data()
        
        CBBT_CONST = 8.629e-6
        mat_cbbt *= CBBT_CONST / np.sqrt(self.ejecta.tem_e)

        return lambda nd_e: nd_e*mat_cbbt

    def _get_cbbt_data(self) -> npt.NDArray[np.float64]:
        # collisional bound-bound transition
        match self.setup.cbbt_data:
            case "normal":
                fname_cbbt = Path(DATA_HE_DIR, "He_I_eie_ecs.csv")
            case "modified":
                fname_cbbt = Path(DATA_HE_DIR, "He_I_eie_ecs_modified.csv")
        
        df_cbbt = pd.read_csv(fname_cbbt, comment='#')

        data_cbbt = np.zeros((self.setup.dim_rate_mat, self.setup.dim_rate_mat))

        term_names = self.setup.use_id["term"].to_list()

        for _, lower, upper, E_ul, *values in df_cbbt.itertuples():
            if not (lower in term_names and upper in term_names):
                continue

            idx_lower = term_names.index(lower)
            idx_upper = term_names.index(upper)
            
            g_l = self.setup.use_id.at[idx_lower, "g"]
            g_u = self.setup.use_id.at[idx_upper, "g"]

            col_strength = 10**(np.interp(np.log10(self.ejecta.tem_e), df_cbbt.columns[3:].to_numpy(float, copy=True), np.log10(values)))
            
            data_cbbt[idx_lower, idx_upper] = col_strength / g_u
            data_cbbt[idx_upper, idx_lower] = col_strength * np.exp(- E_ul*EV / (K_B*self.ejecta.tem_e)) / g_l

        return data_cbbt
    
@dataclass(frozen=True, slots=True)
class NonThermalCollisionIonization:
    setup: SolverParameter
    ejecta: EjectaParameter

    name: str = "Non-Thermal Collision Ionization"

    @property
    def mk_rate_mat(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        mat_ntci = partial(self._get_ntci_data)
        
        return lambda nd_levels, nd_e: mat_ntci(nd_levels, nd_e)
    
    @property
    def mk_rate_mat_init(self) -> Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]:
        mat_ntci = np.zeros((self.setup.dim_rate_mat, self.setup.dim_rate_mat))
        
        # Tarumi+ 2023
        wpi_I = self.setup.wpi_He_ev[0]*EV
        wpi_II = self.setup.wpi_He_ev[1]*EV

        if self.setup.solver_mode in ['seven', 'n3_Tarumi', 'all_Tarumi']:
            mat_ntci[-2, 0] =  self.ejecta.r_heat / wpi_I
            mat_ntci[-1, -2] = self.ejecta.r_heat / wpi_II
        else:
            mat_ntci[-1, 0] =  self.ejecta.r_heat / wpi_I
        
        return lambda nd_e: mat_ntci
    
    def _get_ntci_data(self, nd_levels: npt.NDArray[np.float64], nd_e: float) -> npt.NDArray[np.float64]:
        data_ntci = np.zeros((self.setup.dim_rate_mat, self.setup.dim_rate_mat))

        # Tarumi+ 2023
        wpi_I = self.setup.wpi_He_ev[0]*EV
        wpi_II = self.setup.wpi_He_ev[1]*EV

        if self.setup.solver_mode in ['seven', 'n3_Tarumi', 'all_Tarumi']:
            if self.setup.use_Hotokezaka_SF_res:
                data_ntci[-2, 0] =  self.ejecta.r_heat / wpi_I
                data_ntci[-1, -2] = self.ejecta.r_heat / wpi_II
            else:
                e_ion_I = self.setup.use_id["level (eV)"].iat[-2] * EV
                e_ion_II = self.setup.use_id["level (eV)"].iat[-1] * EV

                x_e_He = (nd_levels[-2]+2*nd_levels[-1])/self.ejecta.nd_He
                dfi_I = dep_frac_ion(x_e_He, 'I')
                dfi_II = dep_frac_ion(x_e_He, 'II') 

                data_ntci[-2, 0] =  dfi_I * self.ejecta.r_heat / e_ion_I
                data_ntci[-1, -2] = dfi_II * self.ejecta.r_heat / e_ion_II
        else:
            if self.setup.use_Hotokezaka_SF_res:
                data_ntci[-1, 0] =  self.ejecta.r_heat / wpi_I
            else:
                e_ion_I = self.setup.use_id["level (eV)"].iat[-1] * EV

                x_e_He = nd_levels[-1]/self.ejecta.nd_He
                dfi_I = dep_frac_ion(x_e_He, 'I')

                data_ntci[-1, 0] =  dfi_I * self.ejecta.r_heat / e_ion_I

        return data_ntci
    

def dep_frac_ion(x_e: float | npt.NDArray[np.float64], ion_from: str = 'I') -> float | npt.NDArray[np.float64]:
    fname_sf_res = Path(DATA_HE_DIR, "He_energy_fraction_wide_range.csv")
    df_sf_res = pd.read_csv(fname_sf_res, comment='#')

    x_e_min = df_sf_res["x_e"].iat[0]
    x_e_max = df_sf_res["x_e"].iat[-1]

    if ion_from == 'I':
        dfi_at_x_e_min = df_sf_res["He I ionization"].iat[0]
        dfi_at_x_e_max = df_sf_res["He I ionization"].iat[-1]

        if isinstance(x_e, np.ndarray):
            dfi = np.interp(np.log10(x_e), np.log10(df_sf_res["x_e"]), df_sf_res["He I ionization"])
            dfi = np.where(x_e < x_e_min, dfi_at_x_e_min, dfi)
            dfi = np.where(x_e > x_e_max, dfi_at_x_e_max, dfi)
        else:
            if x_e < x_e_min:
                dfi = dfi_at_x_e_min
            elif x_e > x_e_max:
                dfi = dfi_at_x_e_max
            else:
                dfi = np.interp(np.log10(x_e), np.log10(df_sf_res["x_e"]), df_sf_res["He I ionization"])
    elif ion_from == 'II':
        dfi_at_x_e_min = df_sf_res["He II ionization"].iat[0]
        dfi_at_x_e_max = df_sf_res["He II ionization"].iat[-1]

        if isinstance(x_e, np.ndarray):
            dfi = np.interp(np.log10(x_e), np.log10(df_sf_res["x_e"]), df_sf_res["He II ionization"])
            dfi = np.where(x_e < x_e_min, dfi_at_x_e_min, dfi)
            dfi = np.where(x_e > x_e_max, dfi_at_x_e_max, dfi)
        else:
            if x_e < x_e_min:
                dfi = dfi_at_x_e_min
            elif x_e > x_e_max:
                dfi = dfi_at_x_e_max
            else:
                dfi = np.interp(np.log10(x_e), np.log10(df_sf_res["x_e"]), df_sf_res["He II ionization"])
    else:
        raise ValueError("You should choose ion_from from 'I' or 'II'.")

    return dfi        

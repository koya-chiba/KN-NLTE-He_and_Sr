from pathlib import Path
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from reqlocal.constants import H, K_B, M_ELEC, EV
from reqlocal import DATA_H_DIR, DATA_HE_DIR, DATA_SR_DIR


class IonStateSolver:
    def __init__(self, atom_name: str = "Sr", SrIonEnergy_data: str = "Tarumi23", 
                 nd_e: float=1e7, tem_e: float=5e3, w_geo: float=1.0, log_frac_I: float = -20, 
                 qdot_nt: float | None = None, flag_nt: bool = False):
        
        if (qdot_nt is None) & flag_nt:
            raise ValueError("If 'flag_nt' is True, you should set 'qdot_nt'")
        
        self.atom_name = atom_name
        self.nd_e = nd_e
        self.tem_e = tem_e
        self.w_geo = w_geo
        self.log_frac_I = log_frac_I
        self.qdot_nt = qdot_nt
        self.flag_nt = flag_nt

        match self.atom_name:
            case "He":
                fpath_ion = Path(DATA_HE_DIR, "He_IonEnergy.csv")

            case "Sr":
                match SrIonEnergy_data:
                    case "Tarumi23":
                        fpath_ion = Path(DATA_SR_DIR, "Sr_IonEnergy.csv")
                    case "extended":
                        fpath_ion = Path(DATA_SR_DIR, "Sr_IonEnergy_v2.csv")
                    case _:
                        raise ValueError("You should choose 'Tarumi23' or 'extended' as 'SrIonEnergy_data'.")

            case _:
                raise ValueError("You should choose 'He' or 'Sr' as 'atom_name'.")
 
        # read IonEnergy -----------------------------------
        df_ion = pd.read_csv(fpath_ion, comment='#')
        self.symbol = df_ion["ion"].to_numpy(dtype=str)
        self.g0 = df_ion["g_ground"].to_numpy(dtype=float)
        self.Epot = df_ion["ionization energy(eV)"].to_numpy(dtype=float)[:-1]
        self.wpi = df_ion["wpi(eV)"].to_numpy(dtype=float)[:-1] * EV # in erg
        # ---------------------------------------------------
        
        self.log_ratio_next = np.zeros_like(self.logSaha, dtype=float)
        self.frac_ion = np.zeros_like(self.symbol, dtype=float)
        self.ion_deg_seq = np.arange(self.frac_ion.size)+1
        self.average_ion_deg: float | None = None
        self.frac_e: float | None = None
        self._solved = False
        
    def reset(self):
        self._solved = False
        if hasattr(self, "log_ratio_next"):
            self.log_ratio_next = np.zeros_like(self.logSaha, dtype=float)
            self.frac_ion = np.zeros_like(self.symbol, dtype=float)
            self.ion_deg_seq = np.arange(self.frac_ion.size)+1
            self.average_ion_deg: float | None = None
            self.frac_e: float | None = None

    @property
    def logSaha(self):
        return self.get_logSaha()
    
    def get_logSaha(self):
        SAHA_CONST_A = 3.0*np.log10(H/np.sqrt(2.0*np.pi*M_ELEC*K_B))
        SAHA_CONST_B = np.log10(np.exp(1))*EV/K_B
        logSaha = np.log10(2.0*self.g0[1:]/self.g0[:-1]) - SAHA_CONST_A  + 1.5*np.log10(self.tem_e) - SAHA_CONST_B*self.Epot/self.tem_e

        return logSaha

    @property
    def rrc(self):
        return self.get_rrc()

    def get_rrc(self):
        match self.atom_name:
            case "He":
                # read Helium recombination rate ------------------
                fpath_rrc = Path(DATA_HE_DIR, "He_ssrrc.csv")
                df_He_rrc = pd.read_csv(fpath_rrc, comment='#')
                log_tem_meta = df_He_rrc["log10(T/[K])"].to_numpy(dtype=float)
                log_rrc_I_meta = np.log10(df_He_rrc["HeI_total"].to_numpy(dtype=float))
                log_rrc_II_meta = np.log10(df_He_rrc["HeII_total"].to_numpy(dtype=float))
                # ---------------------------------------------------
                log_rrc_I_interp = np.interp(np.log10(self.tem_e), log_tem_meta, log_rrc_I_meta)
                log_rrc_II_interp = np.interp(np.log10(self.tem_e), log_tem_meta, log_rrc_II_meta)
                rrc = 10**np.array([log_rrc_I_interp, log_rrc_II_interp])
            case "Sr":
                # read Hydrogen recombination rate ------------------
                fpath_rrc = Path(DATA_H_DIR, "H_I_rrc_tot.tsv")
                df_H_rrc = pd.read_table(fpath_rrc, comment='#', sep='\s+')
                log_tem_meta = df_H_rrc["log(T)"].to_numpy(dtype=float)
                log_rrc_meta = np.log10(df_H_rrc["rrc(ion)"].to_numpy(dtype=float))
                # ---------------------------------------------------
                log_tem_scaled = np.log10(self.tem_e/np.square(self.ion_deg_seq[:-1]))
                log_rrc_interp = np.interp(log_tem_scaled, log_tem_meta, log_rrc_meta)
                log_rrc_interp = np.array(log_rrc_interp)
                rrc = self.ion_deg_seq[:-1] * 10**log_rrc_interp
            case _:
                raise ValueError("You should choose 'He' or 'Sr' as 'atom_name'.")
 
        return rrc
    
    @property
    def nd_e(self):
        return self.__nd_e 
    
    @nd_e.setter
    def nd_e(self, nd_e):
        if nd_e <= 0.0:
            raise ValueError("'nd_e' should be positive.")
        
        self.reset()
        self.__nd_e = nd_e
    
    @property
    def tem_e(self):
        return self.__tem_e
    
    @tem_e.setter
    def tem_e(self, tem_e):
        if tem_e <= 0.0:
            raise ValueError("'tem_e' should be positive.")
        
        self.reset()
        self.__tem_e = tem_e
    
    @property
    def w_geo(self):
        return self.__w_geo
    
    @w_geo.setter
    def w_geo(self, w_geo):
        if (self.atom_name == "Sr") & (w_geo == 0.0):
            raise ValueError("If 'atom_name' is 'Sr', 'w_geo' should not be equal to 0.0.")
        if (w_geo < 0.0) | (w_geo > 1.0):
            raise ValueError("'w_geo' should be equal to or between 0.0 and 1.0.")
        
        self.reset()
        self.__w_geo = w_geo

    @property
    def log_frac_I(self):
        return self.__log_frac_I
    
    @log_frac_I.setter
    def log_frac_I(self, log_frac_I):
        if  log_frac_I > 0.0:
            raise ValueError("'log_frac_I' should be negative.")
        
        self.reset()
        self.__log_frac_I = log_frac_I
    
    @property
    def qdot_nt(self):
        return self.__qdot_nt
    
    @qdot_nt.setter
    def qdot_nt(self, qdot_nt):
        if qdot_nt is None:
            self.flag_nt = False
        else:
            self.flag_nt = True
            if  qdot_nt <= 0.0:
                raise ValueError("'qdot_nt' should be positive.")
                        
        self.reset()
        self.__qdot_nt = qdot_nt

    @property
    def flag_nt(self):
        return self.__flag_nt
    
    @flag_nt.setter
    def flag_nt(self, flag_nt):
        if type(flag_nt) is not bool:
            raise ValueError("'flag_nt' should be boolean.")
        
        self.reset()
        self.__flag_nt = flag_nt

    def solve(self):
        assert not self._solved, "Saha equation is already solved."

        if self.flag_nt:
            self.log_ratio_next[:] = np.log10(self.w_geo*(10**self.logSaha) + self.qdot_nt/(self.wpi*self.rrc))
        else:
            self.log_ratio_next[:] = np.log10(self.w_geo) + self.logSaha
        
        self.log_ratio_next[:] -= np.log10(self.nd_e)

        self.frac_ion[0] = 10**self.log_frac_I
        self.frac_ion[1:] = 10**(self.log_frac_I + np.cumsum(self.log_ratio_next))

        self.frac_ion /= self.frac_ion.sum()
        self.average_ion_deg = np.sum(self.ion_deg_seq*self.frac_ion)
        self.frac_e = self.average_ion_deg - 1

        self._solved = True

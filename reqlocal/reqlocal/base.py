from dataclasses import dataclass

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_H_DIR = DATA_DIR / "H"
DATA_HE_DIR = DATA_DIR / "He"
DATA_SR_DIR = DATA_DIR / "Sr"


@dataclass(frozen=True, slots=True)
class UVBlanketingParameter:
    """
    UV line blanketing parameter

    Parameters
    ----------
    e_ph_thr : float
        Threshold photon energy for UV line blanketing in eV
    r_uv : float
        uv photon rate compared with blackbody radiation
    photoion_only : bool
        choose UV blanketing target photons, photoion only or photoion and bound-bound photon
    """
    e_ph_thr: float
    r_uv: float
    photoion_only: bool

    def __post_init__(self):
        if (self.r_uv < 0.0):
            raise ValueError("'r_uv' should be positive")

@dataclass(frozen=True, slots=True)
class SolverParameter:
    """
    Parameters required for calculation

    Parameters
    ----------
    solver_mode : {'three_1', 'three_3', 'four', 'six', 'all_Lucy', 'all_Tarumi', str}
        Select the solver mode to determine levels considered.
        - 'three_1' : He I ground and 21P states, and He II ground state
        - 'three_3' : He I ground and 23P states, and He II ground state
        - 'four': He I ground, 21P and 23P states, and He II ground state
        - 'six' :  He I ground, 21S, 21P, 23S and 23P states, and He II ground state
        - 'seven' : He I ground, 21S, 21P, 23S and 23P states, and He II / III ground state
        - 'n3_Lucy' : He I states up to n=3, and He II ground state
        - 'n3_Tarumi' : He I states up to n=3, and He II / III ground state
        - 'all_Lucy' : He I 19 bound states and He II ground state
        - 'all_Tarumi' :  He I 19 bound states, He II ground state and He III ground state
    
    norm_rr : bool
        whether to normalize (spontaneous) radiative recombination rate, default False

    use_Sobolev_method : bool
        whether to use Sobolev method for radiative bound-bound, default True

    use_Hotokezaka_SF_res : bool
        whether to use work per ion pair for helium derived by Spencer-Fano equation 
        in kilonova ejecta. Values are taken from Tarumi+ (2023). Original data is in Hotokezaka+ in prep.
    
    wpi_He_ev : tuple[float, float]
        If 'use_Hotokezaka_SF_res' = True, This value is used as work per ion pair for helium in eV.
        First is for He I and second is for He II

    cbbt_data : {'normal', 'modified', str}
        determine which data for collisional bound-bound transition data, default 'modified'.
        note that 'modified' data is better in the sense of the accuracy. 

    SrIonEnergy_data : {'Tarumi23', 'extended', str}
        determine which data for ionization energies of Sr, default 'Tarumi23'.

    without_thermal_col : bool
        whether to exclude thermal collisinal transition
        
    check_lte : bool
        To check if solver can reproduce LTE distribution when we consider only collisional transitions, default False
        
    xtol : float, optional
        used in scipy.optimize.root(method='hybr')
        relative error between two consecutive iterates is at most xtol.
    
    log_tem_range : tuple[float, float]
        The range of logarithmic temperatures for which transition coefficient data are available
    """
    solver_mode: str

    # to compare Albert's code
    norm_rr: bool = False
    use_Sobolev_method: bool = True
    use_Hotokezaka_SF_res: bool = False

    # work per ion pair for He
    fpath_ion = Path(DATA_HE_DIR, "He_IonEnergy.csv")
    df_ion = pd.read_csv(fpath_ion, comment='#')
    wpi_He_ev: tuple[float, float] = tuple(df_ion["wpi(eV)"].to_numpy(dtype=float)[:-1])

    cbbt_data: str = "modified"

    SrIonEnergy_data: str = "Tarumi23"
    
    without_thermal_col: bool = False
    check_lte: bool = False
    xtol: float | None = None
    log_tem_range: tuple[float, float] = (3.3, 4.3)

    def __post_init__(self):                                    
        match self.cbbt_data:
            case 'normal':
                pass
            case 'modified':
                pass
            case _:
                raise ValueError("You should choose 'cbbt_data' from 'normal' or 'modified'.")
        
        match self.SrIonEnergy_data:
            case 'Tarumi23':
                pass
            case 'extended':
                pass
            case _:
                raise ValueError("You should choose 'SrIonEnergy_data' from 'Tarumi23' or 'extended'.")
            
        if self.check_lte & self.without_thermal_col:
                raise ValueError("Either 'check_lte' or 'without_thermal_col' can be True.")

    @property
    def use_levels(self) -> tuple[str, ...]:
        match self.solver_mode:
            case 'three_1':
                return ("11S", "21P", "II")
            case 'three_3':
                return ("11S", "23P", "II")
            case 'four':
                return ("11S", "21P", "23P", "II")
            case 'six':
                return ("11S", "21S", "21P", "23S", "23P", "II")
            case 'seven':
                return ("11S", "21S", "21P", "23S", "23P", "II", "III")
            case 'n3_Lucy':
                return ("11S",
                        "23S", "21S", "23P", "21P",
                        "33S", "31S", "33P", "33D", "31D", "31P",
                        "II")
            case 'n3_Tarumi':
                return ("11S",
                        "23S", "21S", "23P", "21P",
                        "33S", "31S", "33P", "33D", "31D", "31P",
                        "II", "III")
            case 'all_Lucy':
                return ("11S",
                        "23S", "21S", "23P", "21P",
                        "33S", "31S", "33P", "33D", "31D", "31P",
                        "43S", "41S", "43P", "43D", "41D", "43F", "41F", "41P",
                        "II")
            case 'all_Tarumi':
                return ("11S",
                        "23S", "21S", "23P", "21P",
                        "33S", "31S", "33P", "33D", "31D", "31P",
                        "43S", "41S", "43P", "43D", "41D", "43F", "41F", "41P",
                        "II", "III")
            case _:
                raise ValueError("You should choose the mode from 'three_1', 'three_3', 'four',\
                                  'six', 'seven', 'n3_Lucy', 'n3_Tarumi', 'all_Lucy' or 'all_Tarumi'.")
    
    @property
    def dim_rate_mat(self) -> int:
        return len(self.use_levels)
    
    @property
    def use_id(self) -> pd.DataFrame:
        # state id
        fname_id = Path(DATA_HE_DIR, "He_state_id.csv")
        df_id = pd.read_csv(fname_id, comment='#')

        return df_id[df_id["term"].isin(self.use_levels)].reset_index(drop=True)

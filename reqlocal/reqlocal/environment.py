from dataclasses import dataclass

import numpy.typing as npt

import numpy as np

from reqlocal.constants import C, H, K_B
from reqlocal.base import UVBlanketingParameter


@dataclass(frozen=True, slots=True)
class EjectaParameter:
    """
    Ejecta parameter of supernova or kilonova

    Parameters
    ----------
    uv_bk_prm : reqlocal.base.UVBlanketingParameter
        UV blanketing parameter

    nd_He : float
        Helium total number density in cm^-3

    w_geo : float
        geometric dilution factor

    t_expl : float
        time after explosion in s

    tem_r : float
        ejecta radiation temperature in K

    tem_e : float
        ejecta electron temperature in K

    r_heat : float
        non-thermal heating rate per ion in erg s^-1 ion^-1

    pr_esc_rr_gr : float
        escape probability of photons emitted upon direct recombination to the ground state, default 1.0
    
    input_j_spec : {'DiluteBlackBody', 'numerical', str}
        Select type of input spectrum, default 'DiluteBlackBody'
        - 'DiluteBlackBody' : W*B_nu
        - 'numerical' : read ascii file
    
    freq_spec_raw_numerical : ndarray, optional
        independent variables of input spectrum (in frequency) used when 'input_j_spec' == 'numerical'

    j_in_raw_numerical : ndarray, optional
        numerical input spectrum (in frequency) used when 'input_j_spec' == 'numerical'
    
    nd_env : float
        number density of environmental nuclei
    
    frac_e_env_init : float
        initial electron fraction of environmental nuclei (that is, frac_e_env_init = (nd_e of environmental atom)/nd_env)
    
    with_Sr_pop : {'LTE', 'NLTE', str}, optional
        whether to consider the Sr population simultaneously and how to treat its ionization

    nd_Sr : float, optional
        Strontium total number density in cm^-3 used when 'with_Sr_pop' is not None
    """
    uv_bk_prm: UVBlanketingParameter

    nd_He: float
    w_geo: float
    t_expl: float
    
    tem_r: float
    tem_e: float
        
    r_heat: float

    pr_esc_rr_gr: float = 1.0

    input_j_spec: str = "DiluteBlackBody"
    freq_spec_raw_numerical: npt.NDArray[np.float64] | None = None
    j_in_raw_numerical: npt.NDArray[np.float64] | None = None

    nd_env: float | None = None
    frac_e_env_init: float | None = None

    with_Sr_pop: str | None = None
    nd_Sr: float | None = None

    def __post_init__(self):
        match self.input_j_spec:
            case 'DiluteBlackBody':
                pass
            case 'numerical':
                if (self.freq_spec_raw_numerical is None) ^ (self.j_in_raw_numerical is None):
                    raise ValueError("Both 'req_spec_raw_numerical' and 'j_in_raw_numerical' must be set.")
            case _:
                raise ValueError("You should choose 'input_j_spec' from 'DiluteBlackBody' or 'numerical'.")

        if (self.nd_env is None) ^ (self.frac_e_env_init is None):
            raise ValueError("Both 'nd_env' and 'frac_e_env_init' must be set.")
        
        if (self.with_Sr_pop is None) ^ (self.nd_Sr is None):
            raise ValueError("Both 'with_Sr_pop' and 'nd_Sr' must be set.")
        
        if self.with_Sr_pop is not None:
            match self.with_Sr_pop:
                case 'LTE':
                    pass
                case 'NLTE':
                    pass
                case _:
                    raise ValueError("You should choose 'with_Sr_pop' from 'LTE' or 'NLTE'.")
        
        if (self.nd_Sr is not None) & (self.frac_e_env_init is None):
            raise ValueError("When 'nd_Sr' is set, 'frac_e_env_init' must be set.")
                   
    def dbb_spec(self, freq_spec: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        # incoming angle-average intensity (W*B_\nu)
        c_1 = 2.0 * H * freq_spec**3 / C**2
        c_2 = H * freq_spec / (K_B * self.tem_r)

        if isinstance(freq_spec, np.ndarray):
            c_2_lower = c_2[c_2 < 10]
            c_2_upper = c_2[c_2 >= 10]
            exp_lower = 1.0 / (np.exp(c_2_lower) - 1.0)
            exp_upper = np.exp(-c_2_upper)
            exp = np.concatenate([exp_lower, exp_upper])

        else:
            if c_2 < 10:
                exp = 1.0 / (np.exp(c_2)-1.0)
            else:
                exp = np.exp(-c_2)

        j_in = self.w_geo * c_1 * exp

        return j_in
    
    def J_in_numerical(self, freq_spec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        freq_spec_raw = self.freq_spec_raw_numerical
        j_in_raw = self.j_in_raw_numerical

        if freq_spec_raw.size != j_in_raw.size:
            message = "'freq_spec_raw_numerical' and 'j_in_raw_numerical' must be of the same length, " \
                    + f"but got {freq_spec_raw.size} and {j_in_raw.size}."
            raise ValueError(message)
        
        j_in = np.interp(freq_spec, freq_spec_raw, j_in_raw)

        return j_in
    
    @classmethod
    def set_ejecta_parameter(cls, e_ph_thr: float = 3.10, r_uv: float = 1.0, 
                             photoion_only: bool = True, **kwargs: dict[str,float]):
        uv_bk_prm = UVBlanketingParameter(e_ph_thr, r_uv, photoion_only)

        return cls(uv_bk_prm, **kwargs)

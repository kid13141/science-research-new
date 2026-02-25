REGISTRY = {}

from .basic_controller import BasicMAC
from .diff_controller import DiffMAC
from .multi_controller import MultiMAC
from .diff_total_controller import Diff_Total_MAC
from .diff_hilp_controller import Diff_Hilp_MAC
from .diff_hilp2_controller import Diff_Hilp2_MAC
from .diff_exp_controller import Diff_Exp_MAC
from .diff_hilp3_controller import Diff_Hilp3_MAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["diff_mac"] = DiffMAC
REGISTRY["multi_mac"] = MultiMAC
REGISTRY["diff_total_mac"] = Diff_Total_MAC
REGISTRY["diff_hilp_mac"] = Diff_Hilp_MAC
REGISTRY["diff_hilp2_mac"] = Diff_Hilp2_MAC
REGISTRY["diff_exp_mac"] = Diff_Exp_MAC
REGISTRY["diff_hilp3_mac"] = Diff_Hilp3_MAC
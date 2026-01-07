REGISTRY = {}

from .basic_controller import BasicMAC
from .diff_controller import DiffMAC
from .multi_controller import MultiMAC
from .diff_total_controller import Diff_Total_MAC
from .diff_hilp_controller import Diff_Hilp_MAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["diff_mac"] = DiffMAC
REGISTRY["multi_mac"] = MultiMAC
REGISTRY["diff_total_mac"] = Diff_Total_MAC
REGISTRY["diff_hilp_mac"] = Diff_Hilp_MAC
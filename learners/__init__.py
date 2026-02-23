from .q_learner import QLearner
from .diff_learner import DiffLearner
from .multi_learner import MultiLearner
from .diff_total_learner import Diff_Total_Learner
from .diff_hilp_learner import Diff_Hilp_Learner
from .diff_exp_learner import Diff_Exp_Learner 
from .diff_hilp2_learner import Diff_Hilp2_Learner
from .diff_hilp3_learner import Diff_Hilp3_Learner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["diff_learner"] = DiffLearner
REGISTRY["multi_learner"] = MultiLearner
REGISTRY["diff_total_learner"] = Diff_Total_Learner
REGISTRY["diff_hilp_learner"] = Diff_Hilp_Learner
REGISTRY["diff_exp_learner"] = Diff_Exp_Learner
REGISTRY["diff_hilp2_learner"] = Diff_Hilp2_Learner
REGISTRY["diff_hilp3_learner"] = Diff_Hilp3_Learner
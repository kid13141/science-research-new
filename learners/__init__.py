from .q_learner import QLearner
from .diff_learner import DiffLearner
from .multi_learner import MultiLearner
from .diff_total_learner import Diff_Total_Learner
from .diff_hilp_learner import Diff_Hilp_Learner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["diff_learner"] = DiffLearner
REGISTRY["multi_learner"] = MultiLearner
REGISTRY["diff_total_learner"] = Diff_Total_Learner
REGISTRY["diff_hilp_learner"] = Diff_Hilp_Learner

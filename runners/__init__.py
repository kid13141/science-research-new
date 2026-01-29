REGISTRY = {}

from .diff_runner import DiffRunner
REGISTRY["diff"] = DiffRunner

from .multi_runner import MultiRunner
REGISTRY["multi"] = MultiRunner

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .diff_total_runner import Diff_Total_Runner
REGISTRY["diff_total"] = Diff_Total_Runner

from .diff_hilp_runner import Diff_Hilp_Runner
REGISTRY["diff_hilp"] = Diff_Hilp_Runner

from .diff_hilp2_runner import Diff_Hilp2_Runner
REGISTRY["diff_hilp2"] = Diff_Hilp2_Runner


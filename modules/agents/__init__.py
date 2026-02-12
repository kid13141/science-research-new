REGISTRY = {}

from .rnn_agent import RNNAgent
from .double_rnn_agent import Double_RNNAgent
from .reset_rnn_agent import Reset_RNNAgent
from .guf_dec_agent import GUF_Dec_Agent
from .guf_exp_agent import GUF_Exp_Agent
from .guf_double_agent import RNNAgent1
from .sgd_agent import SGDAgent
# from .guf_double_agent import RNNAgent2

REGISTRY["rnn"] = RNNAgent
REGISTRY["double_rnn"] = Double_RNNAgent
REGISTRY["guf_dec_agent"] = GUF_Dec_Agent
REGISTRY["guf_exp_agent"] = GUF_Exp_Agent
REGISTRY["reset_agent"] = Reset_RNNAgent
REGISTRY["rnn_agent1"] = RNNAgent1
REGISTRY["sgd_agent"] = SGDAgent
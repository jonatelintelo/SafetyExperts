## Overview

[collect_gate_output.py](collect_gate_output.py) collects and saves the expert pattern traces.

[process_gate_output.py](process_gate_output.py) loads in the previously saved expert traces and transforms it into a usable dataset for the LSTM.

[train_lstm.py](train_lstm.py) trains the LSTM and performs the token / expert analysis.

[data_utils.py](data_utils.py) contains common functions used for any LLM data, such as loading, transformation

[model_utils.py](model_utils.py) contains common functions for things related to LLM models, such as hook functions, loading, generation.
## Overview

Step 1: [collect_gate_output.py](collect_gate_output.py) collects and saves the expert pattern traces.

Step 2: [process_gate_output.py](process_gate_output.py) loads in the previously saved expert traces and transforms it into a usable dataset for the LSTM.

Step 3: [train_lstm.py](train_lstm.py) trains the LSTM and saves the identified utility experts.

Step 4: [prune_model.py](prune_model.py) prunes the model and measures ASR.
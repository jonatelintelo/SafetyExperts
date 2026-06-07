## Overview

Step 1: [1_create_lstm_input.py](1_create_lstm_input.py) collects top-k experts selections and creates the LSTM input data.

Step 2: [2_train_lstm.py](2_train_lstm.py) trains the LSTM used to find safety experts.

Step 3: [3_find_safety_experts.py](3_find_safety_experts.py) finds and saves safety experts to silence.

Step 4: [4_prune_safety_experts.py](4_prune_safety_experts.py) lobotomizes (silences) the model and measures ASR.
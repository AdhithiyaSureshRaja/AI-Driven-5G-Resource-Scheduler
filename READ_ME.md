# AI-Based Resource Scheduling in 5G Networks (Python, srsRAN, Linux)

## Overview
This project demonstrates an AI-driven dynamic scheduler integrated into the **srsRAN 5G open-source framework** running on **Embedded Linux**. It uses a lightweight reinforcement-learning agent to optimize resource block allocation in simulated/real-time testbeds and compares the AI scheduler against traditional algorithms (Proportional Fair, Round Robin).



## Quickstart (simulation mode)

### 1. Create a virtualenv and install dependencies:

    python3 -m venv venv

    source venv/bin/activate

    pip install -r requirements.txt**

### 2. If you have srsRAN: 
    
    `srsran_setup/install_srsran.sh` to install and configure.

### 3. Generate or parse telemetry:

    python3 src/data_extraction.py --simulate

### 4. Train the RL scheduler (on simulated traces):

    python3 src/train_rl_model.py --episodes 200

### 5. Evaluate performance:

    python3 src/evaluate_performance.py --mode simulate

### 6. Run the scheduler in live/sim mode (demo):

    python3 src/ai_scheduler.py --mode simulate


## Notes
- The repository is designed to run in a simulated environment if you don't have srsRAN hardware.
- If integrating with a real srsRAN testbed, update `srsran_setup/gnb.conf` and `srsran_setup/ue.conf` and use `data_extraction.py` to parse logs.


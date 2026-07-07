Arkanoid Reinforcement Learning Agent

This project implements a Reinforcement Learning (RL) agent that learns to play the NES classic Arkanoid. Using a custom Domain-Driven Design (DDD) architecture, it bridges the nes-py emulator, an OpenCV-based computer vision pipeline, and a custom Q-Learning algorithm (using Watkins's TD(λ)) to train an AI to master the game from raw pixel data.

Features

Custom Computer Vision Pipeline: Extracts sub-pixel coordinates, velocities, and ball trajectories directly from the raw emulator frame, avoiding RAM-injection hacks.

TD(λ) Q-Learning Brain: Implements Watkins's replacing traces and custom reward shaping to teach the agent physics-based survival.

Live Telemetry Dashboard: A 60 FPS OpenCV dashboard showing the game, the agent's false-color vision representation, and real-time Q-value charts.

Long-term Metrics: Matplotlib integration to track episodic rewards, survival frames, and exploration decay over time.

Project Structure

main.py: The main orchestrator connecting all domain modules.

rl/: Contains the RL Brain, TD(λ) policy, state discretizer, and reward shaper.

vision/: OpenCV-based physics environment that tracks the ball, paddle, and blocks.

emulator/: Adapter for the nes-py engine and bitmask input translator.

display/: The UI components (Live OpenCV dashboard and Matplotlib metrics).

domain/: Shared data models (e.g., FramePerception, Coordinate).

Prerequisites

Python 3.12 (recommended)

An Arkanoid NES ROM file named arkanoid.nes placed in the roms/ directory.

Installation

To avoid version mismatch errors (especially with the gym library), it is highly recommended to use a virtual environment and install the exact locked dependencies.

Create and activate a virtual environment:

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate


Install the exact dependencies from the requirements file:

pip install -r requirements.txt


(For more details on environment sharing and troubleshooting, refer to VENV_GUIDE.md).

How to Run

The orchestrator is driven by the --mode argument.

1. Showcase Mode (Default)

Watch a pre-trained agent play the game optimally without exploring or updating its Q-table.

# Run using a model in the root directory:
python main.py --mode showcase

# Run using a specific trained session's champion model:
python main.py --mode showcase --session sessions/run_YYYYMMDD_HHMMSS


2. UI Training Mode

Train the agent from scratch while rendering the live game, agent vision, and Q-value decision bars. Automatically creates a timestamped session folder in sessions/.

python main.py --mode train_ui


3. Headless Training Mode

Train the agent as fast as possible by disabling the OpenCV visualization. Ideal for leaving the agent to learn in the background. It will still output telemetry to the console and update the metrics charts at the end of episodes. Automatically creates a timestamped session folder in sessions/.

python main.py --mode train_headless


4. Resuming a Training Session

To continue training from a previous run (either in UI or Headless mode), pass the --session flag pointing to the specific timestamped directory. This seamlessly loads the exact Q-table, exploration rate (epsilon), and historical telemetry to pick up right where it left off.

# Resume with UI:
python main.py --mode train_ui --session sessions/run_YYYYMMDD_HHMMSS

# Resume headless:
python main.py --mode train_headless --session sessions/run_YYYYMMDD_HHMMSS


Checkpoints and Persistence

The RL module automatically saves its progress to the active session folder:

arkanoid_brain.pkl: The continuously updated model.

arkanoid_best_brain.pkl: The "champion" model, saved whenever the agent breaks its own survival record.

telemetry_history.pkl & telemetry_raw.json: The long-term training metrics data.

telemetry_chart.png: An automatically updated visual graph of the agent's progress.
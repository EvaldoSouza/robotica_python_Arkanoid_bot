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

Install the required Python dependencies using pip:

pip install numpy opencv-python matplotlib nes-py


How to Run

The orchestrator is driven by the --mode argument. The default mode is showcase, which is intended for demonstrating a fully trained model.

1. Showcase Mode (Default)

Watch a pre-trained agent play the game optimally without exploring or updating its Q-table. Note: If you run this without a trained arkanoid_best_brain.pkl file, the agent will have no knowledge and will default to holding left.

python main.py
# or explicitly:
python main.py --mode showcase


2. UI Training Mode

Train the agent from scratch (or continue training an existing model) while rendering the live game, agent vision, and Q-value decision bars.

python main.py --mode train_ui


3. Headless Training Mode

Train the agent as fast as possible by disabling the OpenCV visualization. Ideal for leaving the agent to learn in the background. It will still output telemetry to the console and update the metrics charts at the end of episodes.

python main.py --mode train_headless


Checkpoints and Persistence

The RL module automatically saves its progress:

arkanoid_brain.pkl: The continuously updated model.

arkanoid_best_brain.pkl: The "champion" model, saved whenever the agent breaks its own survival record (after passing a 100-frame threshold).
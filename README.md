# AI Snake Game - Neural Network Learning to Play Snake 🐍

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pygame](https://img.shields.io/badge/Pygame-2.0.1%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-blue)
![IPython](https://img.shields.io/badge/IPython-7.16%2B-orange)

Welcome to AI Snake! This project showcases a neural network that learns to play the classic Snake game using reinforcement learning. Watch as the AI evolves from making random moves to mastering the game through continuous learning. Whether you're interested in AI, game development, or just want to see a neural network in action, this project is for you!

## How It Works 🎮
The neural network is developed using reinforcement learning, where the AI agent learns to maximize its score by making the right moves in the Snake game. Here's a quick overview of the process:
- **Random Moves:** Initially, the agent makes almost random moves.
- **Learning Phase:** As the game progresses, the AI starts to recognize patterns, learning which moves are beneficial.
- **Performance Tracking:** After each game, a .png file is saved that shows the learning progress of the network in the form of a graph.

## Project Structure 🛠️
Here's a brief overview of the key files in this project:
```bash
src/
│
├── agent.py        # Main file to run the AI agent
├── game.py         # Contains the logic for the Snake game
├── helper.py       # Helper functions for the project
└── model.py        # Neural network model definition
```
- **agent.py:** This is the main script that runs the AI agent. It handles the interaction between the neural network and the game environment.
- **game.py:** Contains all the game logic, including the rules, the environment setup, and the mechanics of the Snake game.
- **helper.py:** Provides helper functions that assist with various tasks, such as data manipulation and utility functions.
- **model.py:** Defines the neural network architecture used by the agent to learn and make decisions.

## Getting Started 🚀

### Prerequisites
To run this project, you'll need to have the following libraries installed:
- **Pygame:** For the Snake game environment.
- **PyTorch:** To build and train the neural network.
- **Matplotlib:** For visualizing the learning progress.
- **IPython:** Useful for interactive execution and testing.

### Installation
- **Clone** the Repository:
```bash
git clone https://github.com/your-username/AI_Snake.git
```
- **Navigate** to the Source Directory:
```bash
cd AI_Snake/src
```
- **Install** the Required Libraries: If you don't already have the required libraries, you can install them using pip:
```bash
pip install pygame torch matplotlib ipython
```

### Running the AI Snake
To start the AI agent and watch it play Snake:
- **Run** the agent.py script:
```bash
python agent.py
```
- **Watch** the Learning in Action:
- At first, the AI will make random moves.
- As it plays more games, you'll notice the AI making smarter decisions.
- A **.png** file will be saved and updated after each game, showing a graph of the network's learning progress.

## Visualization 📊
The project saves a learning graph as a **.png** file after each game, so you can visually track the progress of the neural network as it learns to play Snake. This file will be located in the **src/** directory.

## Contributing 🤝
Contributions are welcome! If you have ideas for improving the AI, adding new features, or optimizing the code, feel free to fork the repository and submit a pull request.

## Support 📧
If you encounter any issues or have questions, feel free to open an issue or contact me directly via GitHub.

##
Thank you for checking out **AI Snake**! We hope you enjoy watching the AI learn and improve. Happy coding! 🎉

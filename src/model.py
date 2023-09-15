import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

"""The Linear_QNet class is a simple feedforward neural network with an input layer,
 a hidden layer, and an output layer. It is used as an approximation function for 
 the Q-learning agent in the snake game."""
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the module with two linear layers
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # Save the template to a specific file
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(), file_name)


"""The QTrainer class is responsible for training the QNet model. 
It takes as input the current state, the action taken, the reward obtained, 
the next state and a done flag indicating whether the game is finished."""
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr # Learning rate
        self.gamma = gamma # Discount factor for future rewards
        self.model = model # The QNet model being trained
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # Adam Optimizer
        self.criterion = nn.MSELoss() # Mean Squared Error (MSE) loss function

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        #(n,x)

        if len(state.shape) == 1:
            #(1,x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        #1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma*torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action).item()] = Q_new


        #2: Q_new r+y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad() # Calculate the loss between targets and forecasts
        loss = self.criterion(target, pred) # Calculate gradients
        loss.backward() # Update model weights

        self.optimizer.step()









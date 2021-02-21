import torch
import torch.nn as nn
import default


class DQN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """Initialization."""
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 32), 
            nn.ReLU(),
            nn.Linear(32, 64), 
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class DRQN(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 out_dim: int, 
                 **kwargs):
    
        """Initialization."""
        super(DRQN, self).__init__()
        
        self.in_dim = in_dim
        self.num_layers = kwargs.pop('num_layers', default.NUM_LAYERS)
        self.hidden_dim = kwargs.pop('hidden_dim', default.HIDDEN_DIM)
        self.hidden_input_size = kwargs.pop('hidden_input', 64)
        self.last_dim = kwargs.pop('last_dim', 64)

        self.first_layer = nn.Sequential(
            nn.Linear(in_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, self.hidden_input_size), 
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size = self.hidden_input_size, 
                            hidden_size = self.hidden_dim,
                            num_layers = self.num_layers)
        
        self.last_layer = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.last_dim), 
            nn.ReLU(),
            nn.Linear(self.last_dim, out_dim)
        )        

    def forward(self, input: torch.Tensor, 
                hidden_state, cell_state):
        x = self.first_layer(input) 
        x = x.view(1, -1, self.hidden_input_size) # This should be [1, batch, hidden input size]
        x, (next_hidden_state, next_cell_state) = self.lstm(x, (hidden_state, cell_state))

        x = x.view(-1, self.hidden_dim) # This should be [batch, hidden_dim]

        x = self.last_layer(x)

        return x, next_hidden_state, next_cell_state

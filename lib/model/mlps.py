import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Super basic but super useful MLP class.
    """
    def __init__(self, 
        pos_enc_samples_dim,
        pos_enc_ray_dir_dim,
        output_dim,
        activation = torch.relu,
        bias = True,
        layer = nn.Linear,
        num_layers = 4, 
        hidden_dim = 128, 
        skip       = [2],
        output_activation = None
    ):
        """Initialize the MLP.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()

        self.pos_enc_samples_dim = pos_enc_samples_dim
        self.pos_enc_ray_dir_dim = pos_enc_ray_dir_dim
        self.output_dim = output_dim   
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        self.output_activation = output_activation

        if self.skip is None:
            self.skip = []

        self.make()

        self.density_fn = nn.Sequential(
            nn.Linear(128, 1),
            nn.ReLU() # rectified to ensure nonnegative density
        )

        self.rgb_fn = nn.Sequential(
            nn.Linear(128 + self.pos_enc_ray_dir_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def make(self, ):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(self.layer(self.pos_enc_samples_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.pos_enc_samples_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        # self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)


    def forward(self, pos_enc_samples, pos_enc_ray_dir):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = pos_enc_samples.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(pos_enc_samples))
            elif i in self.skip:
                h = torch.cat([pos_enc_samples, h], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        # out = self.lout(h)

        sigma = self.density_fn(h)
        rgb = self.rgb_fn(torch.cat((h, pos_enc_ray_dir), dim=-1))
        return rgb, sigma


def get_activation_class(activation_type):
    """Utility function to return an activation function class based on the string description.

    Args:
        activation_type (str): The name for the activation function.
    
    Returns:
        (Function): The activation function to be used. 
    """
    if activation_type == 'relu':
        return torch.relu
    elif activation_type == 'sin':
        return torch.sin
    elif activation_type == 'softplus':
        return torch.nn.functional.softplus
    elif activation_type == 'lrelu':
        return torch.nn.functional.leaky_relu
    elif activation_type == 'identity':
        return lambda x: x
    else:
        assert False and "activation type does not exist"
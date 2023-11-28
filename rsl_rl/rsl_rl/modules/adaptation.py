import torch
import torch.nn as nn

class Adaptation(nn.Module):
    def __init__(self,  activation,
                        num_actor_observation,
                        num_history_observation,
                        num_latent_dim,
                        beta,
                        encoder_hidden_dims=[128, 64],
                        decoder_hidden_dims=[64, 128]):
        super(Adaptation, self).__init__()
        self.beta = beta
        self.activation = get_activation(activation)
        self.num_actor_obs = num_actor_observation
        self.num_history_obs = num_history_observation

        encoder_layers = []
        encoder_layers.append(nn.Linear(num_history_observation, encoder_hidden_dims[0]))
        encoder_layers.append(self.activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                pass
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(self.activation)

        decoder_layers = []
        decoder_layers.append(nn.Linear(3 + num_latent_dim, decoder_hidden_dims[0]))
        decoder_layers.append(self.activation)
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l], num_actor_observation))
            else:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
                decoder_layers.append(self.activation)

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.velocity = nn.Linear(encoder_hidden_dims[-1], 3)
        self.mu = nn.Linear(encoder_hidden_dims[-1], num_latent_dim)
        self.logvar = nn.Linear(encoder_hidden_dims[-1], num_latent_dim)

    def encode(self, observations):
        x = self.encoder(observations)
        velocity = self.velocity(x)

        mu = self.mu(x)
        logvar = self.logvar(x)
        return velocity, mu, logvar
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, velocity, latent):
        velocity_latent = torch.cat([velocity, latent], dim=-1)
        return self.decoder(velocity_latent)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
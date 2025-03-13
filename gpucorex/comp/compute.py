import torch

class Compute:
    
    def to(self, device='cpu'):
        for var in self.__dict__:
            if isinstance(self.__dict__[var], torch.Tensor):
                self.__dict__[var] = self.__dict__[var].to(device=device)
            elif isinstance(self.__dict__[var], Compute):
                self.__dict__[var] = self.__dict__[var].to(device=device)
            elif isinstance(self.__dict__[var], list):
                for i in range(len(self.__dict__[var])):
                    if isinstance(self.__dict__[var][i], torch.Tensor):
                        self.__dict__[var][i] = self.__dict__[var][i].to(device=device)
            elif isinstance(self.__dict__[var], dict):
                for key in range(len(self.__dict__[var])):
                    if isinstance(self.__dict__[var][key], torch.Tensor):
                        self.__dict__[var][key] = self.__dict__[var][key].to(device=device)
        self.device = device
        return self
    
    def forward(self, *args, **kwargs):
        return args, kwargs
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
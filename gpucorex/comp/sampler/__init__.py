from ._exhaustive import exhaustive
from ._random import randunique
from ._random import randupicate
from ._random import random
from ._montecarlo import montecarlo
from ._adaptive_montecarlo import adaptive_montecarlo

from ._exhaustive import ExhaustiveSampler
from ._random import RandomSampler
from .sampler import Sampler
from ._montecarlo import MonteCarloSampler
from ._adaptive_montecarlo import AdaptiveMonteCarloSampler

    
def get_sampler(name='exhaustive', **kwargs):
    if name == 'exhaustive': return exhaustive
    elif name == 'random': return random
    elif name == 'randunique': return randunique
    elif name == 'randupicate': return randupicate
    elif name == 'montecarlo': return montecarlo
    elif name == 'adaptive_montecarlo': return adaptive_montecarlo
    else: raise KeyError(f'{name} does not exist.')
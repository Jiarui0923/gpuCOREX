import torch
from .compute import Compute

class Criterion(Compute):

    def __init__(self): pass
    def __repr__(self): return '< Criterion >'
    def forward(self): return True
    def parameters(self): pass
    def done(self): pass

class MaxMinCriterion(Criterion):
    """
    Criterion class that defines an optimization criterion based on finding
    the minimum difference from a target value (deltaG), which is smaller
    than a given value.

    Attributes
    ----------
    _target_val : float
        The target value of deltaG.
    _search_field : tuple
        The range within which to search for the optimal parameter.
    _dest_diff : float
        The desired difference threshold for optimization completion.
    _done : bool
        Flag indicating whether the optimization is complete.
    _params : torch.Tensor
        Tensor containing the current batch of parameters for evaluation.
    _scale : float
        Scale used for parameter adjustments.
    _batch_size : int
        The number of parameters in the current batch.

    Methods
    -------
    __init__(self, target_deltaG=5.0, dest_diff=0.1, search_field=(0.01, 2.0)):
        Initializes the criterion with the target value, desired difference,
        and search field.
        
    __repr__(self):
        Returns a string representation of the object.

    parameters(self, size=256):
        Generates a batch of parameters to evaluate.

    done(self, diff):
        Checks if the optimization criterion is met.

    forward(self, x):
        Evaluates the input tensor against the target value and returns
        the optimal parameter and status of optimization.
    """
    def __init__(self, target_deltaG=5.0, dest_diff=0.1,  search_field=(0.01, 2.0)):
        """
        Initializes the MaxMinCriterion.

        Parameters
        ----------
        target_deltaG : float, optional
            The target deltaG value for optimization, the criterion will make the output
            no greater than this value but more approach to this value (default is 5.0).
        dest_diff : float, optional
            The threshold for the difference between the target and actual values (default is 0.1).
        search_field : tuple, optional
            The range of values to search for the optimal parameter (default is (0.01, 2.0)).
        """
        
        self._target_val = target_deltaG
        self._search_field = search_field
        self._dest_diff = dest_diff
        self._done = False
        
    def __repr__(self): return f'< MaxMinCriterion Max={self._max_val}, Min={self._min_val} >'

    def parameters(self, size=256):
        """
        Generates a batch of parameters to evaluate.

        Parameters
        ----------
        size : int, optional
            The number of parameters to generate (default is 256).

        Returns
        -------
        torch.Tensor
            A tensor containing the generated parameters.
        """
        
        self._params = torch.arange(self._search_field[0],
                            self._search_field[1],
                            (self._search_field[1]-self._search_field[0])/size)
        self._scale = (self._search_field[1]-self._search_field[0])/size
        self._batch_size = size
        return self._params
    
    def done(self, diff):
        """
        Checks if the optimization criterion is met.

        Parameters
        ----------
        diff : float
            The current difference between the target value and the actual value.

        Returns
        -------
        bool
            True if the optimization is complete, False otherwise.
        """
        
        if diff < self._dest_diff or self._done: return True
        else: return False

    def forward(self, x):
        """
        Evaluates the input tensor against the target value.

        Parameters
        ----------
        x : torch.Tensor
            The tensor containing values to evaluate.

        Returns
        -------
        tuple
            A tuple containing the optimal parameter and the status of the optimization.
        """
        
        _diff = x - self._target_val
        _status = torch.abs(_diff)
        _status[_diff > 0] = torch.inf
        _min_index = torch.argmax(x)
        _param = self._params[_min_index]
        self._search_field = (_param - (self._search_field[1] - self._search_field[0])/self._batch_size, _param + (self._search_field[1] - self._search_field[0])/self._batch_size)
        if self._search_field[0] == self._search_field[1]: self._done = True
        return self._params[_min_index], _status[_min_index]
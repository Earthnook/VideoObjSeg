""" The base runner for this supervised learning problem.
"""

class RunnerBase:
    """
    """
    def __init__(self,
            dataset,
            model,
            algo,
        ):
        self._dataset = dataset
        self._model = model
        self._algo = algo
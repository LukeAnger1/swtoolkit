"""Evaluator class for getting actions from ML models."""

import os
import pickle
from typing import Any, Optional
import warnings


class Evaluator:
    """
    Evaluator class for loading models and getting actions.

    This class provides an interface for loading trained models and getting
    actions based on the current state. It's designed to be model-agnostic
    and can work with various ML frameworks.

    Args:
        model_path: Path to the trained model file. Can be None for random/untrained behavior.

    Example:
        >>> evaluator = Evaluator("models/my_model.pkl")
        >>> action = evaluator.get_action(current_state)
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Evaluator with a model path.

        Args:
            model_path: Path to the model file. If None, evaluator will need to use
                       default behavior (e.g., random actions).
        """
        self.model_path = model_path
        self.model = None

        if model_path is not None:
            self._load_model()

    def _load_model(self):
        """Load the model from the specified path."""
        if not os.path.exists(self.model_path):
            warnings.warn(
                f"Model file not found at {self.model_path}. "
                "Evaluator will use default behavior."
            )
            return

        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            warnings.warn(
                f"Failed to load model from {self.model_path}: {e}. "
                "Evaluator will use default behavior."
            )

    def get_action(self, state: Any, valid_actions: Optional[list] = None, **kwargs) -> Any:
        """
        Get an action based on the current state.

        Args:
            state: The current state of the environment
            valid_actions: Optional list of valid actions to choose from
            **kwargs: Additional parameters for model inference

        Returns:
            The selected action

        Example:
            >>> state = [0, 1, 0, 2, 0, 0, 0, 0, 0]  # Tic-tac-toe state
            >>> valid_actions = [0, 2, 4, 5, 6, 7, 8]  # Available positions
            >>> action = evaluator.get_action(state, valid_actions)
        """
        if self.model is None:
            # Default behavior: random action from valid actions
            if valid_actions is not None and len(valid_actions) > 0:
                import random
                return random.choice(valid_actions)
            raise ValueError("No model loaded and no valid actions provided")

        # If model is loaded, use it to get action
        # This is a generic interface - specific model types should override this
        if hasattr(self.model, 'predict'):
            # Scikit-learn style model
            prediction = self.model.predict([state])[0]
            return prediction
        elif hasattr(self.model, 'get_action'):
            # Custom model with get_action method
            return self.model.get_action(state, valid_actions=valid_actions, **kwargs)
        elif callable(self.model):
            # Model is a callable function
            return self.model(state, valid_actions=valid_actions, **kwargs)
        else:
            raise NotImplementedError(
                f"Model type {type(self.model)} not supported. "
                "Model should have 'predict' or 'get_action' method, or be callable."
            )

    def save_model(self, save_path: Optional[str] = None):
        """
        Save the current model to disk.

        Args:
            save_path: Path to save the model. If None, uses self.model_path
        """
        if self.model is None:
            warnings.warn("No model to save")
            return

        save_path = save_path or self.model_path
        if save_path is None:
            raise ValueError("No save path specified")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)

    def update_model(self, model: Any):
        """
        Update the evaluator with a new model.

        Args:
            model: The new model to use for evaluation
        """
        self.model = model

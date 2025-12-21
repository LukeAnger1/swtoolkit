"""DataSaver class for managing training data collection."""

import os
import json
import pickle
from typing import Any, Dict, List, Optional
from pathlib import Path
import threading


class DataSaver:
    """
    DataSaver class for collecting and saving training data.

    This class manages the collection of training experiences, generates unique IDs
    for each experience, and handles periodic saving of data. It also supports
    reward propagation to previous actions using a gamma (discount) factor.

    Args:
        save_dir: Directory where data files will be saved
        save_frequency: How often to save data (number of experiences)
        gamma: Discount factor for reward propagation (0-1). If None, no propagation.
        counter_file: Path to the counter file. If None, uses save_dir/.counter

    Example:
        >>> saver = DataSaver("training_data", save_frequency=100, gamma=0.9)
        >>> exp_id = saver.add_experience(state=[0, 0, 0], action=4, reward=0)
        >>> saver.set_reward(exp_id, reward=1.0)  # Update reward later
    """

    def __init__(
        self,
        save_dir: str = "training_data",
        save_frequency: int = 100,
        gamma: Optional[float] = None,
        counter_file: Optional[str] = None
    ):
        """
        Initialize the DataSaver.

        Args:
            save_dir: Directory to save training data
            save_frequency: Save data every N experiences
            gamma: Discount factor for reward propagation (0-1)
            counter_file: Path to counter file for unique ID generation
        """
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        self.gamma = gamma

        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Counter file for unique ID generation
        if counter_file is None:
            self.counter_file = self.save_dir / ".counter"
        else:
            self.counter_file = Path(counter_file)

        # Initialize counter
        self._counter = self._load_counter()
        self._counter_lock = threading.Lock()

        # Buffer for experiences
        self.experiences: List[Dict[str, Any]] = []
        self._experiences_lock = threading.Lock()

        # Track episode sequences for reward propagation
        self._current_episode: List[int] = []
        self._episode_lock = threading.Lock()

    def _load_counter(self) -> int:
        """Load the counter from file or initialize to 0."""
        if self.counter_file.exists():
            try:
                with open(self.counter_file, 'r') as f:
                    return int(f.read().strip())
            except (ValueError, IOError):
                return 0
        return 0

    def _save_counter(self):
        """Save the current counter to file."""
        with open(self.counter_file, 'w') as f:
            f.write(str(self._counter))

    def _get_next_id(self) -> int:
        """
        Generate and return the next unique ID.

        Returns:
            A unique integer ID
        """
        with self._counter_lock:
            unique_id = self._counter
            self._counter += 1
            self._save_counter()
        return unique_id

    def add_experience(
        self,
        state: Any,
        action: Any,
        reward: float = 0.0,
        next_state: Optional[Any] = None,
        done: bool = False,
        **kwargs
    ) -> int:
        """
        Add a new experience to the buffer.

        Args:
            state: The state before taking the action
            action: The action taken
            reward: The immediate reward received
            next_state: The state after taking the action
            done: Whether this is a terminal state
            **kwargs: Additional data to store with the experience

        Returns:
            The unique ID assigned to this experience
        """
        exp_id = self._get_next_id()

        experience = {
            'id': exp_id,
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            **kwargs
        }

        with self._experiences_lock:
            self.experiences.append(experience)

        with self._episode_lock:
            self._current_episode.append(exp_id)

            # If episode is done, handle it
            if done:
                self._finalize_episode()

        # Check if we should save
        if len(self.experiences) >= self.save_frequency:
            self.save()

        return exp_id

    def set_reward(
        self,
        experience_id: int,
        reward: float,
        propagate: bool = True,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """
        Set or update the reward for an experience.

        Args:
            experience_id: The ID of the experience to update
            reward: The new reward value
            propagate: Whether to propagate reward to previous actions in episode
            additional_data: Additional data to add to the experience
        """
        with self._experiences_lock:
            # Find and update the experience
            for exp in self.experiences:
                if exp['id'] == experience_id:
                    exp['reward'] = reward
                    if additional_data:
                        exp.update(additional_data)
                    break

            # Propagate reward if requested and gamma is set
            if propagate and self.gamma is not None:
                self._propagate_reward(experience_id, reward)

    def _propagate_reward(self, target_id: int, reward: float):
        """
        Propagate reward backwards to previous actions in the current episode.

        Args:
            target_id: The ID of the experience that received the reward
            reward: The reward to propagate
        """
        with self._episode_lock:
            if target_id not in self._current_episode:
                return

            # Find the index of the target experience
            target_idx = self._current_episode.index(target_id)

            # Propagate backwards with discount factor
            discounted_reward = reward
            for i in range(target_idx - 1, -1, -1):
                discounted_reward *= self.gamma
                exp_id = self._current_episode[i]

                # Update the experience reward
                for exp in self.experiences:
                    if exp['id'] == exp_id:
                        exp['reward'] = exp.get('reward', 0.0) + discounted_reward
                        break

    def _finalize_episode(self):
        """Finalize the current episode and start a new one."""
        # Reset episode tracking
        self._current_episode = []

    def save(self, filename: Optional[str] = None):
        """
        Save the current experiences to disk and clear the buffer.

        Args:
            filename: Custom filename for this save. If None, auto-generates.
        """
        if not self.experiences:
            return

        with self._experiences_lock:
            if filename is None:
                # Generate filename based on ID range
                first_id = self.experiences[0]['id']
                last_id = self.experiences[-1]['id']
                filename = f"experiences_{first_id:08d}_{last_id:08d}.pkl"

            save_path = self.save_dir / filename

            # Save experiences
            with open(save_path, 'wb') as f:
                pickle.dump(self.experiences, f)

            # Also save a JSON version for easy inspection
            json_path = save_path.with_suffix('.json')
            try:
                with open(json_path, 'w') as f:
                    json.dump(self.experiences, f, indent=2, default=str)
            except (TypeError, ValueError):
                # If experiences contain non-serializable objects, skip JSON
                pass

            # Clear the buffer
            self.experiences = []

    def start_episode(self):
        """Start a new episode (clear episode tracking)."""
        with self._episode_lock:
            self._current_episode = []

    def end_episode(self, final_reward: Optional[float] = None):
        """
        End the current episode.

        Args:
            final_reward: Optional final reward to apply to the last action
        """
        if final_reward is not None and self._current_episode:
            last_id = self._current_episode[-1]
            self.set_reward(last_id, final_reward, propagate=True)

        self._finalize_episode()

    def flush(self):
        """Force save all buffered experiences regardless of save_frequency."""
        self.save()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the data collection.

        Returns:
            Dictionary containing stats like total experiences, buffer size, etc.
        """
        return {
            'total_experiences_generated': self._counter,
            'buffered_experiences': len(self.experiences),
            'save_frequency': self.save_frequency,
            'gamma': self.gamma,
            'save_dir': str(self.save_dir),
            'current_episode_length': len(self._current_episode)
        }

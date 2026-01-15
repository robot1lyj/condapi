import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def infer_rtc(self, obs: Dict, rtc: Dict) -> Dict:
        """Infer actions with RTC guidance if supported."""
        raise NotImplementedError("RTC inference is not supported by this policy.")

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass

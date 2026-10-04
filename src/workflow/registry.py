"""
Workflow Step Registry.
Enables dynamic registration, discovery, validation, and assembly of workflow steps.
"""
from typing import Callable, Dict, List, Optional, Type
from .base import WorkflowStep
from ..utils.logger import get_logger

logger = get_logger("workflow_registry")


class StepRegistry:
    """Central repository of registered workflow steps."""

    _steps: Dict[str, Type[WorkflowStep]] = {}

    @classmethod
    def register(cls, name: Optional[str] = None):
        """
        Decorator to register a WorkflowStep class.
        Usage:
            @StepRegistry.register("dna_validation")
            class DNAValidationStep(WorkflowStep):
                ...
        """
        def decorator(subclass: Type[WorkflowStep]):
            step_name = (name or subclass.__name__).lower()
            cls._steps[step_name] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[WorkflowStep]]:
        """Retrieve a registered step class by name."""
        return cls._steps.get(name.lower())

    @classmethod
    def list_steps(cls) -> List[str]:
        """List all registered step identifiers."""
        return sorted(list(cls._steps.keys()))

    @classmethod
    def instantiate(cls, name: str, *args, **kwargs) -> Optional[WorkflowStep]:
        """Instantiate a registered step."""
        step_cls = cls.get(name)
        if step_cls is None:
            logger.error(f"Step '{name}' is not registered in StepRegistry.")
            return None
        return step_cls(*args, **kwargs)


def register_step(name: Optional[str] = None):
    """Convenience alias for StepRegistry.register."""
    return StepRegistry.register(name)

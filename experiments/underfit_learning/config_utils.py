"""
Config Utilities for Underfit Learning Experiments

This module provides utilities to parse hyperparameter specifications from YAML configs
into callable samplers that can be used by the prior generators.
"""

import re
import random
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch.nn as nn
from priors.utils import (
    uniform_int_sampler_f,
    uniform_sampler_f,
    gamma_sampler_f,
    beta_sampler_f,
    random_activation_sampler
)


def parse_sampler(spec):
    """Parse string specification into callable sampler.

    Args:
        spec: Can be:
            - A number (int/float): Returns lambda that returns that constant
            - A string like "uniform_int(3, 8)": Parses and returns appropriate sampler
            - Already a callable: Returns it as-is

    Returns:
        A callable that returns sampled values

    Examples:
        >>> sampler = parse_sampler("uniform_int(3, 8)")
        >>> value = sampler()  # Returns random int in [3, 8)

        >>> sampler = parse_sampler("gamma(2.0, 20.0)")
        >>> value = sampler()  # Returns gamma-distributed value

        >>> sampler = parse_sampler(100)
        >>> value = sampler()  # Returns 100
    """
    # If it's already a number, return constant sampler
    if isinstance(spec, (int, float)):
        return lambda: spec

    # If it's already callable, return as-is
    if callable(spec):
        return spec

    # If not a string, error
    if not isinstance(spec, str):
        raise ValueError(f"Invalid sampler spec type: {type(spec)}")

    # Parse string specification
    match = re.match(r'(\w+)\((.*)\)', spec)
    if not match:
        raise ValueError(f"Invalid sampler spec format: {spec}. Expected format like 'uniform_int(3, 8)'")

    func_name, args_str = match.groups()
    args = [float(x.strip()) for x in args_str.split(',')]

    # Map function names to sampler constructors
    sampler_map = {
        'uniform_int': lambda: uniform_int_sampler_f(int(args[0]), int(args[1])),
        'uniform': lambda: uniform_sampler_f(args[0], args[1]),
        'gamma': lambda: gamma_sampler_f(args[0], args[1]),
        'beta': lambda: beta_sampler_f(args[0], args[1]),
        'constant': lambda: (lambda: args[0]),
    }

    if func_name not in sampler_map:
        raise ValueError(f"Unknown sampler function: {func_name}. Available: {list(sampler_map.keys())}")

    return sampler_map[func_name]()


def parse_activation(spec):
    """Parse activation specification into activation class or sampler.

    Args:
        spec: Can be:
            - "random": Returns sampler that randomly selects activations
            - Specific name like "relu", "tanh", etc.: Returns that activation class
            - Already an activation class: Returns it as-is

    Returns:
        Either an activation class (e.g., nn.ReLU) or a callable that returns one

    Examples:
        >>> activation = parse_activation("random")
        >>> ActClass = activation()  # Returns random activation class

        >>> activation = parse_activation("relu")
        >>> # activation is nn.ReLU class
    """
    # If already a class, return as-is
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec

    # If callable, return as-is
    if callable(spec):
        return spec

    if not isinstance(spec, str):
        raise ValueError(f"Invalid activation spec type: {type(spec)}")

    # Handle "random" activation
    if spec.lower() == "random":
        return random_activation_sampler()

    # Map string names to activation classes
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "leakyrelu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
    }

    spec_lower = spec.lower()
    if spec_lower in activation_map:
        return activation_map[spec_lower]

    raise ValueError(f"Unknown activation: {spec}. Available: {list(activation_map.keys())} or 'random'")


def process_hyperparameters(hyper_dict):
    """Process hyperparameter dict, converting string specs to callables.

    This function takes a dictionary of hyperparameters (typically from a YAML config)
    and converts string specifications into the appropriate callable samplers that
    the prior generators expect.

    Args:
        hyper_dict: Dictionary of hyperparameters with string specifications

    Returns:
        Dictionary with specifications converted to callables

    Example:
        >>> hypers = {
        ...     'num_layers': 'uniform_int(3, 8)',
        ...     'hidden_dim': 'uniform_int(50, 500)',
        ...     'activation': 'random',
        ...     'init_std': 'gamma(2.0, 20.0)',
        ... }
        >>> processed = process_hyperparameters(hypers)
        >>> # processed['num_layers']() returns random int 3-7
        >>> # processed['activation']() returns random activation class
    """
    if not isinstance(hyper_dict, dict):
        return hyper_dict

    # GP-specific hyperparameters that should stay as numbers (not callables)
    # These are passed directly to GammaPrior and other GP constructors
    GP_NUMERIC_PARAMS = {
        'lengthscale_concentration', 'lengthscale_rate',
        'outputscale_concentration', 'outputscale_rate',
        'noise_concentration', 'noise_rate'
    }

    processed = {}

    for key, value in hyper_dict.items():
        if key == 'activation':
            # Special handling for activation functions
            processed[key] = parse_activation(value)
        elif isinstance(value, str) and '(' in value:
            # Parse sampler specification
            processed[key] = parse_sampler(value)
        elif isinstance(value, (int, float)):
            # For GP numeric params, keep as numbers
            # For MLP params, convert to constant samplers
            if key in GP_NUMERIC_PARAMS:
                processed[key] = value  # Keep as number
            else:
                processed[key] = lambda v=value: v  # Convert to callable
        else:
            # Pass through as-is (callables, None, strings like "random", etc.)
            processed[key] = value

    return processed


def load_and_process_config(config_path):
    """Load YAML config and process hyperparameters.

    Args:
        config_path: Path to YAML config file

    Returns:
        Processed config dictionary
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Process hyperparameters if present
    if 'extra_prior_kwargs_dict' in config and 'hyperparameters' in config['extra_prior_kwargs_dict']:
        config['extra_prior_kwargs_dict']['hyperparameters'] = process_hyperparameters(
            config['extra_prior_kwargs_dict']['hyperparameters']
        )

    return config


# Example usage and tests
if __name__ == '__main__':
    print("Testing config_utils...")

    # Test sampler parsing
    print("\n1. Testing sampler parsing:")
    sampler = parse_sampler("uniform_int(3, 8)")
    print(f"   uniform_int(3, 8) sample: {sampler()}")

    sampler = parse_sampler("gamma(2.0, 20.0)")
    print(f"   gamma(2.0, 20.0) sample: {sampler():.4f}")

    sampler = parse_sampler(100)
    print(f"   constant(100) sample: {sampler()}")

    # Test activation parsing
    print("\n2. Testing activation parsing:")
    activation = parse_activation("relu")
    print(f"   'relu' -> {activation}")

    activation = parse_activation("random")
    print(f"   'random' sample -> {activation()}")

    # Test full processing
    print("\n3. Testing full hyperparameter processing:")
    hypers = {
        'num_layers': 'uniform_int(3, 8)',
        'hidden_dim': 'uniform_int(50, 500)',
        'activation': 'random',
        'init_std': 'gamma(2.0, 20.0)',
        'some_constant': 100,
    }
    processed = process_hyperparameters(hypers)
    print(f"   num_layers sample: {processed['num_layers']()}")
    print(f"   hidden_dim sample: {processed['hidden_dim']()}")
    print(f"   activation sample: {processed['activation']()}")
    print(f"   init_std sample: {processed['init_std']():.4f}")
    print(f"   some_constant: {processed['some_constant']()}")

    print("\n✓ All tests passed!")

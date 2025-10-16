# Contributing to RealityGuard

Thank you for your interest in contributing to RealityGuard! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/RealityGuard.git
   cd RealityGuard
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/JonSnow1807/RealityGuard.git
   ```

4. **Install development environment**:
   ```bash
   ./install.sh
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (for full functionality)
- Git

### Environment Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Making Contributions

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues reported in GitHub Issues
- **Features**: Implement new functionality
- **Performance**: Optimize existing code
- **Documentation**: Improve or add documentation
- **Tests**: Add test coverage
- **Examples**: Create usage examples

### Contribution Workflow

1. **Check existing issues** or create a new one to discuss your idea
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following our coding standards

4. **Write/update tests** for your changes

5. **Run tests locally**:
   ```bash
   pytest tests/
   ```

6. **Update documentation** if needed

7. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

### Commit Message Format

We follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc)
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `test:` Test additions or fixes
- `chore:` Build process or auxiliary tool changes

Example:
```
feat: add support for custom replacement strategies

- Implement ReplacementStrategy base class
- Add registration mechanism for custom strategies
- Include example custom strategy in documentation
```

## Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub

4. **PR Title** should follow the commit message format

5. **PR Description** should include:
   - Summary of changes
   - Related issue(s)
   - Test results
   - Screenshots/videos if applicable

6. **Address review feedback** promptly

### PR Checklist

Before submitting:

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] PR description is complete
- [ ] No merge conflicts

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use type hints where appropriate
- Document all public functions/classes

### Code Quality Tools

Run before committing:

```bash
# Format code
black .

# Check style
flake8

# Type checking
mypy realityguard/

# Run all checks
make lint
```

### File Organization

```
realityguard/
├── core/           # Core functionality
├── models/         # AI models
├── utils/          # Utility functions
├── strategies/     # Replacement strategies
└── cache/          # Caching mechanisms

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── fixtures/       # Test data
```

## Testing Guidelines

### Writing Tests

1. **Test file naming**: `test_<module_name>.py`

2. **Test function naming**: `test_<functionality>_<scenario>`

3. **Use fixtures** for common test data:
   ```python
   @pytest.fixture
   def sample_frame():
       return np.ones((720, 1280, 3), dtype=np.uint8)
   ```

4. **Test edge cases** and error conditions

5. **Mock external dependencies**:
   ```python
   @patch('realityguard.models.yolo.YOLO')
   def test_detection(mock_yolo):
       # Test code
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=realityguard --cov-report=html

# Run specific test file
pytest tests/unit/test_cache.py

# Run with verbose output
pytest -v
```

### Performance Testing

For performance-critical changes:

1. Include benchmark tests
2. Compare before/after metrics
3. Document performance impact in PR

Example:
```python
def test_performance_process_frame(benchmark, sample_frame):
    system = RealityGuardAIReplacement()
    result = benchmark(system.process_frame, sample_frame)
    assert result[1]['fps'] > 24  # Minimum FPS requirement
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """Process a single frame with AI replacement.

    Args:
        frame: Input frame in BGR format, shape (H, W, 3)

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Protected frame with replacements
            - dict: Statistics including FPS and replacement count

    Raises:
        ValueError: If frame is not in correct format

    Example:
        >>> system = RealityGuardAIReplacement()
        >>> protected, stats = system.process_frame(frame)
        >>> print(f"FPS: {stats['fps']}")
    """
```

### Documentation Updates

When adding features, update:

1. `README.md` - If it's a major feature
2. `docs/API.md` - API documentation
3. Code docstrings - Always
4. `examples/` - Add usage examples

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Email**: cshrivastava2000@gmail.com

### Review Process

All contributions go through review:

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** on multiple platforms
4. **Documentation review**

### Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes
- Annual contributor spotlight

## Development Tips

### GPU Debugging

```python
import torch
torch.cuda.set_device(0)
torch.cuda.empty_cache()

# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

### Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Your code here
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Common Issues

1. **CUDA out of memory**: Reduce batch_size or use half precision
2. **Slow performance**: Check cache efficiency, enable GPU acceleration
3. **Import errors**: Ensure virtual environment is activated

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or reach out directly. We're here to help!

---

Thank you for contributing to RealityGuard! Your efforts help protect privacy for users worldwide.
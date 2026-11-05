# Contributing

Thank you for your interest in contributing to the Advanced CAx Tool.

## Getting started

- Check the existing documentation and examples before starting work.
- For substantial changes, open an issue or discussion first to align on the proposed approach.
- Keep changes focused and update relevant documentation or examples.

## Submitting a change

- Use a clear, descriptive commit message.
- Run the applicable checks for the code you changed and include their results in your pull request.
- Open a pull request describing the motivation, the changes made, and any relevant validation.

## Development

```bash
pip install -e ".[cad]"
python -m unittest discover -s tests -p "test_*.py"
```

Toolpaths are deterministic: identical geometry and process parameters give identical paths. If a change is meant to alter them, run `python tests/regenerate_golden.py` from the `tests` folder and include the diff of `tests/golden/patterns.json` in your pull request.

Further instructions will be added as more software components become available.

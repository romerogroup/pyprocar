# CLAUDE.md
- You must exlude these folder types `**/**/__pycache__`, `docs/figures`,`docs/meeting_notes`,`docs/research_notes`, `.pixi`, `.ruff_cache` from any tool call, such as read, write, glob, find

- For typing, type annotations, and basedpyright issues, consult `specs/typing-standards.md`.
- Use `pixi run -q -e dev lint-src` to run linting for all files in src. Use `pixi run -q -e dev lint-tests` for tests. Use `pixi run -q -e dev lint` for both.
- Use `pixi run -q -e dev ruff-lint <file_path>` or `pixi run -q -e dev ruff-format <file_path>` for specific files. Add `--src` or `--tests` to use respective config. Add `--verbose` or `-v` for full output (only works with specific files).
- Use `pixi run -q -e dev typecheck` to run typechecks on all files. Use `pixi run -q -e dev typecheck <file_path>` to typecheck a specific file. Add `--verbose` or `-v` for full output (only works with specific files).
- Use `pixi run -q -e dev test` to run all tests. Use `pixi run -q -e dev test <test_pattern>` to run specific tests matching a pattern. Add `--verbose` or `-v` for full output.

- Do not try to get around the above. You must only use these when testing, linting, and typechecking. You should think on the error message and then go to the place the error is happening it see more context about the code.

- All commands have bash script wrappers that summarize output. To solve errors, read and think about the error message and go to the code that is showing the error.
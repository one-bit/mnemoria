# Contributing to Mnemoria

Thank you for your interest in contributing to Mnemoria!

## Getting Started

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/one-bit/mnemoria
   cd mnemoria
   ```

2. Install the Rust stable toolchain with `rustfmt` and `clippy`:

   ```bash
   rustup toolchain install stable
   rustup component add rustfmt clippy
   ```

3. Enable the pre-commit hook:

   ```bash
   git config core.hooksPath .githooks
   ```

   This runs `cargo fmt --check` and `cargo clippy -D warnings` before every commit so issues are caught locally before reaching CI.

4. Build and run the tests:

   ```bash
   cargo build
   cargo test
   ```

## Development Workflow

### Pre-Commit Hook

The repository includes a pre-commit hook in `.githooks/pre-commit` that enforces formatting and lint checks. After cloning, enable it with:

```bash
git config core.hooksPath .githooks
```

If a commit is blocked, the hook will tell you what failed:

- **Formatting** - Run `cargo fmt --all` to auto-fix.
- **Clippy** - Address the warnings shown in the output.

### CI Checks

All pull requests run through GitHub Actions (see `.github/workflows/ci.yml`). The CI pipeline runs:

1. `cargo fmt --all -- --check` - Code formatting
2. `cargo clippy -- -D warnings` - Lint (warnings treated as errors)
3. `cargo test` - Full test suite
4. `cargo build --release` - Release build verification
5. `cargo-deny` - Security vulnerability and license compliance audit

The pre-commit hook covers steps 1-2 locally, so most issues are caught before you push.

### Running Tests

```bash
cargo test
```

### Running Benchmarks

```bash
cargo bench --bench api_perf
```

## Pull Requests

- Create a feature branch from `main`.
- Keep commits focused and write clear commit messages.
- Ensure all CI checks pass before requesting review.
- Add tests for new functionality.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

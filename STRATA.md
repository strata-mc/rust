# Strata fork of rustc: autodiff patches

Branch `strata/autodiff` adds fixes for `std::autodiff` (Enzyme) found while differentiating the
Strata civilization model (`strata-mc/strata-mod`, `tools/civ-model`). `src/tools/enzyme` points at
`strata-mc/enzyme` branch `strata/main`, which carries the matching Enzyme fixes. Each fix should
also go upstream (rust-lang/rust, EnzymeAD/Enzyme) with its reproduction.

## Build and link the toolchain

    git clone --branch strata/autodiff git@github.com:strata-mc/rust.git rust-enzyme
    cd rust-enzyme && git submodule update --init src/tools/enzyme
    cat > bootstrap.toml <<'TOML'
    change-id = "ignore"
    profile = "compiler"
    [llvm]
    download-ci-llvm = false
    enzyme = true
    plugins = true
    link-shared = true
    assertions = false
    ninja = true
    [build]
    extended = false
    docs = false
    [rust]
    debug-assertions = false
    TOML
    ./x.py build --stage 1 library          # about 20 minutes on 32 cores, LLVM included
    rustup toolchain link enzyme build/host/stage1

Needs cmake and ninja. Use with `RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme ...` and `lto = "fat"`.
After changing only Enzyme, `ninja libEnzyme-23.so` in `build/<host>/enzyme/build` and copying the
library into `build/<host>/stage1/lib/rustlib/<host>/lib/` is enough.

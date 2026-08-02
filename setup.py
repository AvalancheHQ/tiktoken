import os

from setuptools import setup
from setuptools_rust import Binding, RustExtension

# Build PCRE2 (used for the tokeniser split) from the vendored C sources rather
# than linking against whatever `libpcre2-8` happens to be installed on the
# build machine. Without this, `pcre2-sys` picks up e.g. Homebrew's PCRE2 via
# pkg-config and the resulting extension module carries a dynamic dependency on
# a library that is not present on users' machines. `setdefault` leaves an
# explicit `PCRE2_SYS_STATIC=0` from the environment alone.
os.environ.setdefault("PCRE2_SYS_STATIC", "1")

setup(
    name="tiktoken",
    rust_extensions=[
        RustExtension(
            "tiktoken._tiktoken",
            binding=Binding.PyO3,
            # Between our use of editable installs and wanting to use Rust for performance sensitive
            # code, it makes sense to just always use --release
            debug=False,
            features=["python"],
        )
    ],
    package_data={"tiktoken": ["py.typed"]},
    packages=["tiktoken", "tiktoken_ext"],
    zip_safe=False,
)

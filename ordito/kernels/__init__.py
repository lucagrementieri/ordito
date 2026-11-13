import warp._src.context as _warp_context

# The Warp modules that existed before any of ordito's kernels were defined (Warp's own, and any a
# caller defined first): ``ordito.precompile.kernel_variants`` builds the modules created after.
WARP_MODULES_BEFORE_ORDITO = frozenset(_warp_context.user_modules)

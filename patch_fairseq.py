"""
Patch fairseq for Python 3.12 compatibility.
Import this before any fairseq imports.
"""
import sys

if sys.version_info >= (3, 12):
    import dataclasses
    
    if not hasattr(dataclasses, '_patched_for_fairseq'):
        _original_get_field = dataclasses._get_field
        
        def _patched_get_field(cls, name, type, kw_only):
            try:
                return _original_get_field(cls, name, type, kw_only)
            except ValueError as e:
                if "mutable default" in str(e):
                    # For fairseq: allow mutable defaults by converting to default_factory
                    import dataclasses as dc
                    # Get the default value
                    default = getattr(cls, name, dc.MISSING)
                    if default is not dc.MISSING:
                        # Create a factory function
                        if isinstance(default, type):
                            # It's a class, create factory that instantiates it
                            factory = lambda: default()
                        else:
                            # It's an instance, create factory that returns a copy
                            factory = lambda: type(default)(default) if hasattr(default, '__dict__') else default
                        
                        return dc.Field(
                            name=name, type=type, default=dc.MISSING,
                            default_factory=factory, repr=True, hash=None,
                            init=True, compare=True, metadata=None,
                            kw_only=kw_only, _field_type=dc._FIELD
                        )
                raise
        
        dataclasses._get_field = _patched_get_field
        dataclasses._patched_for_fairseq = True

from typing import TypedDict


class TypedDictWithDefaultsMeta(type(TypedDict)):  # type: ignore
    def __call__(cls, *args, **kwargs):
        defaults = {}
        """all defaults"""
        keys = []
        """all possible keys"""

        # get defaults and keys info
        anno = getattr(cls, "__annotations__", None)
        if anno is not None:
            keys = list(anno.keys())
            if keys:
                defaults.update(
                    {attr: getattr(cls, attr) for attr in keys if hasattr(cls, attr)}
                )

        args_kw = {}
        if args:  # convert args to kwargs
            assert len(keys) >= len(args), f"found {len(args) - len(keys)} excess args"
            args_kw = {k: v for k, v in zip(keys, args)}

        if kwargs:
            if args_kw:  # check no args-kwargs intersection
                assert not set(args_kw.keys()).intersection(kwargs.keys())
            args_kw.update(kwargs)

        defaults.update(args_kw)
        _cur_k = set(defaults.keys())

        keys = set(keys)
        diff = keys - _cur_k
        if diff:
            raise ValueError(f"some required keys not found: {diff}")
        diff = _cur_k - keys
        if diff:
            raise ValueError(f"excess keys found: {diff}")

        return super().__call__(**defaults)

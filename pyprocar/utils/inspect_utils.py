import inspect


def example_func(a, b, c=10, d=20, *, e=30, f=40, **kwargs):
    pass

        
def get_kwargs(func, defaults=True):
    sig = inspect.signature(func)
    return {
        name: param.default if param.default is not param.empty else None
        for name, param in sig.parameters.items()
        if param.default is not param.empty or param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        )
    }
    
    
def get_args(func, defaults=True):
    sig = inspect.signature(func)
    return [name for name, param in sig.parameters.items() if param.default is not param.empty]
    
    
def keep_func_kwargs(kwargs, func, ):
    func_kwargs = get_kwargs(func)
    return {k: v for k, v in kwargs.items() if k in func_kwargs}
    
def keep_func_args(args, func, defaults=True):
    func_args = get_args(func, defaults)
    return [v for v in args if v in func_args]
    
def keep_func_kwargs_and_args(kwargs, args, func, defaults=True):
    func_kwargs = get_kwargs(func, defaults)
    func_args = get_args(func, defaults)
    return {k: v for k, v in kwargs.items() if k in func_kwargs}, [v for v in args if v in func_args]
    
if __name__ == "__main__":
    print(get_kwargs(example_func))
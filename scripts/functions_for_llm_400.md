### Функция ID: 1

**Исходный код:**
```java
public static Map<TypeVariable<?>, Type> determineTypeArguments(final Class<?> cls, final ParameterizedType superParameterizedType) {
        Objects.requireNonNull(cls, "cls");
        Objects.requireNonNull(superParameterizedType, "superParameterizedType");
        final Class<?> superClass = getRawType(superParameterizedType);
        // compatibility check
        if (!isAssignable(cls, superClass)) {
            return null;
        }
        if (cls.equals(superClass)) {
            return getTypeArguments(superParameterizedType, superClass, null);
        }
        // get the next class in the inheritance hierarchy
        final Type midType = getClosestParentType(cls, superClass);
        // can only be a class or a parameterized type
        if (midType instanceof Class<?>) {
            return determineTypeArguments((Class<?>) midType, superParameterizedType);
        }
        final ParameterizedType midParameterizedType = (ParameterizedType) midType;
        final Class<?> midClass = getRawType(midParameterizedType);
        // get the type variables of the mid class that map to the type
        // arguments of the super class
        final Map<TypeVariable<?>, Type> typeVarAssigns = determineTypeArguments(midClass, superParameterizedType);
        // map the arguments of the mid type to the class type variables
        mapTypeVariablesToArguments(cls, midParameterizedType, typeVarAssigns);
        return typeVarAssigns;
    }
```



---

### Функция ID: 2

**Исходный код:**
```python
def map(
    f: Callable[[pytree.PyTree, tuple[pytree.PyTree, ...]], pytree.PyTree],
    xs: Union[pytree.PyTree, torch.Tensor],
    *args: TypeVarTuple,
):
    r"""
    Performs a map of f with xs. Intuitively, you can think of the semantic being:

    out = []
    for idx in len(xs.size(0)):
        xs_sliced = xs.select(0, idx)
        out.append(f(xs_sliced, *args))
    torch.stack(out)

    .. warning::
        `torch._higher_order_ops.map` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype


    Args:
        f (Callable): a callable that takes an input x, that could either be a single Tensor
            or a nested dict, list of tensors and some additional inputs
        xs: the inputs that're to be mapped over. We'll iterate over the first dim of each x
            and perform f on each slice.

        *args: additional arguments provided to each step of f. They could also be omitted and
            map is able to automatically figure out the read dependency.

    Return:
        the stacked output for each step of f

    Example:

        def f(xs):
            return xs[0] + xs[1] + const1 + const2

        xs = [torch.randn(2, 3), torch.randn(2, 3)]
        const1 = torch.randn(2, 3)
        const2 = torch.randn(2, 3)
        # returns a tensor of shape [2, 2, 3]
        torch._higher_order_ops.map(f, xs)

    """
    flat_xs, xs_spec = pytree.tree_flatten(xs)
    flat_args, args_spec = pytree.tree_flatten(args)
    if not all(isinstance(t, torch.Tensor) for t in flat_xs):
        raise RuntimeError(f"Mapped xs can only consist of tensors. Got xs {flat_xs}.")

    shapes = [xs.shape for xs in flat_xs]
    leading_dim_size = shapes[0][0]
    if leading_dim_size == 0:
        raise RuntimeError("Leading dimensions of mapped xs cannot be 0.")

    if any(cur_shape[0] != leading_dim_size for cur_shape in shapes):
        raise RuntimeError(
            f"Leading dimensions of mapped xs must be consistent. Got shapes {shapes}."
        )

    def run_flattened_map(f, flat_xs, flat_args):
        def wrapped_fn(*flat_args, f, xs_tree_spec, args_tree_spec, num_xs):
            xs = pytree.tree_unflatten(flat_args[:num_xs], xs_tree_spec)
            args = pytree.tree_unflatten(flat_args[num_xs:], args_tree_spec)
            return f(xs, *args)

        inner_f = functools.partial(
            wrapped_fn,
            f=f,
            xs_tree_spec=xs_spec,
            args_tree_spec=args_spec,
            num_xs=len(flat_xs),
        )
        return map_impl(inner_f, flat_xs, flat_args)

    from torch._higher_order_ops.utils import _maybe_compile_and_run_fn

    return _maybe_compile_and_run_fn(run_flattened_map, f, flat_xs, flat_args)
```



---

### Функция ID: 3

**Исходный код:**
```python
def expand(self, *args: Dim) -> _Tensor:
        """
        Expand tensor by adding new dimensions or expanding existing dimensions.

        If all arguments are Dim objects, adds new named dimensions.
        Otherwise, falls back to regular tensor expansion behavior.

        Args:
            args: Either Dim objects for new dimensions or sizes for regular expansion

        Returns:
            New tensor with expanded dimensions

        Example:
            >>> i, j = dims()
            >>> t = torch.randn(3, 4)
            >>> expanded = t[i].expand(j, k)  # Add j, k dimensions
            >>> expanded2 = t[i].expand(2, 4)  # Regular expand with sizes
        """
        info = TensorInfo.create(self, ensure_batched=False, ensure_present=False)

        for arg in args:
            if not isinstance(arg, Dim):
                # Not all args are Dims, fallback to regular expand
                if isinstance(self, torch.Tensor) and not isinstance(self, _Tensor):
                    return torch.Tensor.expand(self, *args)
                else:
                    return self.__torch_function__(
                        torch.Tensor.expand, (type(self),), (self,) + args
                    )

        # All args are Dim objects - proceed with first-class dimension expansion
        if not info:
            # No tensor info available, fallback
            return self.__torch_function__(
                torch.Tensor.expand, (type(self),), (self,) + args
            )

        # First-class dimension expansion - all args are Dim objects
        data = info.tensor
        if data is None:
            # No tensor data available, fallback
            return self.__torch_function__(
                torch.Tensor.expand, (type(self),), (self,) + args
            )

        levels = info.levels

        new_levels: list[DimEntry] = []
        new_sizes = []
        new_strides = []

        for d in args:
            # Check if dimension already exists in current levels or new_levels
            for level in levels:
                if not level.is_positional() and level.dim() is d:
                    raise DimensionBindError(
                        f"expanding dimension {d} already exists in tensor with dims"
                    )
            for new_level in new_levels:
                if not new_level.is_positional() and new_level.dim() is d:
                    raise DimensionBindError(
                        f"expanding dimension {d} already exists in tensor with dims"
                    )

            new_levels.append(DimEntry(d))
            new_sizes.append(d.size)
            new_strides.append(0)

        # Add existing levels
        new_levels.extend(levels)

        # Add existing sizes and strides
        orig_sizes = list(data.size())
        orig_strides = list(data.stride())
        new_sizes.extend(orig_sizes)
        new_strides.extend(orig_strides)

        # Create expanded tensor using as_strided
        expanded_data = data.as_strided(new_sizes, new_strides, data.storage_offset())

        # Return new tensor with expanded dimensions
        result = Tensor.from_positional(expanded_data, new_levels, info.has_device)
        return result  # type: ignore[return-value]  # Tensor and torch.Tensor are interchangeable
```



---

### Функция ID: 4

**Исходный код:**
```python
def get_import_mappings(tree) -> dict[str, str]:
    """
    Retrieve a mapping of local import names to their fully qualified module paths from an AST tree.

    :param tree: The AST tree to analyze for import statements.

    :return: A dictionary where the keys are the local names (aliases) used in the current module
        and the values are the fully qualified names of the imported modules or their members.

    Example:
        >>> import ast
        >>> code = '''
        ... import os
        ... import numpy as np
        ... from collections import defaultdict
        ... from datetime import datetime as dt
        ... '''
        >>> get_import_mappings(ast.parse(code))
        {'os': 'os', 'np': 'numpy', 'defaultdict': 'collections.defaultdict', 'dt': 'datetime.datetime'}
    """
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                module_prefix = f"{node.module}." if hasattr(node, "module") and node.module else ""
                imports[alias.asname or alias.name] = f"{module_prefix}{alias.name}"
    return imports
```



---

### Функция ID: 5

**Исходный код:**
```python
def linearize(func: Callable, *primals) -> tuple[Any, Callable]:
    """
    Returns the value of ``func`` at ``primals`` and linear approximation
    at ``primals``.

    Args:
        func (Callable): A Python function that takes one or more arguments.
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. These are the values at which the function is linearly approximated.

    Returns:
        Returns a ``(output, jvp_fn)`` tuple containing the output of ``func``
        applied to ``primals`` and a function that computes the jvp of
        ``func`` evaluated at ``primals``.

    linearize is useful if jvp is to be computed multiple times at ``primals``. However,
    to achieve this, linearize saves intermediate computation and has higher memory requirements
    than directly applying `jvp`. So, if all the ``tangents`` are known, it maybe more efficient
    to compute vmap(jvp) instead of using linearize.

    .. note::
        linearize evaluates ``func`` twice. Please file an issue for an implementation
        with a single evaluation.

    Example::

        >>> import torch
        >>> from torch.func import linearize
        >>> def fn(x):
        ...     return x.sin()
        ...
        >>> output, jvp_fn = linearize(fn, torch.zeros(3, 3))
        >>> jvp_fn(torch.ones(3, 3))
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]])
        >>>

    """
    # Note: We evaluate `fn` twice.
    # Once for returning the output and other while
    # tracing the graph.
    # If this becomes a bottle-neck, we should update
    # make_fx such that it also returns the output.

    output = func(*primals)
    _, output_spec = tree_flatten(output)

    flat_primals, primals_argspec = tree_flatten(primals)

    # tangents for tracing
    flat_tangents = tuple(p.new_empty(()).expand_as(p) for p in flat_primals)

    # function to trace
    def trace_fn(flat_tangents):
        with fwAD.dual_level():
            flat_duals = tuple(
                fwAD.make_dual(p, t) for p, t in zip(flat_primals, flat_tangents)
            )
            duals = tree_unflatten(flat_duals, primals_argspec)
            output = func(*duals)
            tangents = tree_map_only(
                torch.Tensor, lambda dual: safe_unpack_dual(dual, False)[1], output
            )

        return tangents

    jvp_graph = lazy_dynamo_disallow(make_fx)(trace_fn)(flat_tangents)
    const_folded_jvp_graph = lazy_dynamo_disallow(const_fold.split_const_subgraphs)(
        jvp_graph
    )

    # Hold only the meta-data regarding the primals.
    flat_primals_shape = tuple(p.shape for p in flat_primals)
    flat_primals_device = tuple(p.device for p in flat_primals)
    flat_primals_dtype = tuple(p.dtype for p in flat_primals)

    def forward_ad_checks(flat_tangents):
        for idx, t in enumerate(flat_tangents):
            if t.shape != flat_primals_shape[idx]:
                msg = (
                    f"tangent:{idx} with shape {t.shape} in flattened "
                    f"pytree doesn't match the shape {flat_primals_shape[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

            if t.device != flat_primals_device[idx]:
                msg = (
                    f"tangent:{idx} with device {t.device} in flattened "
                    f"pytree doesn't match the device {flat_primals_device[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

            if t.dtype != flat_primals_dtype[idx]:
                msg = (
                    f"tangent:{idx} with dtype {t.dtype} in flattened "
                    f"pytree doesn't match the dtype {flat_primals_dtype[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

    # jvp_fn : callable to return
    #   It takes care of checking the argspec of tangents,
    #   calling the folded fx graph and unflattening fx graph output
    def jvp_fn(*tangents):
        flat_tangents, tangent_argspec = tree_flatten(tangents)
        if tangent_argspec != primals_argspec:
            raise RuntimeError(
                f"Expected the tangents {tangent_argspec} to have "
                f"the same argspec as the primals {primals_argspec}"
            )

        forward_ad_checks(flat_tangents)

        flat_output = const_folded_jvp_graph(*flat_tangents)
        # const folded graph can return flat output,
        # so transform output.
        return tree_unflatten(flat_output, output_spec)

    return output, jvp_fn
```



---

### Функция ID: 6

**Исходный код:**
```java
private static String getCanonicalName(final String name) {
        String className = StringUtils.deleteWhitespace(name);
        if (className == null) {
            return null;
        }
        int dim = 0;
        final int len = className.length();
        while (dim < len && className.charAt(dim) == '[') {
            dim++;
            if (dim > MAX_DIMENSIONS) {
                throw new IllegalArgumentException(String.format("Maximum array dimension %d exceeded", MAX_DIMENSIONS));
            }
        }
        if (dim >= len) {
            throw new IllegalArgumentException(String.format("Invalid class name %s", name));
        }
        if (dim < 1) {
            return className;
        }
        className = className.substring(dim);
        if (className.startsWith("L")) {
            if (!className.endsWith(";") || className.length() < 3) {
                throw new IllegalArgumentException(String.format("Invalid class name %s", name));
            }
            className = className.substring(1, className.length() - 1);
        } else if (className.length() == 1) {
            final String primitive = REVERSE_ABBREVIATION_MAP.get(className.substring(0, 1));
            if (primitive == null) {
                throw new IllegalArgumentException(String.format("Invalid class name %s", name));
            }
            className = primitive;
        } else {
            throw new IllegalArgumentException(String.format("Invalid class name %s", name));
        }
        final StringBuilder canonicalClassNameBuffer = new StringBuilder(className.length() + dim * 2);
        canonicalClassNameBuffer.append(className);
        for (int i = 0; i < dim; i++) {
            canonicalClassNameBuffer.append("[]");
        }
        return canonicalClassNameBuffer.toString();
    }
```



---

### Функция ID: 7

**Исходный код:**
```python
def rearrange(
    tensor: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]],
    pattern: str,
    **axes_lengths: int,
) -> torch.Tensor:
    r"""A native implementation of `einops.rearrange`, a reader-friendly smart element reordering for multidimensional
    tensors. This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations.

    See: https://einops.rocks/api/rearrange/

    Args:
        tensor (Tensor or sequence of Tensor): the tensor(s) to rearrange
        pattern (str): the rearrangement pattern
        axes_lengths (int): any additional length specifications for dimensions

    Returns:
        Tensor: the rearranged tensor

    Examples:
        >>> # suppose we have a set of 32 images in "h w c" format (height-width-channel)
        >>> images = torch.randn((32, 30, 40, 3))

        >>> # stack along first (batch) axis, output is a single array
        >>> rearrange(images, "b h w c -> b h w c").shape
        torch.Size([32, 30, 40, 3])

        >>> # concatenate images along height (vertical axis), 960 = 32 * 30
        >>> rearrange(images, "b h w c -> (b h) w c").shape
        torch.Size([960, 40, 3])

        >>> # concatenated images along horizontal axis, 1280 = 32 * 40
        >>> rearrange(images, "b h w c -> h (b w) c").shape
        torch.Size([30, 1280, 3])

        >>> # reordered axes to "b c h w" format for deep learning
        >>> rearrange(images, "b h w c -> b c h w").shape
        torch.Size([32, 3, 30, 40])

        >>> # flattened each image into a vector, 3600 = 30 * 40 * 3
        >>> rearrange(images, "b h w c -> b (c h w)").shape
        torch.Size([32, 3600])

        >>> # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
        >>> rearrange(images, "b (h1 h) (w1 w) c -> (b h1 w1) h w c", h1=2, w1=2).shape
        torch.Size([128, 15, 20, 3])

        >>> # space-to-depth operation
        >>> rearrange(images, "b (h h1) (w w1) c -> b h w (c h1 w1)", h1=2, w1=2).shape
        torch.Size([32, 15, 20, 12])
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.stack(tensor)

    rearrange_callable = _create_rearrange_callable(
        tensor.ndim, pattern, **axes_lengths
    )

    return rearrange_callable(tensor)
```



---

### Функция ID: 8

**Исходный код:**
```python
def find_class_methods_with_specific_calls(
    class_node: ast.ClassDef, target_calls: set[str], import_mappings: dict[str, str]
) -> set[str]:
    """
    Identifies class methods that make specific calls.

    This function only tracks target calls within the class scope. Method calling some function defined
    will not be taken into consideration even if this function performs a target call.

    Method calling other method that performs a target call will also be included.

    This function performs a two-pass analysis of the AST:
    1. It first identifies methods containing direct calls to the specified functions
       and records method calls on `self`.
    2. It then identifies methods that indirectly make such calls by invoking the
       methods identified in the first pass.

    :param class_node: The root node of the AST representing the class to analyze.
    :param target_calls: A set of full paths to the method names to track when called.
    :param import_mappings: A mapping of import names to fully qualified module names.

    :return: Method names within the class that either directly or indirectly make the specified calls.

    Examples:
        > source_code = '''
        ... class Example:
        ...     def method1(self):
        ...         my_method().ok()

        ...     def method2(self):
        ...         self.method1()

        ...     def method3(self):
        ...         my_method().not_ok()

        ...     def method4(self):
        ...         self.some_other_method()
        ... '''
        > find_methods_with_specific_calls(
            ast.parse(source_code),
            {"airflow.my_method.not_ok", "airflow.my_method.ok"},
            {"my_method": "airflow.my_method"}
        )
        {'method1', 'method2', 'method3'}
    """
    method_call_map: dict[str, set[str]] = {}
    methods_with_calls: set[str] = set()

    # First pass: Collect all calls and identify methods with specific calls we are looking for
    for node in ast.walk(class_node):
        if not isinstance(node, ast.FunctionDef):
            continue
        method_call_map[node.name] = set()
        for sub_node in ast.walk(node):
            if not isinstance(sub_node, ast.Call):
                continue
            called_function = sub_node.func
            if not isinstance(called_function, ast.Attribute):
                continue
            if isinstance(called_function.value, ast.Call) and isinstance(
                called_function.value.func, ast.Name
            ):
                full_method_call = (
                    f"{import_mappings.get(called_function.value.func.id)}.{called_function.attr}"
                )
                if full_method_call in target_calls:
                    methods_with_calls.add(node.name)
            elif isinstance(called_function.value, ast.Name) and called_function.value.id == "self":
                method_call_map[node.name].add(called_function.attr)

    # Second pass: Identify all methods that call the ones in `methods_with_calls`
    def find_calling_methods(method_name):
        for caller, callees in method_call_map.items():
            if method_name in callees and caller not in methods_with_calls:
                methods_with_calls.add(caller)
                find_calling_methods(caller)

    for method in list(methods_with_calls):
        find_calling_methods(method)

    return methods_with_calls
```



---

### Функция ID: 9

**Исходный код:**
```typescript
function repeat<P extends L.Update<P, 0, R>, R>(f: (...p: P) => R, again: (...p: F.NoInfer<P>) => boolean) {
  return (...p: P) => {
    // ts does not understand
    const pClone: any = [...p]

    while (again(...pClone)) {
      pClone[0] = f(...pClone)
    }

    return pClone[0] as R
  }
}
```



---

### Функция ID: 10

**Исходный код:**
```python
def _has_method(
    class_path: str,
    method_names: Iterable[str],
    class_registry: dict[str, dict[str, Any]],
    ignored_classes: list[str] | None = None,
) -> bool:
    """
    Determines if a class or its bases in the registry have any of the specified methods.

    :param class_path: The path of the class to check.
    :param method_names: A list of names of methods to search for.
    :param class_registry: A dictionary representing the class registry, where each key is a class name
        and the value is its metadata.
    :param ignored_classes: A list of classes to ignore when searching. If a base class has
        OL method but is ignored, the class_path will be treated as it would not have ol methods.
    :return: True if any of the specified methods are found in the class or its base classes; False otherwise.

    Example:
    >>> example_class_registry = {
    ...     "some.module.MyClass": {"methods": {"foo", "bar"}, "base_classes": ["BaseClass"]},
    ...     "another.module.BaseClass": {"methods": {"base_foo"}, "base_classes": []},
    ... }
    >>> _has_method("some.module.MyClass", ["foo"], example_class_registry)
    True
    >>> _has_method("some.module.MyClass", ["base_foo"], example_class_registry)
    True
    >>> _has_method("some.module.MyClass", ["not_a_method"], example_class_registry)
    False
    """
    ignored_classes = ignored_classes or []
    if class_path in ignored_classes:
        return False
    if class_path in class_registry:
        if any(method in class_registry[class_path]["methods"] for method in method_names):
            return True
        for base_name in class_registry[class_path]["base_classes"]:
            if base_name in ignored_classes:
                continue
            if _has_method(base_name, method_names, class_registry, ignored_classes):
                return True
    return False
```



---

### Функция ID: 11

**Исходный код:**
```python
def record(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: Callable[_P, Callable[[_R], _EncodedR]] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Record a function call result with custom encoding to both caches.

        This is a decorator that wraps a function to enable memoization
        with custom encoding/decoding logic. Results are stored in both
        the in-memory cache and the on-disk cache.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_encoder: Optional encoder factory for function results.
                                  Takes function parameters and returns an encoder
                                  function that converts R -> _EncodedR.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @persistent_memoizer.record(
                custom_params_encoder=my_param_encoder,
                custom_result_encoder=my_result_encoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to enable memoization.

            Args:
                fn: The function to wrap.

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, return the original function unchanged
            if not config.IS_CACHING_MODULE_ENABLED():
                return fn

            # Get the memory-cached version from the memoizer
            memory_record_fn = self._memoizer.record(
                custom_params_encoder, custom_result_encoder
            )(fn)

            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Call the original function and cache the result in both caches.

                Args:
                    *args: Positional arguments to pass to the function.
                    **kwargs: Keyword arguments to pass to the function.

                Returns:
                    The result of calling the original function.
                """
                # Call the memory-cached version (which calls fn and caches in memory)
                result = memory_record_fn(*args, **kwargs)

                # Also store in disk cache
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)

                # Get the cache entry from memory cache
                # We know it must be there since memory_record_fn just cached it
                cached_hit = self._memoizer._cache.get(cache_key)
                assert cached_hit, "Cache entry must exist in memory cache"
                cache_entry = cast(CacheEntry, cached_hit.value)

                # Store the full CacheEntry in disk cache for easier debugging
                pickled_entry: bytes = pickle.dumps(cache_entry)
                self._disk_cache.insert(cache_key, pickled_entry)

                return result

            return inner

        return wrapper
```



---

### Функция ID: 12

**Исходный код:**
```python
def memoize(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: Callable[_P, Callable[[_R], _EncodedR]] | None = None,
        custom_result_decoder: Callable[_P, Callable[[_EncodedR], _R]] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Memoize a function with record and replay functionality.

        This is a decorator that attempts to replay cached results first.
        If a cache miss occurs, it records the result by executing the wrapped function.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_encoder: Optional encoder factory for function results.
                                  Takes function parameters and returns an encoder
                                  function that converts R -> _EncodedR.
            custom_result_decoder: Optional decoder factory for cached results.
                                  Takes function parameters and returns a decoder
                                  function that converts _EncodedR -> R.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @memoizer.memoize(
                custom_params_encoder=my_param_encoder,
                custom_result_encoder=my_result_encoder_factory,
                custom_result_decoder=my_result_decoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to enable memoization with replay and record.

            Args:
                fn: The function to wrap.

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, return the original function unchanged
            if not config.IS_CACHING_MODULE_ENABLED():
                return fn

            # Create decorated versions using record and replay
            replay_fn = self.replay(
                custom_params_encoder,
                custom_result_decoder,
            )(fn)
            record_fn = self.record(
                custom_params_encoder,
                custom_result_encoder,
            )(fn)

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Attempt to replay from cache, or record on cache miss.

                Args:
                    *args: Positional arguments to pass to the function.
                    **kwargs: Keyword arguments to pass to the function.

                Returns:
                    The result from cache (if hit) or from executing the function (if miss).
                """
                # Try to replay first
                try:
                    return replay_fn(*args, **kwargs)
                except KeyError:
                    # Cache miss - record the result
                    return record_fn(*args, **kwargs)

            return inner

        return wrapper
```



---

### Функция ID: 13

**Исходный код:**
```python
def record(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: Callable[_P, Callable[[_R], _EncodedR]] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Record a function call result with custom encoding.

        This is a decorator that wraps a function to enable memoization
        with custom encoding/decoding logic.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_encoder: Optional encoder factory for function results.
                                  Takes function parameters and returns an encoder
                                  function that converts R -> _EncodedR.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @memoizer.record(
                custom_params_encoder=my_param_encoder,
                custom_result_encoder=my_result_encoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to enable memoization.

            Args:
                fn: The function to wrap.

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, return the original function unchanged
            if not config.IS_CACHING_MODULE_ENABLED():
                return fn

            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Call the original function and cache the result.

                Args:
                    *args: Positional arguments to pass to the function.
                    **kwargs: Keyword arguments to pass to the function.

                Returns:
                    The result of calling the original function.
                """
                # Call the function to compute the result
                result = fn(*args, **kwargs)

                # Generate cache key from parameters
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)

                # Encode params for human-readable dump
                if custom_params_encoder is not None:
                    encoded_params = custom_params_encoder(*args, **kwargs)
                else:
                    encoded_params = {
                        "args": args,
                        "kwargs": kwargs,
                    }

                # Encode the result if encoder is provided
                if custom_result_encoder is not None:
                    # Get the encoder function by calling the factory with params
                    encoder_fn = custom_result_encoder(*args, **kwargs)
                    encoded_result = encoder_fn(result)
                else:
                    encoded_result = result

                # Store CacheEntry in cache
                cache_entry = CacheEntry(
                    encoded_params=encoded_params,
                    encoded_result=encoded_result,
                )
                self._cache.insert(cache_key, cache_entry)

                # Return the original result (not the encoded version)
                return result

            return inner

        return wrapper
```



---

### Функция ID: 14

**Исходный код:**
```python
def replay(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_decoder: Callable[_P, Callable[[_EncodedR], _R]] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Replay a cached function result without executing the function.

        This is a decorator that retrieves cached results using a two-level
        cache strategy. It checks the in-memory cache first (fast), then
        falls back to the on-disk cache. If found on disk, the result is
        cached in memory for future access.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_decoder: Optional decoder factory for cached results.
                                  Takes function parameters and returns a decoder
                                  function that converts _EncodedR -> R.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @persistent_memoizer.replay(
                custom_params_encoder=my_param_encoder,
                custom_result_decoder=my_result_decoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to retrieve from cache.

            Args:
                fn: The function to wrap (not actually called).

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, always raise KeyError (cache miss)
            if not config.IS_CACHING_MODULE_ENABLED():

                def always_miss(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                    raise KeyError("Caching is disabled")

                return always_miss

            # Get the memory replay function
            memory_replay_fn = self._memoizer.replay(
                custom_params_encoder, custom_result_decoder
            )(fn)

            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Retrieve the cached result without calling the function.

                Checks memory cache first, then disk cache. Populates memory
                cache from disk on a disk hit.

                Args:
                    *args: Positional arguments to generate the cache key.
                    **kwargs: Keyword arguments to generate the cache key.

                Returns:
                    The cached result (decoded if decoder is provided).

                Raises:
                    KeyError: If no cached result exists for the given parameters.
                """
                # Try memory cache first via memoizer
                try:
                    return memory_replay_fn(*args, **kwargs)
                except KeyError:
                    pass  # Memory miss, check disk

                # Memory miss - check disk cache
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)
                disk_hit = self._disk_cache.get(cache_key)
                if disk_hit is not None:
                    # Disk cache hit - unpickle the CacheEntry
                    pickled_value = disk_hit.value
                    cache_entry = cast(CacheEntry, pickle.loads(pickled_value))

                    # Populate memory cache for future access
                    self._memoizer._cache.insert(cache_key, cache_entry)

                    # Decode and return
                    if custom_result_decoder is not None:
                        decoder_fn = custom_result_decoder(*args, **kwargs)
                        return decoder_fn(cast(_EncodedR, cache_entry.encoded_result))
                    return cast(_R, cache_entry.encoded_result)

                # Complete miss
                raise KeyError(f"No cached result found for key: {cache_key}")

            return inner

        return wrapper
```



---

### Функция ID: 15

**Исходный код:**
```python
def _create_ranges_from_split_points(
    split_points: list[int],
) -> list[tuple[int, int] | tuple[int, float]]:
    """Convert split points into ranges for autotuning dispatch.

    Example:
        split_points=[512, 2048]
        returns:
               [(1, 512), (513, 2048), (2049, float('inf'))]
    """
    ranges: list[tuple[int, int] | tuple[int, float]] = []
    start = 1

    for split_point in split_points:
        ranges.append((start, split_point))
        start = split_point + 1

    ranges.append((start, float("inf")))

    return ranges
```



---

### Функция ID: 16

**Исходный код:**
```python
def shared_task(*args, **kwargs):
    """Create shared task (decorator).

    This can be used by library authors to create tasks that'll work
    for any app environment.

    Returns:
        ~celery.local.Proxy: A proxy that always takes the task from the
        current apps task registry.

    Example:

        >>> from celery import Celery, shared_task
        >>> @shared_task
        ... def add(x, y):
        ...     return x + y
        ...
        >>> app1 = Celery(broker='amqp://')
        >>> add.app is app1
        True
        >>> app2 = Celery(broker='redis://')
        >>> add.app is app2
        True
    """
    def create_shared_task(**options):

        def __inner(fun):
            name = options.get('name')
            # Set as shared task so that unfinalized apps,
            # and future apps will register a copy of this task.
            _state.connect_on_app_finalize(
                lambda app: app._task_from_fun(fun, **options)
            )

            # Force all finalized apps to take this task as well.
            for app in _state._get_active_apps():
                if app.finalized:
                    with app._finalize_mutex:
                        app._task_from_fun(fun, **options)

            # Return a proxy that always gets the task from the current
            # apps task registry.
            def task_by_cons():
                app = _state.get_current_app()
                return app.tasks[
                    name or app.gen_task_name(fun.__name__, fun.__module__)
                ]
            return Proxy(task_by_cons)
        return __inner

    if len(args) == 1 and callable(args[0]):
        return create_shared_task(**kwargs)(args[0])
    return create_shared_task(*args, **kwargs)
```



---

### Функция ID: 17

**Исходный код:**
```typescript
function getProviderName(child: any): string[] {
  const providers = child?.providers || [];
  const names = providers.map((p: any) => p.name);
  return names || [];
}
```



---

### Функция ID: 18

**Исходный код:**
```typescript
function triggerUpdate() {
  const hooks = getHooksContextOrNull();
  // Rerun storyFn if updates were triggered synchronously, force rerender otherwise
  if (hooks != null && hooks.currentPhase !== 'NONE') {
    hooks.hasUpdates = true;
  } else {
    try {
      addons.getChannel().emit(FORCE_RE_RENDER);
    } catch (e) {
      logger.warn('State updates of Storybook preview hooks work only in browser');
    }
  }
}
```



---

### Функция ID: 19

**Исходный код:**
```typescript
function getGuardNames(child: AngularRoute, type: RouteGuard): string[] {
  const guards = child?.[type] || [];

  const names = guards.map((g: any) => getClassOrFunctionName(g));
  return names || [];
}
```



---

### Функция ID: 20

**Исходный код:**
```python
def __call__(self, num: float) -> str:
        """
        Formats a number in engineering notation, appending a letter
        representing the power of 1000 of the original number. Some examples:
        >>> format_eng = EngFormatter(accuracy=0, use_eng_prefix=True)
        >>> format_eng(0)
        ' 0'
        >>> format_eng = EngFormatter(accuracy=1, use_eng_prefix=True)
        >>> format_eng(1_000_000)
        ' 1.0M'
        >>> format_eng = EngFormatter(accuracy=2, use_eng_prefix=False)
        >>> format_eng("-1e-6")
        '-1.00E-06'

        @param num: the value to represent
        @type num: either a numeric value or a string that can be converted to
                   a numeric value (as per decimal.Decimal constructor)

        @return: engineering formatted string
        """
        dnum = Decimal(str(num))

        if Decimal.is_nan(dnum):
            return "NaN"

        if Decimal.is_infinite(dnum):
            return "inf"

        sign = 1

        if dnum < 0:  # pragma: no cover
            sign = -1
            dnum = -dnum

        if dnum != 0:
            pow10 = Decimal(int(math.floor(dnum.log10() / 3) * 3))
        else:
            pow10 = Decimal(0)

        pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))
        pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))
        int_pow10 = int(pow10)

        if self.use_eng_prefix:
            prefix = self.ENG_PREFIXES[int_pow10]
        elif int_pow10 < 0:
            prefix = f"E-{-int_pow10:02d}"
        else:
            prefix = f"E+{int_pow10:02d}"

        mant = sign * dnum / (10**pow10)

        if self.accuracy is None:  # pragma: no cover
            format_str = "{mant: g}{prefix}"
        else:
            format_str = f"{{mant: .{self.accuracy:d}f}}{{prefix}}"

        formatted = format_str.format(mant=mant, prefix=prefix)

        return formatted
```



---

### Функция ID: 21

**Исходный код:**
```python
def op_list(**configs):
    """Generate a list of ops organized in a specific format.
    It takes two parameters which are "attr_names" and "attr".
    attrs stores the name and function of operators.
    Args:
        configs: key-value pairs including the name and function of
        operators. attrs and attr_names must be present in configs.
    Return:
        a sequence of dictionaries which stores the name and function
        of ops in a specifal format
    Example:
    attrs = [
        ["abs", torch.abs],
        ["abs_", torch.abs_],
    ]
    attr_names = ["op_name", "op"].

    With those two examples,
    we will generate (({"op_name": "abs"}, {"op" : torch.abs}),
                      ({"op_name": "abs_"}, {"op" : torch.abs_}))
    """
    generated_configs = []
    if "attrs" not in configs:
        raise ValueError("Missing attrs in configs")
    for inputs in configs["attrs"]:
        tmp_result = {
            configs["attr_names"][i]: input_value
            for i, input_value in enumerate(inputs)
        }
        generated_configs.append(tmp_result)
    return generated_configs
```



---

### Функция ID: 22

**Исходный код:**
```javascript
function noConflict() {
      if (root._ === this) {
        root._ = oldDash;
      }
      return this;
    }
```



---

### Функция ID: 23

**Исходный код:**
```javascript
function wrapperReverse() {
      var value = this.__wrapped__;
      if (value instanceof LazyWrapper) {
        var wrapped = value;
        if (this.__actions__.length) {
          wrapped = new LazyWrapper(this);
        }
        wrapped = wrapped.reverse();
        wrapped.__actions__.push({
          'func': thru,
          'args': [reverse],
          'thisArg': undefined
        });
        return new LodashWrapper(wrapped, this.__chain__);
      }
      return this.thru(reverse);
    }
```



---

### Функция ID: 24

**Исходный код:**
```javascript
function castArray() {
      if (!arguments.length) {
        return [];
      }
      var value = arguments[0];
      return isArray(value) ? value : [value];
    }
```



---

### Функция ID: 25

**Исходный код:**
```python
def get(self, key: str):
        """
        Retrieve pandas object stored in file.

        Parameters
        ----------
        key : str
            Object to retrieve from file. Raises KeyError if not found.

        Returns
        -------
        object
            Same type as object stored in file.

        See Also
        --------
        HDFStore.get_node : Returns the node with the key.
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        with patch_pickle():
            # GH#31167 Without this patch, pickle doesn't know how to unpickle
            #  old DateOffset objects now that they are cdef classes.
            group = self.get_node(key)
            if group is None:
                raise KeyError(f"No object named {key} in the file")
            return self._read_group(group)
```



---

### Функция ID: 26

**Исходный код:**
```python
def _simple_json_normalize(
    ds: dict | list[dict],
    sep: str = ".",
) -> dict | list[dict] | Any:
    """
    An optimized basic json_normalize

    Converts a nested dict into a flat dict ("record"), unlike
    json_normalize and nested_to_record it doesn't do anything clever.
    But for the most basic use cases it enhances performance.
    E.g. pd.json_normalize(data)

    Parameters
    ----------
    ds : dict or list of dicts
    sep : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    frame : DataFrame
    d - dict or list of dicts, matching `normalized_json_object`

    Examples
    --------
    >>> _simple_json_normalize(
    ...     {
    ...         "flat1": 1,
    ...         "dict1": {"c": 1, "d": 2},
    ...         "nested": {"e": {"c": 1, "d": 2}, "d": 2},
    ...     }
    ... )
    {\
'flat1': 1, \
'dict1.c': 1, \
'dict1.d': 2, \
'nested.e.c': 1, \
'nested.e.d': 2, \
'nested.d': 2\
}

    """
    normalized_json_object = {}
    # expect a dictionary, as most jsons are. However, lists are perfectly valid
    if isinstance(ds, dict):
        normalized_json_object = _normalize_json_ordered(data=ds, separator=sep)
    elif isinstance(ds, list):
        normalized_json_list = [_simple_json_normalize(row, sep=sep) for row in ds]
        return normalized_json_list
    return normalized_json_object
```



---

### Функция ID: 27

**Исходный код:**
```python
def update_dtype(self, dtype) -> SparseDtype:
        """
        Convert the SparseDtype to a new dtype.

        This takes care of converting the ``fill_value``.

        Parameters
        ----------
        dtype : Union[str, numpy.dtype, SparseDtype]
            The new dtype to use.

            * For a SparseDtype, it is simply returned
            * For a NumPy dtype (or str), the current fill value
              is converted to the new dtype, and a SparseDtype
              with `dtype` and the new fill value is returned.

        Returns
        -------
        SparseDtype
            A new SparseDtype with the correct `dtype` and fill value
            for that `dtype`.

        Raises
        ------
        ValueError
            When the current fill value cannot be converted to the
            new `dtype` (e.g. trying to convert ``np.nan`` to an
            integer dtype).


        Examples
        --------
        >>> SparseDtype(int, 0).update_dtype(float)
        Sparse[float64, 0.0]

        >>> SparseDtype(int, 1).update_dtype(SparseDtype(float, np.nan))
        Sparse[float64, nan]
        """
        from pandas.core.dtypes.astype import astype_array
        from pandas.core.dtypes.common import pandas_dtype

        cls = type(self)
        dtype = pandas_dtype(dtype)

        if not isinstance(dtype, cls):
            if not isinstance(dtype, np.dtype):
                raise TypeError("sparse arrays of extension dtypes not supported")

            fv_asarray = np.atleast_1d(np.array(self.fill_value))
            fvarr = astype_array(fv_asarray, dtype)
            # NB: not fv_0d.item(), as that casts dt64->int
            fill_value = fvarr[0]
            dtype = cls(dtype, fill_value=fill_value)

        return dtype
```



---

### Функция ID: 28

**Исходный код:**
```python
def to_frame(self, name: Hashable = lib.no_default) -> DataFrame:
        """
        Convert Series to DataFrame.

        Parameters
        ----------
        name : object, optional
            The passed name should substitute for the series name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame representation of Series.

        See Also
        --------
        Series.to_dict : Convert Series to dict object.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c"], name="vals")
        >>> s.to_frame()
          vals
        0    a
        1    b
        2    c
        """
        columns: Index
        if name is lib.no_default:
            name = self.name
            if name is None:
                # default to [0], same as we would get with DataFrame(self)
                columns = default_index(1)
            else:
                columns = Index([name])
        else:
            columns = Index([name])

        mgr = self._mgr.to_2d_mgr(columns)
        df = self._constructor_expanddim_from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(self, method="to_frame")
```



---

### Функция ID: 29

**Исходный код:**
```python
def slice_indexer(
        self,
        start: Hashable | None = None,
        end: Hashable | None = None,
        step: int | None = None,
    ) -> slice:
        """
        Compute the slice indexer for input labels and step.

        Index needs to be ordered and unique.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning.
        end : label, default None
            If None, defaults to the end.
        step : int, default None
            If None, defaults to 1.

        Returns
        -------
        slice
            A slice object.

        Raises
        ------
        KeyError : If key does not exist, or key is not unique and index is
            not ordered.

        See Also
        --------
        Index.slice_locs : Computes slice locations for input labels.
        Index.get_slice_bound : Retrieves slice bound that corresponds to given label.

        Notes
        -----
        This function assumes that the data is sorted, so use at your own peril.

        Examples
        --------
        This is a method on all index types. For example you can do:

        >>> idx = pd.Index(list("abcd"))
        >>> idx.slice_indexer(start="b", end="c")
        slice(1, 3, None)

        >>> idx = pd.MultiIndex.from_arrays([list("abcd"), list("efgh")])
        >>> idx.slice_indexer(start="b", end=("c", "g"))
        slice(1, 3, None)
        """
        start_slice, end_slice = self.slice_locs(start, end, step=step)

        # return a slice
        if not is_scalar(start_slice):
            raise AssertionError("Start slice bound is non-scalar")
        if not is_scalar(end_slice):
            raise AssertionError("End slice bound is non-scalar")

        return slice(start_slice, end_slice, step)
```



---

### Функция ID: 30

**Исходный код:**
```java
private boolean is(Type formalType, TypeVariable<?> declaration) {
    if (runtimeType.equals(formalType)) {
      return true;
    }
    if (formalType instanceof WildcardType) {
      WildcardType your = canonicalizeWildcardType(declaration, (WildcardType) formalType);
      // if "formalType" is <? extends Foo>, "this" can be:
      // Foo, SubFoo, <? extends Foo>, <? extends SubFoo>, <T extends Foo> or
      // <T extends SubFoo>.
      // if "formalType" is <? super Foo>, "this" can be:
      // Foo, SuperFoo, <? super Foo> or <? super SuperFoo>.
      return every(your.getUpperBounds()).isSupertypeOf(runtimeType)
          && every(your.getLowerBounds()).isSubtypeOf(runtimeType);
    }
    return canonicalizeWildcardsInType(runtimeType).equals(canonicalizeWildcardsInType(formalType));
  }
```



---

### Функция ID: 31

**Исходный код:**
```python
def remove_repeating(substr: str, s: str) -> str:
    """Remove repeating module names from string.

    Arguments:
        task_name (str): Task name (full path including module),
            to use as the basis for removing module names.
        s (str): The string we want to work on.

    Example:

        >>> _shorten_names(
        ...    'x.tasks.add',
        ...    'x.tasks.add(2, 2) | x.tasks.add(4) | x.tasks.mul(8)',
        ... )
        'x.tasks.add(2, 2) | add(4) | mul(8)'
    """
    # find the first occurrence of substr in the string.
    index = s.find(substr)
    if index >= 0:
        return ''.join([
            # leave the first occurrence of substr untouched.
            s[:index + len(substr)],
            # strip seen substr from the rest of the string.
            s[index + len(substr):].replace(substr, ''),
        ])
    return s
```



---

### Функция ID: 32

**Исходный код:**
```python
def setdiff1d(
    x1: Array | complex,
    x2: Array | complex,
    /,
    *,
    assume_unique: bool = False,
    xp: ModuleType | None = None,
) -> Array:
    """
    Find the set difference of two arrays.

    Return the unique values in `x1` that are not in `x2`.

    Parameters
    ----------
    x1 : array | int | float | complex | bool
        Input array.
    x2 : array
        Input comparison array.
    assume_unique : bool
        If ``True``, the input arrays are both assumed to be unique, which
        can speed up the calculation. Default is ``False``.
    xp : array_namespace, optional
        The standard-compatible namespace for `x1` and `x2`. Default: infer.

    Returns
    -------
    array
        1D array of values in `x1` that are not in `x2`. The result
        is sorted when `assume_unique` is ``False``, but otherwise only sorted
        if the input is sorted.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx

    >>> x1 = xp.asarray([1, 2, 3, 2, 4, 1])
    >>> x2 = xp.asarray([3, 4, 5, 6])
    >>> xpx.setdiff1d(x1, x2, xp=xp)
    Array([1, 2], dtype=array_api_strict.int64)
    """
    if xp is None:
        xp = array_namespace(x1, x2)
    # https://github.com/microsoft/pyright/issues/10103
    x1_, x2_ = asarrays(x1, x2, xp=xp)

    if assume_unique:
        x1_ = xp.reshape(x1_, (-1,))
        x2_ = xp.reshape(x2_, (-1,))
    else:
        x1_ = xp.unique_values(x1_)
        x2_ = xp.unique_values(x2_)

    return x1_[_helpers.in1d(x1_, x2_, assume_unique=True, invert=True, xp=xp)]
```



---

### Функция ID: 33

**Исходный код:**
```python
def delete(
        self, loc: int | np.integer | list[int] | npt.NDArray[np.integer]
    ) -> Self:
        """
        Make new Index with passed location(-s) deleted.

        Parameters
        ----------
        loc : int or list of int
            Location of item(-s) which will be deleted.
            Use a list of locations to delete more than one value at the same time.

        Returns
        -------
        Index
            Will be same type as self, except for RangeIndex.

        See Also
        --------
        numpy.delete : Delete any rows and column from NumPy array (ndarray).

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.delete(1)
        Index(['a', 'c'], dtype='str')

        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.delete([0, 2])
        Index(['b'], dtype='str')
        """
        values = self._values
        res_values: ArrayLike
        if isinstance(values, np.ndarray):
            # TODO(__array_function__): special casing will be unnecessary
            res_values = np.delete(values, loc)
        else:
            res_values = values.delete(loc)

        # _constructor so RangeIndex-> Index with an int64 dtype
        return self._constructor._simple_new(res_values, name=self.name)
```



---

### Функция ID: 34

**Исходный код:**
```python
def create_iter_data_given_by(
    data: DataFrame, kind: str = "hist"
) -> dict[Hashable, DataFrame | Series]:
    """
    Create data for iteration given `by` is assigned or not, and it is only
    used in both hist and boxplot.

    If `by` is assigned, return a dictionary of DataFrames in which the key of
    dictionary is the values in groups.
    If `by` is not assigned, return input as is, and this preserves current
    status of iter_data.

    Parameters
    ----------
    data : reformatted grouped data from `_compute_plot_data` method.
    kind : str, plot kind. This function is only used for `hist` and `box` plots.

    Returns
    -------
    iter_data : DataFrame or Dictionary of DataFrames

    Examples
    --------
    If `by` is assigned:

    >>> import numpy as np
    >>> tuples = [("h1", "a"), ("h1", "b"), ("h2", "a"), ("h2", "b")]
    >>> mi = pd.MultiIndex.from_tuples(tuples)
    >>> value = [[1, 3, np.nan, np.nan], [3, 4, np.nan, np.nan], [np.nan, np.nan, 5, 6]]
    >>> data = pd.DataFrame(value, columns=mi)
    >>> create_iter_data_given_by(data)
    {'h1':     h1
         a    b
    0  1.0  3.0
    1  3.0  4.0
    2  NaN  NaN, 'h2':     h2
         a    b
    0  NaN  NaN
    1  NaN  NaN
    2  5.0  6.0}
    """

    # For `hist` plot, before transformation, the values in level 0 are values
    # in groups and subplot titles, and later used for column subselection and
    # iteration; For `box` plot, values in level 1 are column names to show,
    # and are used for iteration and as subplots titles.
    if kind == "hist":
        level = 0
    else:
        level = 1

    # Select sub-columns based on the value of level of MI, and if `by` is
    # assigned, data must be a MI DataFrame
    assert isinstance(data.columns, MultiIndex)
    return {
        col: data.loc[:, data.columns.get_level_values(level) == col]
        for col in data.columns.levels[level]
    }
```



---

### Функция ID: 35

**Исходный код:**
```python
def to_string(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        na_rep: str = "NaN",
        float_format: str | None = None,
        header: bool = True,
        index: bool = True,
        length: bool = False,
        dtype: bool = False,
        name: bool = False,
        max_rows: int | None = None,
        min_rows: int | None = None,
    ) -> str | None:
        """
        Render a string representation of the Series.

        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        na_rep : str, optional
            String representation of NaN to use, default 'NaN'.
        float_format : one-parameter function, optional
            Formatter function to apply to columns' elements if they are
            floats, default None.
        header : bool, default True
            Add the Series header (index name).
        index : bool, optional
            Add index (row) labels, default True.
        length : bool, default False
            Add the Series length.
        dtype : bool, default False
            Add the Series dtype.
        name : bool, default False
            Add the Series name if not None.
        max_rows : int, optional
            Maximum number of rows to show before truncating. If None, show
            all.
        min_rows : int, optional
            The number of rows to display in a truncated repr (when number
            of rows is above `max_rows`).

        Returns
        -------
        str or None
            String representation of Series if ``buf=None``, otherwise None.

        See Also
        --------
        Series.to_dict : Convert Series to dict object.
        Series.to_frame : Convert Series to DataFrame object.
        Series.to_markdown : Print Series in Markdown-friendly format.
        Series.to_timestamp : Cast to DatetimeIndex of Timestamps.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3]).to_string()
        >>> ser
        '0    1\\n1    2\\n2    3'
        """
        formatter = fmt.SeriesFormatter(
            self,
            name=name,
            length=length,
            header=header,
            index=index,
            dtype=dtype,
            na_rep=na_rep,
            float_format=float_format,
            min_rows=min_rows,
            max_rows=max_rows,
        )
        result = formatter.to_string()

        # catch contract violations
        if not isinstance(result, str):
            raise AssertionError(
                "result must be of type str, type "
                f"of result is {type(result).__name__!r}"
            )

        if buf is None:
            return result
        else:
            if hasattr(buf, "write"):
                buf.write(result)
            else:
                with open(buf, "w", encoding="utf-8") as f:
                    f.write(result)
        return None
```



---

### Функция ID: 36

**Исходный код:**
```python
def __getitem__(self, key: int | slice) -> Series:
        """
        Index or slice lists in the Series.

        Parameters
        ----------
        key : int | slice
            Index or slice of indices to access from each list.

        Returns
        -------
        pandas.Series
            The list at requested index.

        See Also
        --------
        ListAccessor.flatten : Flatten list values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(pa.int64())),
        ... )
        >>> s.list[0]
        0    1
        1    3
        dtype: int64[pyarrow]
        """
        from pandas import Series

        if isinstance(key, int):
            # TODO: Support negative key but pyarrow does not allow
            # element index to be an array.
            # if key < 0:
            #     key = pc.add(key, pc.list_value_length(self._pa_array))
            element = pc.list_element(self._pa_array, key)
            return Series(
                element,
                dtype=ArrowDtype(element.type),
                index=self._data.index,
                name=self._data.name,
            )
        elif isinstance(key, slice):
            # TODO: Support negative start/stop/step, ideally this would be added
            # upstream in pyarrow.
            start, stop, step = key.start, key.stop, key.step
            if start is None:
                # TODO: When adding negative step support
                #  this should be setto last element of array
                # when step is negative.
                start = 0
            if step is None:
                step = 1
            sliced = pc.list_slice(self._pa_array, start, stop, step)
            return Series(
                sliced,
                dtype=ArrowDtype(sliced.type),
                index=self._data.index,
                name=self._data.name,
            )
        else:
            raise ValueError(f"key must be an int or slice, got {type(key).__name__}")
```



---

### Функция ID: 37

**Исходный код:**
```python
def send_email_smtp(
    to: str | Iterable[str],
    subject: str,
    html_content: str,
    files: list[str] | None = None,
    dryrun: bool = False,
    cc: str | Iterable[str] | None = None,
    bcc: str | Iterable[str] | None = None,
    mime_subtype: str = "mixed",
    mime_charset: str = "utf-8",
    conn_id: str = "smtp_default",
    from_email: str | None = None,
    custom_headers: dict[str, Any] | None = None,
    **kwargs,
) -> None:
    """
    Send an email with html content.

    :param to: Recipient email address or list of addresses.
    :param subject: Email subject.
    :param html_content: Email body in HTML format.
    :param files: List of file paths to attach to the email.
    :param dryrun: If True, the email will not be sent, but all other actions will be performed.
    :param cc: Carbon copy recipient email address or list of addresses.
    :param bcc: Blind carbon copy recipient email address or list of addresses.
    :param mime_subtype: MIME subtype of the email.
    :param mime_charset: MIME charset of the email.
    :param conn_id: Connection ID of the SMTP server.
    :param from_email: Sender email address.
    :param custom_headers: Dictionary of custom headers to include in the email.
    :param kwargs: Additional keyword arguments.

    >>> send_email("test@example.com", "foo", "<b>Foo</b> bar", ["/dev/null"], dryrun=True)
    """
    smtp_mail_from = conf.get("smtp", "SMTP_MAIL_FROM")

    if smtp_mail_from is not None:
        mail_from = smtp_mail_from
    else:
        if from_email is None:
            raise ValueError(
                "You should set from email - either by smtp/smtp_mail_from config or `from_email` parameter"
            )
        mail_from = from_email

    msg, recipients = build_mime_message(
        mail_from=mail_from,
        to=to,
        subject=subject,
        html_content=html_content,
        files=files,
        cc=cc,
        bcc=bcc,
        mime_subtype=mime_subtype,
        mime_charset=mime_charset,
        custom_headers=custom_headers,
    )

    send_mime_email(e_from=mail_from, e_to=recipients, mime_msg=msg, conn_id=conn_id, dryrun=dryrun)
```



---

### Функция ID: 38

**Исходный код:**
```python
def __or__(self, other):
        """Chaining operator.

        Example:
            >>> add.s(2, 2) | add.s(4) | add.s(8)

        Returns:
            chain: Constructs a :class:`~celery.canvas.chain` of the given signatures.
        """
        if isinstance(other, _chain):
            # task | chain -> chain
            return _chain(seq_concat_seq(
                (self,), other.unchain_tasks()), app=self._app)
        elif isinstance(other, group):
            # unroll group with one member
            other = maybe_unroll_group(other)
            # task | group() -> chain
            return _chain(self, other, app=self.app)
        elif isinstance(other, Signature):
            # task | task -> chain
            return _chain(self, other, app=self._app)
        return NotImplemented
```



---

### Функция ID: 39

**Исходный код:**
```python
def match(
        self,
        pat: str | re.Pattern,
        case: bool | lib.NoDefault = lib.no_default,
        flags: int | lib.NoDefault = lib.no_default,
        na=lib.no_default,
    ):
        """
        Determine if each string starts with a match of a regular expression.

        Determines whether each string in the Series or Index starts with a
        match to a specified regular expression. This function is especially
        useful for validating prefixes, such as ensuring that codes, tags, or
        identifiers begin with a specific pattern.

        Parameters
        ----------
        pat : str or compiled regex
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Regex module flags, e.g. re.IGNORECASE.
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For the ``"str"`` dtype, ``False`` is used. For object
            dtype, ``numpy.nan`` is used. For the nullable ``StringDtype``,
            ``pandas.NA`` is used.

        Returns
        -------
        Series/Index/array of boolean values
            A Series, Index, or array of boolean values indicating whether the start
            of each string matches the pattern. The result will be of the same type
            as the input.

        See Also
        --------
        fullmatch : Stricter matching that requires the entire string to match.
        contains : Analogous, but less strict, relying on re.search instead of
            re.match.
        extract : Extract matched groups.

        Examples
        --------
        >>> ser = pd.Series(["horse", "eagle", "donkey"])
        >>> ser.str.match("e")
        0   False
        1   True
        2   False
        dtype: bool
        """
        if flags is not lib.no_default:
            # pat.flags will have re.U regardless, so we need to add it here
            # before checking for a match
            flags = flags | re.U
            if is_re(pat):
                if pat.flags != flags:
                    raise ValueError(
                        "Cannot both specify 'flags' and pass a compiled regexp "
                        "object with conflicting flags"
                    )
            else:
                pat = re.compile(pat, flags=flags)
            # set flags=0 to ensure that when we call
            #  re.compile(pat, flags=flags) the constructor does not raise.
            flags = 0
        else:
            flags = 0

        if case is lib.no_default:
            if is_re(pat):
                case = not bool(pat.flags & re.IGNORECASE)
            else:
                # Case-sensitive default
                case = True
        elif is_re(pat):
            implicit_case = not bool(pat.flags & re.IGNORECASE)
            if implicit_case != case:
                # GH#62240
                raise ValueError(
                    "Cannot both specify 'case' and pass a compiled regexp "
                    "object with conflicting case-sensitivity"
                )

        result = self._data.array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)
```



---

### Функция ID: 40

**Исходный код:**
```python
def get_group(self, name) -> DataFrame | Series:
        """
        Construct DataFrame from group with provided name.

        Parameters
        ----------
        name : object
            The name of the group to get as a DataFrame.

        Returns
        -------
        Series or DataFrame
            Get the respective Series or DataFrame corresponding to the group provided.

        See Also
        --------
        DataFrameGroupBy.groups: Dictionary representation of the groupings formed
            during a groupby operation.
        DataFrameGroupBy.indices: Provides a mapping of group rows to positions
            of the elements.
        SeriesGroupBy.groups: Dictionary representation of the groupings formed
            during a groupby operation.
        SeriesGroupBy.indices: Provides a mapping of group rows to positions
            of the elements.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).get_group("a")
        a    1
        a    2
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["owl", "toucan", "eagle"]
        ... )
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby(by=["a"]).get_group((1,))
                a  b  c
        owl     1  2  3
        toucan  1  5  6

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").get_group("2023-01-01")
        2023-01-01    1
        2023-01-15    2
        dtype: int64
        """
        keys = self.keys
        level = self.level
        # mypy doesn't recognize level/keys as being sized when passed to len
        if (is_list_like(level) and len(level) == 1) or (  # type: ignore[arg-type]
            is_list_like(keys) and len(keys) == 1  # type: ignore[arg-type]
        ):
            # GH#25971
            if isinstance(name, tuple) and len(name) == 1:
                name = name[0]
            else:
                raise KeyError(name)

        inds = self._get_index(name)
        if not len(inds):
            raise KeyError(name)
        return self._selected_obj.iloc[inds]
```



---

### Функция ID: 41

**Исходный код:**
```python
def quantile(
        self,
        q: float | Sequence[float] | AnyArrayLike = 0.5,
        interpolation: QuantileInterpolation = "linear",
    ) -> float | Series:
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            The quantile(s) to compute, which can lie in range: 0 <= q <= 1.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

                * linear: `i + (j - i) * (x-i)/(j-i)`, where `(x-i)/(j-i)` is
                  the fractional part of the index surrounded by `i > j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.

        Returns
        -------
        float or Series
            If ``q`` is an array, a Series will be returned where the
            index is ``q`` and the values are the quantiles, otherwise
            a float will be returned.

        See Also
        --------
        core.window.Rolling.quantile : Calculate the rolling quantile.
        numpy.percentile : Returns the q-th percentile(s) of the array elements.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.quantile(0.5)
        2.5
        >>> s.quantile([0.25, 0.5, 0.75])
        0.25    1.75
        0.50    2.50
        0.75    3.25
        dtype: float64
        """
        validate_percentile(q)

        # We dispatch to DataFrame so that core.internals only has to worry
        #  about 2D cases.
        df = self.to_frame()

        result = df.quantile(q=q, interpolation=interpolation, numeric_only=False)
        if result.ndim == 2:
            result = result.iloc[:, 0]

        if is_list_like(q):
            result.name = self.name
            idx = Index(q, dtype=np.float64)
            return self._constructor(result, index=idx, name=self.name)
        else:
            # scalar
            return result.iloc[0]
```



---

### Функция ID: 42

**Исходный код:**
```python
def mode(self, dropna: bool = True) -> Series:
        """
        Return the mode(s) of the Series.

        The mode is the value that appears most often. There can be multiple modes.

        Always returns Series even if only one value is returned.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NaN/NaT.

        Returns
        -------
        Series
            Modes of the Series in sorted order.

        See Also
        --------
        numpy.mode : Equivalent numpy function for computing median.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.

        Examples
        --------
        >>> s = pd.Series([2, 4, 2, 2, 4, None])
        >>> s.mode()
        0    2.0
        dtype: float64

        More than one mode:

        >>> s = pd.Series([2, 4, 8, 2, 4, None])
        >>> s.mode()
        0    2.0
        1    4.0
        dtype: float64

        With and without considering null value:

        >>> s = pd.Series([2, 4, None, None, 4, None])
        >>> s.mode(dropna=False)
        0   NaN
        dtype: float64
        >>> s = pd.Series([2, 4, None, None, 4, None])
        >>> s.mode()
        0    4.0
        dtype: float64
        """
        # TODO: Add option for bins like value_counts()
        values = self._values
        if isinstance(values, np.ndarray):
            res_values, _ = algorithms.mode(values, dropna=dropna)
        else:
            res_values = values._mode(dropna=dropna)

        # Ensure index is type stable (should always use int index)
        return self._constructor(
            res_values,
            index=range(len(res_values)),
            name=self.name,
            copy=False,
            dtype=self.dtype,
        ).__finalize__(self, method="mode")
```



---

### Функция ID: 43

**Исходный код:**
```python
def write_file(self) -> None:
        """
        Export DataFrame object to Stata dta format.

        This method writes the contents of a pandas DataFrame to a `.dta` file
        compatible with Stata. It includes features for handling value labels,
        variable types, and metadata like timestamps and data labels. The output
        file can then be read and used in Stata or other compatible statistical
        tools.

        See Also
        --------
        read_stata : Read Stata file into DataFrame.
        DataFrame.to_stata : Export DataFrame object to Stata dta format.
        io.stata.StataWriter : A class for writing Stata binary dta files.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "fully_labelled": [1, 2, 3, 3, 1],
        ...         "partially_labelled": [1.0, 2.0, np.nan, 9.0, np.nan],
        ...         "Y": [7, 7, 9, 8, 10],
        ...         "Z": pd.Categorical(["j", "k", "l", "k", "j"]),
        ...     }
        ... )
        >>> path = "/My_path/filename.dta"
        >>> labels = {
        ...     "fully_labelled": {1: "one", 2: "two", 3: "three"},
        ...     "partially_labelled": {1.0: "one", 2.0: "two"},
        ... }
        >>> writer = pd.io.stata.StataWriter(
        ...     path, df, value_labels=labels
        ... )  # doctest: +SKIP
        >>> writer.write_file()  # doctest: +SKIP
        >>> df = pd.read_stata(path)  # doctest: +SKIP
        >>> df  # doctest: +SKIP
            index fully_labelled  partially_labeled  Y  Z
        0       0            one                one  7  j
        1       1            two                two  7  k
        2       2          three                NaN  9  l
        3       3          three                9.0  8  k
        4       4            one                NaN 10  j
        """
        with get_handle(
            self._fname,
            "wb",
            compression=self._compression,
            is_text=False,
            storage_options=self.storage_options,
        ) as self.handles:
            if self.handles.compression["method"] is not None:
                # ZipFile creates a file (with the same name) for each write call.
                # Write it first into a buffer and then write the buffer to the ZipFile.
                self._output_file, self.handles.handle = self.handles.handle, BytesIO()
                self.handles.created_handles.append(self.handles.handle)

            try:
                self._write_header(
                    data_label=self._data_label, time_stamp=self._time_stamp
                )
                self._write_map()
                self._write_variable_types()
                self._write_varnames()
                self._write_sortlist()
                self._write_formats()
                self._write_value_label_names()
                self._write_variable_labels()
                self._write_expansion_fields()
                self._write_characteristics()
                records = self._prepare_data()
                self._write_data(records)
                self._write_strls()
                self._write_value_labels()
                self._write_file_close_tag()
                self._write_map()
                self._close()
            except Exception as exc:
                self.handles.close()
                if isinstance(self._fname, (str, os.PathLike)) and os.path.isfile(
                    self._fname
                ):
                    try:
                        os.unlink(self._fname)
                    except OSError:
                        warnings.warn(
                            f"This save was not successful but {self._fname} could not "
                            "be deleted. This file is not valid.",
                            ResourceWarning,
                            stacklevel=find_stack_level(),
                        )
                raise exc
```



---

### Функция ID: 44

**Исходный код:**
```python
def insert(
        self,
        loc: int,
        column: Hashable,
        value: object,
        allow_duplicates: bool | lib.NoDefault = lib.no_default,
    ) -> None:
        """
        Insert column into DataFrame at specified location.

        Raises a ValueError if `column` is already contained in the DataFrame,
        unless `allow_duplicates` is set to True.

        Parameters
        ----------
        loc : int
            Insertion index. Must verify 0 <= loc <= len(columns).
        column : str, number, or hashable object
            Label of the inserted column.
        value : Scalar, Series, or array-like
            Content of the inserted column.
        allow_duplicates : bool, optional, default lib.no_default
            Allow duplicate column labels to be created.

        See Also
        --------
        Index.insert : Insert new item by index.

        Examples
        --------
        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df
           col1  col2
        0     1     3
        1     2     4
        >>> df.insert(1, "newcol", [99, 99])
        >>> df
           col1  newcol  col2
        0     1      99     3
        1     2      99     4
        >>> df.insert(0, "col1", [100, 100], allow_duplicates=True)
        >>> df
           col1  col1  newcol  col2
        0   100     1      99     3
        1   100     2      99     4

        Notice that pandas uses index alignment in case of `value` from type `Series`:

        >>> df.insert(0, "col0", pd.Series([5, 6], index=[1, 2]))
        >>> df
           col0  col1  col1  newcol  col2
        0   NaN   100     1      99     3
        1   5.0   100     2      99     4
        """
        if allow_duplicates is lib.no_default:
            allow_duplicates = False
        if allow_duplicates and not self.flags.allows_duplicate_labels:
            raise ValueError(
                "Cannot specify 'allow_duplicates=True' when "
                "'self.flags.allows_duplicate_labels' is False."
            )
        if not allow_duplicates and column in self.columns:
            # Should this be a different kind of error??
            raise ValueError(f"cannot insert {column}, already exists")
        if not is_integer(loc):
            raise TypeError("loc must be int")
        # convert non stdlib ints to satisfy typing checks
        loc = int(loc)
        if isinstance(value, DataFrame) and len(value.columns) > 1:
            raise ValueError(
                f"Expected a one-dimensional object, got a DataFrame with "
                f"{len(value.columns)} columns instead."
            )
        elif isinstance(value, DataFrame):
            value = value.iloc[:, 0]

        value, refs = self._sanitize_column(value)
        self._mgr.insert(loc, column, value, refs=refs)
```



---

### Функция ID: 45

**Исходный код:**
```python
def strings_with_wrong_placed_whitespace(
    file_obj: IO[str],
) -> Iterable[tuple[int, str]]:
    """
    Test case for leading spaces in concated strings.

    For example:

    >>> rule = "We want the space at the end of the line, not at the beginning"

    Instead of:

    >>> rule = "We want the space at the end of the line, not at the beginning"

    Parameters
    ----------
    file_obj : IO
        File-like object containing the Python code to validate.

    Yields
    ------
    line_number : int
        Line number of unconcatenated string.
    msg : str
        Explanation of the error.
    """

    def has_wrong_whitespace(first_line: str, second_line: str) -> bool:
        """
        Checking if the two lines are mattching the unwanted pattern.

        Parameters
        ----------
        first_line : str
            First line to check.
        second_line : str
            Second line to check.

        Returns
        -------
        bool
            True if the two received string match, an unwanted pattern.

        Notes
        -----
        The unwanted pattern that we are trying to catch is if the spaces in
        a string that is concatenated over multiple lines are placed at the
        end of each string, unless this string is ending with a
        newline character (\n).

        For example, this is bad:

        >>> rule = "We want the space at the end of the line, not at the beginning"

        And what we want is:

        >>> rule = "We want the space at the end of the line, not at the beginning"

        And if the string is ending with a new line character (\n) we
        do not want any trailing whitespaces after it.

        For example, this is bad:

        >>> rule = (
        ...     "We want the space at the begging of "
        ...     "the line if the previous line is ending with a \n "
        ...     "not at the end, like always"
        ... )

        And what we do want is:

        >>> rule = (
        ...     "We want the space at the begging of "
        ...     "the line if the previous line is ending with a \n"
        ...     " not at the end, like always"
        ... )
        """
        if first_line.endswith(r"\n"):
            return False
        elif first_line.startswith("  ") or second_line.startswith("  "):
            return False
        elif first_line.endswith("  ") or second_line.endswith("  "):
            return False
        elif (not first_line.endswith(" ")) and second_line.startswith(" "):
            return True
        return False

    tokens: list = list(tokenize.generate_tokens(file_obj.readline))

    for first_token, second_token, third_token in zip(
        tokens, tokens[1:], tokens[2:], strict=False
    ):
        # Checking if we are in a block of concated string
        if (
            first_token.type == third_token.type == token.STRING
            and second_token.type == token.NL
        ):
            # Striping the quotes, with the string literal prefix
            first_string: str = first_token.string[
                _get_literal_string_prefix_len(first_token.string) + 1 : -1
            ]
            second_string: str = third_token.string[
                _get_literal_string_prefix_len(third_token.string) + 1 : -1
            ]

            if has_wrong_whitespace(first_string, second_string):
                yield (
                    third_token.start[0],
                    (
                        "String has a space at the beginning instead "
                        "of the end of the previous string."
                    ),
                )
```



---

### Функция ID: 46

**Исходный код:**
```python
def combine_first(self, other) -> Series:
        """
        Update null elements with value in the same location in 'other'.

        Combine two Series objects by filling null values in one Series with
        non-null values from the other Series. Result index will be the union
        of the two indexes.

        Parameters
        ----------
        other : Series
            The value(s) to be used for filling null values.

        Returns
        -------
        Series
            The result of combining the provided Series with the other object.

        See Also
        --------
        Series.combine : Perform element-wise operation on two Series
            using a given function.

        Examples
        --------
        >>> s1 = pd.Series([1, np.nan])
        >>> s2 = pd.Series([3, 4, 5])
        >>> s1.combine_first(s2)
        0    1.0
        1    4.0
        2    5.0
        dtype: float64

        Null values still persist if the location of that null value
        does not exist in `other`

        >>> s1 = pd.Series({"falcon": np.nan, "eagle": 160.0})
        >>> s2 = pd.Series({"eagle": 200.0, "duck": 30.0})
        >>> s1.combine_first(s2)
        duck       30.0
        eagle     160.0
        falcon      NaN
        dtype: float64
        """
        from pandas.core.reshape.concat import concat

        if self.dtype == other.dtype:
            if self.index.equals(other.index):
                return self.mask(self.isna(), other)

        new_index = self.index.union(other.index)

        this = self
        # identify the index subset to keep for each series
        keep_other = other.index.difference(this.index[notna(this)])
        keep_this = this.index.difference(keep_other)

        this = this.reindex(keep_this)
        other = other.reindex(keep_other)

        if this.dtype.kind == "M" and other.dtype.kind != "M":
            # TODO: try to match resos?
            other = to_datetime(other)
            warnings.warn(
                # GH#62931
                "Silently casting non-datetime 'other' to datetime in "
                "Series.combine_first is deprecated and will be removed "
                "in a future version. Explicitly cast before calling "
                "combine_first instead.",
                Pandas4Warning,
                stacklevel=find_stack_level(),
            )

        combined = concat([this, other])
        combined = combined.reindex(new_index)
        return combined.__finalize__(self, method="combine_first")
```



---

### Функция ID: 47

**Исходный код:**
```python
def add_prefix(self, prefix: str, axis: Axis | None = None) -> Self:
        """
        Prefix labels with string `prefix`.

        For Series, the row labels are prefixed.
        For DataFrame, the column labels are prefixed.

        Parameters
        ----------
        prefix : str
            The string to add before each label.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            Axis to add prefix on

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or DataFrame
            New Series or DataFrame with updated labels.

        See Also
        --------
        Series.add_suffix: Suffix row labels with string `suffix`.
        DataFrame.add_suffix: Suffix column labels with string `suffix`.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.add_prefix("item_")
        item_0    1
        item_1    2
        item_2    3
        item_3    4
        dtype: int64

        >>> df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6

        >>> df.add_prefix("col_")
             col_A  col_B
        0       1       3
        1       2       4
        2       3       5
        3       4       6
        """
        f = lambda x: f"{prefix}{x}"

        axis_name = self._info_axis_name
        if axis is not None:
            axis_name = self._get_axis_name(axis)

        mapper = {axis_name: f}

        # error: Keywords must be strings
        # error: No overload variant of "_rename" of "NDFrame" matches
        # argument type "dict[Literal['index', 'columns'], Callable[[Any], str]]"
        return self._rename(**mapper)  # type: ignore[call-overload, misc]
```



---

### Функция ID: 48

**Исходный код:**
```python
def combine(
        self,
        other: Series | Hashable,
        func: Callable[[Hashable, Hashable], Hashable],
        fill_value: Hashable | None = None,
    ) -> Series:
        """
        Combine the Series with a Series or scalar according to `func`.

        Combine the Series and `other` using `func` to perform elementwise
        selection for combined Series.
        `fill_value` is assumed when value is not present at some index
        from one of the two Series being combined.

        Parameters
        ----------
        other : Series or scalar
            The value(s) to be combined with the `Series`.
        func : function
            Function that takes two scalars as inputs and returns an element.
        fill_value : scalar, optional
            The value to assume when an index is missing from
            one Series or the other. The default specifies to use the
            appropriate NaN value for the underlying dtype of the Series.

        Returns
        -------
        Series
            The result of combining the Series with the other object.

        See Also
        --------
        Series.combine_first : Combine Series values, choosing the calling
            Series' values first.

        Examples
        --------
        Consider 2 Datasets ``s1`` and ``s2`` containing
        highest clocked speeds of different birds.

        >>> s1 = pd.Series({"falcon": 330.0, "eagle": 160.0})
        >>> s1
        falcon    330.0
        eagle     160.0
        dtype: float64
        >>> s2 = pd.Series({"falcon": 345.0, "eagle": 200.0, "duck": 30.0})
        >>> s2
        falcon    345.0
        eagle     200.0
        duck       30.0
        dtype: float64

        Now, to combine the two datasets and view the highest speeds
        of the birds across the two datasets

        >>> s1.combine(s2, max)
        duck        NaN
        eagle     200.0
        falcon    345.0
        dtype: float64

        In the previous example, the resulting value for duck is missing,
        because the maximum of a NaN and a float is a NaN.
        So, in the example, we set ``fill_value=0``,
        so the maximum value returned will be the value from some dataset.

        >>> s1.combine(s2, max, fill_value=0)
        duck       30.0
        eagle     200.0
        falcon    345.0
        dtype: float64
        """
        if fill_value is None:
            fill_value = na_value_for_dtype(self.dtype, compat=False)

        if isinstance(other, Series):
            # If other is a Series, result is based on union of Series,
            # so do this element by element
            new_index = self.index.union(other.index)
            new_name = ops.get_op_result_name(self, other)
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all="ignore"):
                for i, idx in enumerate(new_index):
                    lv = self.get(idx, fill_value)
                    rv = other.get(idx, fill_value)
                    new_values[i] = func(lv, rv)
        else:
            # Assume that other is a scalar, so apply the function for
            # each element in the Series
            new_index = self.index
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all="ignore"):
                new_values[:] = [func(lv, other) for lv in self._values]
            new_name = self.name

        res_values = self.array._cast_pointwise_result(new_values)
        return self._constructor(
            res_values,
            dtype=res_values.dtype,
            index=new_index,
            name=new_name,
            copy=False,
        )
```



---

### Функция ID: 49

**Исходный код:**
```python
def filter(
        self,
        items=None,
        like: str | None = None,
        regex: str | None = None,
        axis: Axis | None = None,
    ) -> Self:
        """
        Subset the DataFrame or Series according to the specified index labels.

        For DataFrame, filter rows or columns depending on ``axis`` argument.
        Note that this routine does not filter based on content.
        The filter is applied to the labels of the index.

        Parameters
        ----------
        items : list-like
            Keep labels from axis which are in items.
        like : str
            Keep labels from axis for which "like in label == True".
        regex : str (regular expression)
            Keep labels from axis for which re.search(regex, label) == True.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            The axis to filter on, expressed either as an index (int)
            or axis name (str). By default this is the info axis, 'columns' for
            ``DataFrame``. For ``Series`` this parameter is unused and defaults to
            ``None``.

        Returns
        -------
        Same type as caller
            The filtered subset of the DataFrame or Series.

        See Also
        --------
        DataFrame.loc : Access a group of rows and columns
            by label(s) or a boolean array.

        Notes
        -----
        The ``items``, ``like``, and ``regex`` parameters are
        enforced to be mutually exclusive.

        ``axis`` defaults to the info axis that is used when indexing
        with ``[]``.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     np.array(([1, 2, 3], [4, 5, 6])),
        ...     index=["mouse", "rabbit"],
        ...     columns=["one", "two", "three"],
        ... )
        >>> df
                one  two  three
        mouse     1    2      3
        rabbit    4    5      6

        >>> # select columns by name
        >>> df.filter(items=["one", "three"])
                 one  three
        mouse     1      3
        rabbit    4      6

        >>> # select columns by regular expression
        >>> df.filter(regex="e$", axis=1)
                 one  three
        mouse     1      3
        rabbit    4      6

        >>> # select rows containing 'bbi'
        >>> df.filter(like="bbi", axis=0)
                 one  two  three
        rabbit    4    5      6
        """
        nkw = common.count_not_none(items, like, regex)
        if nkw > 1:
            raise TypeError(
                "Keyword arguments `items`, `like`, or `regex` are mutually exclusive"
            )

        if axis is None:
            axis = self._info_axis_name
        labels = self._get_axis(axis)

        if items is not None:
            name = self._get_axis_name(axis)
            items = Index(items).intersection(labels)
            if len(items) == 0:
                # Keep the dtype of labels when we are empty
                items = items.astype(labels.dtype)
            # error: Keywords must be strings
            return self.reindex(**{name: items})  # type: ignore[misc]
        elif like:

            def f(x) -> bool:
                assert like is not None  # needed for mypy
                return like in ensure_str(x)

            values = labels.map(f)
            return self.loc(axis=axis)[values]
        elif regex:

            def f(x) -> bool:
                return matcher.search(ensure_str(x)) is not None

            matcher = re.compile(regex)
            values = labels.map(f)
            return self.loc(axis=axis)[values]
        else:
            raise TypeError("Must pass either `items`, `like`, or `regex`")
```



---

### Функция ID: 50

**Исходный код:**
```python
def add_suffix(self, suffix: str, axis: Axis | None = None) -> Self:
        """
        Suffix labels with string `suffix`.

        For Series, the row labels are suffixed.
        For DataFrame, the column labels are suffixed.

        Parameters
        ----------
        suffix : str
            The string to add after each label.
        axis : {0 or 'index', 1 or 'columns', None}, default None
            Axis to add suffix on

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or DataFrame
            New Series or DataFrame with updated labels.

        See Also
        --------
        Series.add_prefix: Prefix row labels with string `prefix`.
        DataFrame.add_prefix: Prefix column labels with string `prefix`.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.add_suffix("_item")
        0_item    1
        1_item    2
        2_item    3
        3_item    4
        dtype: int64

        >>> df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6

        >>> df.add_suffix("_col")
             A_col  B_col
        0       1       3
        1       2       4
        2       3       5
        3       4       6
        """
        f = lambda x: f"{x}{suffix}"

        axis_name = self._info_axis_name
        if axis is not None:
            axis_name = self._get_axis_name(axis)

        mapper = {axis_name: f}
        # error: Keywords must be strings
        # error: No overload variant of "_rename" of "NDFrame" matches argument
        # type "dict[Literal['index', 'columns'], Callable[[Any], str]]"
        return self._rename(**mapper)  # type: ignore[call-overload, misc]
```



---

### Функция ID: 51

**Исходный код:**
```python
def pct_change(
        self,
        periods: int = 1,
        fill_method: None = None,
        freq=None,
    ):
        """
        Calculate pct_change of each value to previous entry in group.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating percentage change. Comparing with
            a period of 1 means adjacent elements are compared, whereas a period
            of 2 compares every other element.

        fill_method : None
            Must be None. This argument will be removed in a future version of pandas.

        freq : str, pandas offset object, or None, default None
            The frequency increment for time series data (e.g., 'M' for month-end).
            If None, the frequency is inferred from the index. Relevant for time
            series data only.

        Returns
        -------
        Series or DataFrame
            Percentage changes within each group.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b", "b"]
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).pct_change()
        a         NaN
        a    1.000000
        b         NaN
        b    0.333333
        dtype: float64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(
        ...     data,
        ...     columns=["a", "b", "c"],
        ...     index=["tuna", "salmon", "catfish", "goldfish"],
        ... )
        >>> df
                   a  b  c
            tuna   1  2  3
          salmon   1  5  6
         catfish   2  5  8
        goldfish   2  6  9
        >>> df.groupby("a").pct_change()
                    b  c
            tuna    NaN    NaN
          salmon    1.5  1.000
         catfish    NaN    NaN
        goldfish    0.2  0.125
        """
        # GH#53491
        if fill_method is not None:
            raise ValueError(f"fill_method must be None; got {fill_method=}.")

        # TODO(GH#23918): Remove this conditional for SeriesGroupBy when
        #  GH#23918 is fixed
        if freq is not None:
            f = lambda x: x.pct_change(
                periods=periods,
                freq=freq,
                axis=0,
            )
            return self._python_apply_general(f, self._selected_obj, is_transform=True)

        if fill_method is None:  # GH30463
            op = "ffill"
        else:
            op = fill_method
        filled = getattr(self, op)(limit=0)
        fill_grp = filled.groupby(self._grouper.codes, group_keys=self.group_keys)
        shifted = fill_grp.shift(periods=periods, freq=freq)
        return (filled / shifted) - 1
```



---

### Функция ID: 52

**Исходный код:**
```python
def slice_locs(
        self,
        start: SliceType = None,
        end: SliceType = None,
        step: int | None = None,
    ) -> tuple[int, int]:
        """
        Compute slice locations for input labels.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning.
        end : label, default None
            If None, defaults to the end.
        step : int, defaults None
            If None, defaults to 1.

        Returns
        -------
        tuple[int, int]
            Returns a tuple of two integers representing the slice locations for the
            input labels within the index.

        See Also
        --------
        Index.get_loc : Get location for a single label.

        Notes
        -----
        This method only works if the index is monotonic or unique.

        Examples
        --------
        >>> idx = pd.Index(list("abcd"))
        >>> idx.slice_locs(start="b", end="c")
        (1, 3)

        >>> idx = pd.Index(list("bcde"))
        >>> idx.slice_locs(start="a", end="c")
        (0, 2)
        """
        inc = step is None or step >= 0

        if not inc:
            # If it's a reverse slice, temporarily swap bounds.
            start, end = end, start

        # GH 16785: If start and end happen to be date strings with UTC offsets
        # attempt to parse and check that the offsets are the same
        if isinstance(start, (str, datetime)) and isinstance(end, (str, datetime)):
            try:
                ts_start = Timestamp(start)
                ts_end = Timestamp(end)
            except (ValueError, TypeError):
                pass
            else:
                if not tz_compare(ts_start.tzinfo, ts_end.tzinfo):
                    raise ValueError("Both dates must have the same UTC offset")

        start_slice = None
        if start is not None:
            start_slice = self.get_slice_bound(start, "left")
        if start_slice is None:
            start_slice = 0

        end_slice = None
        if end is not None:
            end_slice = self.get_slice_bound(end, "right")
        if end_slice is None:
            end_slice = len(self)

        if not inc:
            # Bounds at this moment are swapped, swap them back and shift by 1.
            #
            # slice_locs('B', 'A', step=-1): s='B', e='A'
            #
            #              s='A'                 e='B'
            # AFTER SWAP:    |                     |
            #                v ------------------> V
            #           -----------------------------------
            #           | | |A|A|A|A| | | | | |B|B| | | | |
            #           -----------------------------------
            #              ^ <------------------ ^
            # SHOULD BE:   |                     |
            #           end=s-1              start=e-1
            #
            end_slice, start_slice = start_slice - 1, end_slice - 1

            # i == -1 triggers ``len(self) + i`` selection that points to the
            # last element, not before-the-first one, subtracting len(self)
            # compensates that.
            if end_slice == -1:
                end_slice -= len(self)
            if start_slice == -1:
                start_slice -= len(self)

        return start_slice, end_slice
```



---

### Функция ID: 53

**Исходный код:**
```python
def dropna(
        self,
        *,
        axis: Axis = 0,
        inplace: bool = False,
        how: AnyAll | None = None,
        ignore_index: bool = False,
    ) -> Series | None:
        """
        Return a new Series with missing values removed.

        See the :ref:`User Guide <missing_data>` for more on which values are
        considered missing, and how to work with missing data.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        inplace : bool, default False
            If True, do operation inplace and return None.
        how : str, optional
            Not in use. Kept for compatibility.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or None
            Series with NA entries dropped from it or None if ``inplace=True``.

        See Also
        --------
        Series.isna: Indicate missing values.
        Series.notna : Indicate existing (non-missing) values.
        Series.fillna : Replace missing values.
        DataFrame.dropna : Drop rows or columns which contain NA values.
        Index.dropna : Drop missing indices.

        Examples
        --------
        >>> ser = pd.Series([1.0, 2.0, np.nan])
        >>> ser
        0    1.0
        1    2.0
        2    NaN
        dtype: float64

        Drop NA values from a Series.

        >>> ser.dropna()
        0    1.0
        1    2.0
        dtype: float64

        Empty strings are not considered NA values. ``None`` is considered an
        NA value.

        >>> ser = pd.Series([np.nan, 2, pd.NaT, "", None, "I stay"])
        >>> ser
        0       NaN
        1         2
        2       NaT
        3
        4      None
        5    I stay
        dtype: object
        >>> ser.dropna()
        1         2
        3
        5    I stay
        dtype: object
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        ignore_index = validate_bool_kwarg(ignore_index, "ignore_index")
        # Validate the axis parameter
        self._get_axis_number(axis or 0)

        if self._can_hold_na:
            result = remove_na_arraylike(self)
        else:
            if not inplace:
                result = self.copy(deep=False)
            else:
                result = self

        if ignore_index:
            result.index = default_index(len(result))

        if inplace:
            return self._update_inplace(result)
        else:
            return result
```



---

### Функция ID: 54

**Исходный код:**
```python
def putmask(self, mask, value) -> Index:
        """
        Return a new Index of the values set with the mask.

        Parameters
        ----------
        mask : array-like of bool
            Array of booleans denoting where values should be replaced.
        value : scalar
            Scalar value to use to fill holes (e.g. 0).
            This value cannot be a list-likes.

        Returns
        -------
        Index
            A new Index of the values set with the mask.

        See Also
        --------
        numpy.putmask : Changes elements of an array
            based on conditional and input values.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3])
        >>> idx2 = pd.Index([5, 6, 7])
        >>> idx1.putmask([True, False, False], idx2)
        Index([5, 2, 3], dtype='int64')
        """
        mask, noop = validate_putmask(self._values, mask)
        if noop:
            return self.copy()

        if self.dtype != object and is_valid_na_for_dtype(value, self.dtype):
            # e.g. None -> np.nan, see also Block._standardize_fill_value
            value = self._na_value

        try:
            converted = self._validate_fill_value(value)
        except (LossySetitemError, ValueError, TypeError) as err:
            if is_object_dtype(self.dtype):  # pragma: no cover
                raise err

            # See also: Block.coerce_to_target_dtype
            dtype = self._find_common_type_compat(value)
            if dtype == self.dtype:
                # GH#56376 avoid RecursionError
                raise AssertionError(
                    "Something has gone wrong. Please report a bug at "
                    "github.com/pandas-dev/pandas"
                ) from err
            return self.astype(dtype).putmask(mask, value)

        values = self._values.copy()

        if isinstance(values, np.ndarray):
            converted = setitem_datetimelike_compat(values, mask.sum(), converted)
            np.putmask(values, mask, converted)

        else:
            # Note: we use the original value here, not converted, as
            #  _validate_fill_value is not idempotent
            values._putmask(mask, value)

        return self._shallow_copy(values)
```



---

### Функция ID: 55

**Исходный код:**
```python
def from_spmatrix(cls, data, index=None, columns=None) -> DataFrame:
        """
        Create a new DataFrame from a scipy sparse matrix.

        Parameters
        ----------
        data : scipy.sparse.spmatrix
            Must be convertible to csc format.
        index, columns : Index, optional
            Row and column labels to use for the resulting DataFrame.
            Defaults to a RangeIndex.

        Returns
        -------
        DataFrame
            Each column of the DataFrame is stored as a
            :class:`arrays.SparseArray`.

        See Also
        --------
        DataFrame.sparse.to_coo : Return the contents of the frame as a
            sparse SciPy COO matrix.

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.eye(3, dtype=int)
        >>> pd.DataFrame.sparse.from_spmatrix(mat)
             0    1    2
        0    1    0    0
        1    0    1    0
        2    0    0    1
        """
        from pandas._libs.sparse import IntIndex

        from pandas import DataFrame

        data = data.tocsc()
        index, columns = cls._prep_index(data, index, columns)
        n_rows, n_columns = data.shape
        # We need to make sure indices are sorted, as we create
        # IntIndex with no input validation (i.e. check_integrity=False ).
        # Indices may already be sorted in scipy in which case this adds
        # a small overhead.
        data.sort_indices()
        indices = data.indices
        indptr = data.indptr
        array_data = data.data
        dtype = SparseDtype(array_data.dtype)
        arrays = []
        for i in range(n_columns):
            sl = slice(indptr[i], indptr[i + 1])
            idx = IntIndex(n_rows, indices[sl], check_integrity=False)
            arr = SparseArray._simple_new(array_data[sl], idx, dtype)
            arrays.append(arr)
        return DataFrame._from_arrays(
            arrays, columns=columns, index=index, verify_integrity=False
        )
```



---

### Функция ID: 56

**Исходный код:**
```python
def map(self, mapper, na_action: Literal["ignore"] | None = None):
        """
        Map values using an input mapping or function.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        Union[Index, MultiIndex]
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.

        See Also
        --------
        Index.where : Replace values where the condition is False.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.map({1: "a", 2: "b", 3: "c"})
        Index(['a', 'b', 'c'], dtype='object')

        Using `map` with a function:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx.map("I am a {}".format)
        Index(['I am a 1', 'I am a 2', 'I am a 3'], dtype='object')

        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.map(lambda x: x.upper())
        Index(['A', 'B', 'C'], dtype='object')
        """
        from pandas.core.indexes.multi import MultiIndex

        new_values = self._map_values(mapper, na_action=na_action)

        # we can return a MultiIndex
        if new_values.size and isinstance(new_values[0], tuple):
            if isinstance(self, MultiIndex):
                names = self.names
            elif self.name:
                names = [self.name] * len(new_values[0])
            else:
                names = None
            return MultiIndex.from_tuples(new_values, names=names)

        dtype = None
        if not new_values.size:
            # empty
            dtype = self.dtype
        elif isinstance(new_values, Categorical):
            # cast_pointwise_result is unnecessary
            dtype = new_values.dtype
        else:
            if isinstance(self, MultiIndex):
                arr = self[:0].to_flat_index().array
            else:
                arr = self[:0].array
            # e.g. if we are floating and new_values is all ints, then we
            #  don't want to cast back to floating.  But if we are UInt64
            #  and new_values is all ints, we want to try.
            new_values = arr._cast_pointwise_result(new_values)
            dtype = new_values.dtype
        return Index(new_values, dtype=dtype, copy=False, name=self.name)
```



---

### Функция ID: 57

**Исходный код:**
```python
def count(self, axis: Axis = 0, numeric_only: bool = False) -> Series:
        """
        Count non-NA cells for each column or row.

        The values `None`, `NaN`, `NaT`, ``pandas.NA`` are considered NA.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            If 0 or 'index' counts are generated for each column.
            If 1 or 'columns' counts are generated for each row.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

        Returns
        -------
        Series
            For each column/row the number of non-NA/null entries.

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.value_counts: Count unique combinations of columns.
        DataFrame.shape: Number of DataFrame rows and columns (including NA
            elements).
        DataFrame.isna: Boolean same-sized DataFrame showing places of NA
            elements.

        Examples
        --------
        Constructing DataFrame from a dictionary:

        >>> df = pd.DataFrame(
        ...     {
        ...         "Person": ["John", "Myla", "Lewis", "John", "Myla"],
        ...         "Age": [24.0, np.nan, 21.0, 33, 26],
        ...         "Single": [False, True, True, True, False],
        ...     }
        ... )
        >>> df
           Person   Age  Single
        0    John  24.0   False
        1    Myla   NaN    True
        2   Lewis  21.0    True
        3    John  33.0    True
        4    Myla  26.0   False

        Notice the uncounted NA values:

        >>> df.count()
        Person    5
        Age       4
        Single    5
        dtype: int64

        Counts for each **row**:

        >>> df.count(axis="columns")
        0    3
        1    2
        2    3
        3    3
        4    3
        dtype: int64
        """
        axis = self._get_axis_number(axis)

        if numeric_only:
            frame = self._get_numeric_data()
        else:
            frame = self

        # GH #423
        if len(frame._get_axis(axis)) == 0:
            result = self._constructor_sliced(0, index=frame._get_agg_axis(axis))
        else:
            result = notna(frame).sum(axis=axis)

        return result.astype("int64").__finalize__(self, method="count")
```



---

### Функция ID: 58

**Исходный код:**
```javascript
function concat() {
      var length = arguments.length;
      if (!length) {
        return [];
      }
      var args = Array(length - 1),
          array = arguments[0],
          index = length;

      while (index--) {
        args[index - 1] = arguments[index];
      }
      return arrayPush(isArray(array) ? copyArray(array) : [array], baseFlatten(args, 1));
    }
```



---

### Функция ID: 59

**Исходный код:**
```python
def isocalendar(self) -> DataFrame:
        """
        Calculate year, week, and day according to the ISO 8601 standard.

        Returns
        -------
        DataFrame
            With columns year, week and day.

        See Also
        --------
        Timestamp.isocalendar : Function return a 3-tuple containing ISO year,
            week number, and weekday for the given Timestamp object.
        datetime.date.isocalendar : Return a named tuple object with
            three components: year, week and weekday.

        Examples
        --------
        >>> idx = pd.date_range(start="2019-12-29", freq="D", periods=4)
        >>> idx.isocalendar()
                    year  week  day
        2019-12-29  2019    52    7
        2019-12-30  2020     1    1
        2019-12-31  2020     1    2
        2020-01-01  2020     1    3
        >>> idx.isocalendar().week
        2019-12-29    52
        2019-12-30     1
        2019-12-31     1
        2020-01-01     1
        Freq: D, Name: week, dtype: UInt32
        """
        from pandas import DataFrame

        values = self._local_timestamps()
        sarray = fields.build_isocalendar_sarray(values, reso=self._creso)
        iso_calendar_df = DataFrame(
            sarray, columns=["year", "week", "day"], dtype="UInt32"
        )
        if self._hasna:
            iso_calendar_df.iloc[self._isnan] = None
        return iso_calendar_df
```



---

### Функция ID: 60

**Исходный код:**
```typescript
function getBlockIndent(sourceFile: SourceFile, position: number, options: EditorSettings): number {

        // move backwards until we find a line with a non-whitespace character,

        // then find the first non-whitespace character for that line.

        let current = position;

        while (current > 0) {

            const char = sourceFile.text.charCodeAt(current);

            if (!isWhiteSpaceLike(char)) {

                break;

            }

            current--;

        }



        const lineStart = getLineStartPositionForPosition(current, sourceFile);

        return findFirstNonWhitespaceColumn(lineStart, current, sourceFile, options);

    }
```



---

### Функция ID: 61

**Исходный код:**
```python
def round(self, decimals: int = 0) -> Self | Index:  # type: ignore[override]
        """
        Round each value in the Index to the given number of decimals.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point
            e.g. ``round(11.0, -1) == 10.0``.

        Returns
        -------
        Index or RangeIndex
            A new Index with the rounded values.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.RangeIndex(10, 30, 10)
        >>> idx.round(decimals=-1)
        RangeIndex(start=10, stop=30, step=10)
        >>> idx = pd.RangeIndex(10, 15, 1)
        >>> idx.round(decimals=-1)
        Index([10, 10, 10, 10, 10], dtype='int64')
        """
        if decimals >= 0:
            return self.copy()
        elif self.start % 10**-decimals == 0 and self.step % 10**-decimals == 0:
            # e.g. RangeIndex(10, 30, 10).round(-1) doesn't need rounding
            return self.copy()
        else:
            return super().round(decimals=decimals)
```



---

### Функция ID: 62

**Исходный код:**
```python
def format_percentiles(
    percentiles: np.ndarray | Sequence[float],
) -> list[str]:
    """
    Outputs rounded and formatted percentiles.

    Parameters
    ----------
    percentiles : list-like, containing floats from interval [0,1]

    Returns
    -------
    formatted : list of strings

    Notes
    -----
    Rounding precision is chosen so that: (1) if any two elements of
    ``percentiles`` differ, they remain different after rounding
    (2) no entry is *rounded* to 0% or 100%.
    Any non-integer is always rounded to at least 1 decimal place.

    Examples
    --------
    Keeps all entries different after rounding:

    >>> format_percentiles([0.01999, 0.02001, 0.5, 0.666666, 0.9999])
    ['1.999%', '2.001%', '50%', '66.667%', '99.99%']

    No element is rounded to 0% or 100% (unless already equal to it).
    Duplicates are allowed:

    >>> format_percentiles([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999])
    ['0%', '50%', '2.0%', '50%', '66.67%', '99.99%']
    """
    if len(percentiles) == 0:
        return []

    percentiles = np.asarray(percentiles)

    # It checks for np.nan as well
    if (
        not is_numeric_dtype(percentiles)
        or not np.all(percentiles >= 0)
        or not np.all(percentiles <= 1)
    ):
        raise ValueError("percentiles should all be in the interval [0,1]")

    percentiles = 100 * percentiles
    prec = get_precision(percentiles)
    percentiles_round_type = percentiles.round(prec).astype(int)

    int_idx = np.isclose(percentiles_round_type, percentiles)

    if np.all(int_idx):
        out = percentiles_round_type.astype(str)
        return [i + "%" for i in out]

    unique_pcts = np.unique(percentiles)
    prec = get_precision(unique_pcts)
    out = np.empty_like(percentiles, dtype=object)
    out[int_idx] = percentiles[int_idx].round().astype(int).astype(str)

    out[~int_idx] = percentiles[~int_idx].round(prec).astype(str)
    return [i + "%" for i in out]
```



---

### Функция ID: 63

**Исходный код:**
```typescript
function getActualIndentationForNode(current: Node, parent: Node, currentLineAndChar: LineAndCharacter, parentAndChildShareLine: boolean, sourceFile: SourceFile, options: EditorSettings): number {

        // actual indentation is used for statements\declarations if one of cases below is true:

        // - parent is SourceFile - by default immediate children of SourceFile are not indented except when user indents them manually

        // - parent and child are not on the same line

        const useActualIndentation = (isDeclaration(current) || isStatementButNotDeclaration(current)) &&

            (parent.kind === SyntaxKind.SourceFile || !parentAndChildShareLine);



        if (!useActualIndentation) {

            return Value.Unknown;

        }



        return findColumnForFirstNonWhitespaceCharacterInLine(currentLineAndChar, sourceFile, options);

    }
```



---

### Функция ID: 64

**Исходный код:**
```python
def nansem(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute the standard error in the mean along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nansem(s.values)
     np.float64(0.5773502691896258)
    """
    # This checks if non-numeric-like data is passed with numeric_only=False
    # and raises a TypeError otherwise
    nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")

    if not skipna and mask is not None and mask.any():
        return np.nan

    count, _ = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    var = nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    return np.sqrt(var) / np.sqrt(count)
```



---

### Функция ID: 65

**Исходный код:**
```python
def construct_from_string(cls, string: str) -> Self:
        r"""
        Construct this type from a string.

        This is useful mainly for data types that accept parameters.
        For example, a period dtype accepts a frequency parameter that
        can be set as ``period[h]`` (where H means hourly frequency).

        By default, in the abstract class, just the name of the type is
        expected. But subclasses can overwrite this method to accept
        parameters.

        Parameters
        ----------
        string : str
            The name of the type, for example ``category``.

        Returns
        -------
        ExtensionDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a class cannot be constructed from this 'string'.

        Examples
        --------
        For extension dtypes with arguments the following may be an
        adequate implementation.

        >>> import re
        >>> @classmethod
        ... def construct_from_string(cls, string):
        ...     pattern = re.compile(r"^my_type\[(?P<arg_name>.+)\]$")
        ...     match = pattern.match(string)
        ...     if match:
        ...         return cls(**match.groupdict())
        ...     else:
        ...         raise TypeError(
        ...             f"Cannot construct a '{cls.__name__}' from '{string}'"
        ...         )
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        # error: Non-overlapping equality check (left operand type: "str", right
        #  operand type: "Callable[[ExtensionDtype], str]")  [comparison-overlap]
        assert isinstance(cls.name, str), (cls, type(cls.name))
        if string != cls.name:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        return cls()
```



---

### Функция ID: 66

**Исходный код:**
```python
def round(self, decimals: int = 0, *args, **kwargs) -> Series:
        """
        Round each value in a Series to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Series
            Rounded values of the Series.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.
        Series.dt.round : Round values of data to the specified freq.

        Notes
        -----
        For values exactly halfway between rounded decimal values, pandas rounds
        to the nearest even value (e.g. -0.5 and 0.5 round to 0.0, 1.5 and 2.5
        round to 2.0, etc.).

        Examples
        --------
        >>> s = pd.Series([-0.5, 0.1, 2.5, 1.3, 2.7])
        >>> s.round()
        0   -0.0
        1    0.0
        2    2.0
        3    1.0
        4    3.0
        dtype: float64
        """
        nv.validate_round(args, kwargs)
        if self.dtype == "object":
            raise TypeError("Expected numeric dtype, got object instead.")
        new_mgr = self._mgr.round(decimals=decimals)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(
            self, method="round"
        )
```



---

### Функция ID: 67

**Исходный код:**
```java
public static void swap(final boolean[] array, int offset1, int offset2, int len) {
        if (isEmpty(array) || offset1 >= array.length || offset2 >= array.length) {
            return;
        }
        offset1 = max0(offset1);
        offset2 = max0(offset2);
        len = Math.min(Math.min(len, array.length - offset1), array.length - offset2);
        for (int i = 0; i < len; i++, offset1++, offset2++) {
            final boolean aux = array[offset1];
            array[offset1] = array[offset2];
            array[offset2] = aux;
        }
    }
```



---

### Функция ID: 68

**Исходный код:**
```typescript
function nextTokenIsCurlyBraceOnSameLineAsCursor(precedingToken: Node, current: Node, lineAtPosition: number, sourceFile: SourceFile): NextTokenKind {

        const nextToken = findNextToken(precedingToken, current, sourceFile);

        if (!nextToken) {

            return NextTokenKind.Unknown;

        }



        if (nextToken.kind === SyntaxKind.OpenBraceToken) {

            // open braces are always indented at the parent level

            return NextTokenKind.OpenBrace;

        }

        else if (nextToken.kind === SyntaxKind.CloseBraceToken) {

            // close braces are indented at the parent level if they are located on the same line with cursor

            // this means that if new line will be added at $ position, this case will be indented

            // class A {

            //    $

            // }

            /// and this one - not

            // class A {

            // $}



            const nextTokenStartLine = getStartLineAndCharacterForNode(nextToken, sourceFile).line;

            return lineAtPosition === nextTokenStartLine ? NextTokenKind.CloseBrace : NextTokenKind.Unknown;

        }



        return NextTokenKind.Unknown;

    }
```



---

### Функция ID: 69

**Исходный код:**
```python
def is_dict_like(obj: object) -> bool:
    """
    Check if the object is dict-like.

    Parameters
    ----------
    obj : object
        The object to check. This can be any Python object,
        and the function will determine whether it
        behaves like a dictionary.

    Returns
    -------
    bool
        Whether `obj` has dict-like properties.

    See Also
    --------
    api.types.is_list_like : Check if the object is list-like.
    api.types.is_file_like : Check if the object is a file-like.
    api.types.is_named_tuple : Check if the object is a named tuple.

    Examples
    --------
    >>> from pandas.api.types import is_dict_like
    >>> is_dict_like({1: 2})
    True
    >>> is_dict_like([1, 2, 3])
    False
    >>> is_dict_like(dict)
    False
    >>> is_dict_like(dict())
    True
    """
    dict_like_attrs = ("__getitem__", "keys", "__contains__")
    return (
        all(hasattr(obj, attr) for attr in dict_like_attrs)
        # [GH 25196] exclude classes
        and not isinstance(obj, type)
    )
```



---

### Функция ID: 70

**Исходный код:**
```python
def collapse_resume_frames(stack: StackSummary) -> StackSummary:
    """
    When we graph break, we create a resume function and make a regular Python call
    to it, which gets intercepted by Dynamo. This behavior is normally shown in the
    traceback, which can be confusing to a user. So we can filter out resume frames
    for better traceback clarity.

    Example:
    File "..." line 3, in f
        <line 3>
    File "..." line 5, in torch_dynamo_resume_in_f_at_80
        <line 5>
    File "..." line 10, in torch_dynamo_resume_in_f_at_120
        <line 10>

    becomes
    File "..." line 10, in f
        <line 10>
    """

    new_stack = StackSummary()
    for frame in stack:
        if frame.filename is None:
            continue
        name = remove_resume_prefix(frame.name)
        if new_stack and name and new_stack[-1].name == name:
            new_stack[-1] = frame
            frame.name = name
        else:
            new_stack.append(frame)

    return new_stack
```



---

### Функция ID: 71

**Исходный код:**
```python
def reorder_levels(self, order: Sequence[Level]) -> Series:
        """
        Rearrange index levels using input order.

        May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int representing new level order
            Reference level by number or key.

        Returns
        -------
        Series
            Type of caller with index as MultiIndex (new object).

        See Also
        --------
        DataFrame.reorder_levels : Rearrange index or column levels using
            input ``order``.

        Examples
        --------
        >>> arrays = [
        ...     np.array(["dog", "dog", "cat", "cat", "bird", "bird"]),
        ...     np.array(["white", "black", "white", "black", "white", "black"]),
        ... ]
        >>> s = pd.Series([1, 2, 3, 3, 5, 2], index=arrays)
        >>> s
        dog   white    1
              black    2
        cat   white    3
              black    3
        bird  white    5
              black    2
        dtype: int64
        >>> s.reorder_levels([1, 0])
        white  dog     1
        black  dog     2
        white  cat     3
        black  cat     3
        white  bird    5
        black  bird    2
        dtype: int64
        """
        if not isinstance(self.index, MultiIndex):  # pragma: no cover
            raise Exception("Can only reorder levels on a hierarchical axis.")

        result = self.copy(deep=False)
        assert isinstance(result.index, MultiIndex)
        result.index = result.index.reorder_levels(order)
        return result
```



---

### Функция ID: 72

**Исходный код:**
```python
def extract_array(
    obj: T, extract_numpy: bool = False, extract_range: bool = False
) -> T | ArrayLike:
    """
    Extract the ndarray or ExtensionArray from a Series or Index.

    For all other types, `obj` is just returned as is.

    Parameters
    ----------
    obj : object
        For Series / Index, the underlying ExtensionArray is unboxed.

    extract_numpy : bool, default False
        Whether to extract the ndarray from a NumpyExtensionArray.

    extract_range : bool, default False
        If we have a RangeIndex, return range._values if True
        (which is a materialized integer ndarray), otherwise return unchanged.

    Returns
    -------
    arr : object

    Examples
    --------
    >>> extract_array(pd.Series(["a", "b", "c"], dtype="category"))
    ['a', 'b', 'c']
    Categories (3, str): ['a', 'b', 'c']

    Other objects like lists, arrays, and DataFrames are just passed through.

    >>> extract_array([1, 2, 3])
    [1, 2, 3]

    For an ndarray-backed Series / Index the ndarray is returned.

    >>> extract_array(pd.Series([1, 2, 3]))
    array([1, 2, 3])

    To extract all the way down to the ndarray, pass ``extract_numpy=True``.

    >>> extract_array(pd.Series([1, 2, 3]), extract_numpy=True)
    array([1, 2, 3])
    """
    typ = getattr(obj, "_typ", None)
    if typ in _typs:
        # i.e. isinstance(obj, (ABCIndex, ABCSeries))
        if typ == "rangeindex":
            if extract_range:
                # error: "T" has no attribute "_values"
                return obj._values  # type: ignore[attr-defined]
            return obj

        # error: "T" has no attribute "_values"
        return obj._values  # type: ignore[attr-defined]

    elif extract_numpy and typ == "npy_extension":
        # i.e. isinstance(obj, ABCNumpyExtensionArray)
        # error: "T" has no attribute "to_numpy"
        return obj.to_numpy()  # type: ignore[attr-defined]

    return obj
```



---

### Функция ID: 73

**Исходный код:**
```python
def decode(
        self, encoding, errors: str = "strict", dtype: str | DtypeObj | None = None
    ):
        """
        Decode character string in the Series/Index using indicated encoding.

        Equivalent to :meth:`str.decode` in python2 and :meth:`bytes.decode` in
        python3.

        Parameters
        ----------
        encoding : str
            Specifies the encoding to be used.
        errors : str, optional
            Specifies the error handling scheme.
            Possible values are those supported by :meth:`bytes.decode`.
        dtype : str or dtype, optional
            The dtype of the result. When not ``None``, must be either a string or
            object dtype. When ``None``, the dtype of the result is determined by
            ``pd.options.future.infer_string``.

            .. versionadded:: 2.3.0

        Returns
        -------
        Series or Index
            A Series or Index with decoded strings.

        See Also
        --------
        Series.str.encode : Encodes strings into bytes in a Series/Index.

        Examples
        --------
        For Series:

        >>> ser = pd.Series([b"cow", b"123", b"()"])
        >>> ser.str.decode("ascii")
        0   cow
        1   123
        2   ()
        dtype: str
        """
        if dtype is not None and not is_string_dtype(dtype):
            raise ValueError(f"dtype must be string or object, got {dtype=}")
        if dtype is None and using_string_dtype():
            dtype = "str"
        # TODO: Add a similar _bytes interface.
        if encoding in _cpython_optimized_decoders:
            # CPython optimized implementation
            f = lambda x: x.decode(encoding, errors)
        else:
            decoder = codecs.getdecoder(encoding)
            f = lambda x: decoder(x, errors)[0]
        arr = self._data.array
        result = arr._str_map(f)
        return self._wrap_result(result, dtype=dtype)
```



---

### Функция ID: 74

**Исходный код:**
```python
def argmin(self, skipna: bool = True) -> int:
        """
        Return the index of minimum value.

        In case of multiple occurrences of the minimum value, the index
        corresponding to the first occurrence is returned.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        int

        See Also
        --------
        ExtensionArray.argmax : Return the index of the maximum value.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argmin()
        np.int64(1)
        """
        # Implementer note: You have two places to override the behavior of
        # argmin.
        # 1. _values_for_argsort : construct the values used in nargminmax
        # 2. argmin itself : total control over sorting.
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        return nargminmax(self, "argmin")
```



---

### Функция ID: 75

**Исходный код:**
```python
def map(self, mapper, na_action: Literal["ignore"] | None = None) -> Self:
        """
        Map categories using an input mapping or function.

        Parameters
        ----------
        mapper : dict, Series, callable
            The correspondence from old values to new.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        SparseArray
            The output array will have the same density as the input.
            The output fill value will be the result of applying the
            mapping to ``self.fill_value``

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 1, 2])
        >>> arr.map(lambda x: x + 10)
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map({0: 10, 1: 11, 2: 12})
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map(pd.Series([10, 11, 12], index=[0, 1, 2]))
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)
        """
        is_map = isinstance(mapper, (abc.Mapping, ABCSeries))

        fill_val = self.fill_value

        if na_action is None or notna(fill_val):
            fill_val = mapper.get(fill_val, fill_val) if is_map else mapper(fill_val)

        def func(sp_val):
            new_sp_val = mapper.get(sp_val, None) if is_map else mapper(sp_val)
            # check identity and equality because nans are not equal to each other
            if new_sp_val is fill_val or new_sp_val == fill_val:
                msg = "fill value in the sparse values not supported"
                raise ValueError(msg)
            return new_sp_val

        sp_values = [func(x) for x in self.sp_values]

        return type(self)(sp_values, sparse_index=self.sp_index, fill_value=fill_val)
```



---

### Функция ID: 76

**Исходный код:**
```python
def _reorder_fw_output(self) -> None:
        """
        Before the pass, fw_gm returns (*fw_outputs, *intermediates1)
        and bw_gm takes (*intermediates2, *grad_fw_outputs) as input.
        intermediates1 and intermediates2 share the same node names but
        they might be in different order. E.g. this could happen if there
        are inputs that contain symints.

        To simplify downstream processing, this graph pass normalizes the output of fw_gm
        to be consistent with the bacwkard inputs:

        fw_gm:
          - input: fw_args
          - output: (*fw_outputs, *intermediates)

        bw_gm:
          - input: (*intermediates, *grad_fw_outputs)
          - output: grad_fw_args

        Example:

        def fw_gm(x, y, z):
           a, b, c = f(x), g(y), k(z)
           return a, b, c, f_tmp, g_tmp, k_tmp

        , where a, b, c are fw_outputs, f_tmp, g_tmp, k_tmp are intermediates

        The corresponding bw_gm has the following signature:

        def bw_gm(f_tmp, g_tmp, k_tmp, grad_a, grad_b, grac):
          return grad_x, grad_y, grad_z
        """
        fw_gm_output_nodes = _find_hop_subgraph_outputs(self.fw_gm)
        fw_outputs_nodes = fw_gm_output_nodes[: self.n_fw_outputs]
        fw_intermediates_nodes = fw_gm_output_nodes[self.n_fw_outputs :]
        if len(fw_intermediates_nodes) > 0:
            fw_intermediates_name_to_node = {n.name: n for n in fw_intermediates_nodes}

            # First n_intermediates placeholders
            bw_names: list[str] = [
                ph.name
                for ph in list(self.bw_gm.graph.find_nodes(op="placeholder"))[
                    : self.n_intermediates
                ]
            ]
            new_fw_outputs = list(fw_outputs_nodes) + [
                fw_intermediates_name_to_node[name] for name in bw_names
            ]

            output_node = self.fw_gm.graph.find_nodes(op="output")[0]
            output_node.args = (tuple(new_fw_outputs),)

            self.fw_gm.graph.lint()
            self.fw_gm.recompile()
```



---

### Функция ID: 77

**Исходный код:**
```python
def equals(self, other: object) -> bool:
        """
        Return if another array is equivalent to this array.

        Equivalent means that both arrays have the same shape and dtype, and
        all values compare equal. Missing values in the same location are
        considered equal (in contrast with normal equality).

        Parameters
        ----------
        other : ExtensionArray
            Array to compare to this Array.

        Returns
        -------
        boolean
            Whether the arrays are equivalent.

        See Also
        --------
        numpy.array_equal : Equivalent method for numpy array.
        Series.equals : Equivalent method for Series.
        DataFrame.equals : Equivalent method for DataFrame.

        Examples
        --------
        >>> arr1 = pd.array([1, 2, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        True

        >>> arr1 = pd.array([1, 3, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        False
        """
        if type(self) != type(other):
            return False
        other = cast(ExtensionArray, other)
        if self.dtype != other.dtype:
            return False
        elif len(self) != len(other):
            return False
        else:
            equal_values = self == other
            if isinstance(equal_values, ExtensionArray):
                # boolean array with NA -> fill with False
                equal_values = equal_values.fillna(False)
            # error: Unsupported left operand type for & ("ExtensionArray")
            equal_na = self.isna() & other.isna()  # type: ignore[operator]
            return bool((equal_values | equal_na).all())
```



---

### Функция ID: 78

**Исходный код:**
```python
def isetitem(self, loc, value) -> None:
        """
        Set the given value in the column with position `loc`.

        This is a positional analogue to ``__setitem__``.

        Parameters
        ----------
        loc : int or sequence of ints
            Index position for the column.
        value : scalar or arraylike
            Value(s) for the column.

        See Also
        --------
        DataFrame.iloc : Purely integer-location based indexing for selection by
            position.

        Notes
        -----
        ``frame.isetitem(loc, value)`` is an in-place method as it will
        modify the DataFrame in place (not returning a new object). In contrast to
        ``frame.iloc[:, i] = value`` which will try to update the existing values in
        place, ``frame.isetitem(loc, value)`` will not update the values of the column
        itself in place, it will instead insert a new array.

        In cases where ``frame.columns`` is unique, this is equivalent to
        ``frame[frame.columns[i]] = value``.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.isetitem(1, [5, 6])
        >>> df
              A  B
        0     1  5
        1     2  6
        """
        if isinstance(value, DataFrame):
            if is_integer(loc):
                loc = [loc]

            if len(loc) != len(value.columns):
                raise ValueError(
                    f"Got {len(loc)} positions but value has {len(value.columns)} "
                    f"columns."
                )

            for i, idx in enumerate(loc):
                arraylike, refs = self._sanitize_column(value.iloc[:, i])
                self._iset_item_mgr(idx, arraylike, inplace=False, refs=refs)
            return

        arraylike, refs = self._sanitize_column(value)
        self._iset_item_mgr(loc, arraylike, inplace=False, refs=refs)
```



---

### Функция ID: 79

**Исходный код:**
```python
def view(self, dtype: Dtype | None = None) -> ArrayLike:
        """
        Return a view on the array.

        Parameters
        ----------
        dtype : str, np.dtype, or ExtensionDtype, optional
            Default None.

        Returns
        -------
        ExtensionArray or np.ndarray
            A view on the :class:`ExtensionArray`'s data.

        See Also
        --------
        api.extensions.ExtensionArray.ravel: Return a flattened view on input array.
        Index.view: Equivalent function for Index.
        ndarray.view: New view of array with the same data.

        Examples
        --------
        This gives view on the underlying data of an ``ExtensionArray`` and is not a
        copy. Modifications on either the view or the original ``ExtensionArray``
        will be reflected on the underlying data:

        >>> arr = pd.array([1, 2, 3])
        >>> arr2 = arr.view()
        >>> arr[0] = 2
        >>> arr2
        <IntegerArray>
        [2, 2, 3]
        Length: 3, dtype: Int64
        """
        # NB:
        # - This must return a *new* object referencing the same data, not self.
        # - The only case that *must* be implemented is with dtype=None,
        #   giving a view with the same dtype as self.
        if dtype is not None:
            raise NotImplementedError(dtype)
        return self[:]
```



---

### Функция ID: 80

**Исходный код:**
```python
def argsort(
        self,
        axis: Axis = 0,
        kind: SortKind = "quicksort",
        order: None = None,
        stable: None = None,
    ) -> Series:
        """
        Return the integer indices that would sort the Series values.

        Override ndarray.argsort. Argsorts the value, omitting NA/null values,
        and places the result in the same locations as the non-NA values.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        kind : {'mergesort', 'quicksort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable algorithms.
        order : None
            Has no effect but is accepted for compatibility with numpy.
        stable : None
            Has no effect but is accepted for compatibility with numpy.

        Returns
        -------
        Series[np.intp]
            Positions of values within the sort order with -1 indicating
            nan values.

        See Also
        --------
        numpy.ndarray.argsort : Returns the indices that would sort this array.

        Examples
        --------
        >>> s = pd.Series([3, 2, 1])
        >>> s.argsort()
        0    2
        1    1
        2    0
        dtype: int64
        """
        if axis != -1:
            # GH#54257 We allow -1 here so that np.argsort(series) works
            self._get_axis_number(axis)

        result = self.array.argsort(kind=kind)

        res = self._constructor(
            result, index=self.index, name=self.name, dtype=np.intp, copy=False
        )
        return res.__finalize__(self, method="argsort")
```



---

### Функция ID: 81

**Исходный код:**
```python
def min(self, axis: AxisInt | None = None, skipna: bool = True, *args, **kwargs):
        """
        Return the minimum value of the Index.

        Parameters
        ----------
        axis : {None}
            Dummy argument for consistency with Series.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        scalar
            Minimum value.

        See Also
        --------
        Index.max : Return the maximum value of the object.
        Series.min : Return the minimum value in a Series.
        DataFrame.min : Return the minimum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.min()
        1

        >>> idx = pd.Index(["c", "b", "a"])
        >>> idx.min()
        'a'

        For a MultiIndex, the minimum is determined lexicographically.

        >>> idx = pd.MultiIndex.from_product([("a", "b"), (2, 1)])
        >>> idx.min()
        ('a', 1)
        """
        nv.validate_min(args, kwargs)
        nv.validate_minmax_axis(axis)

        if not len(self):
            return self._na_value

        if len(self) and self.is_monotonic_increasing:
            # quick check
            first = self[0]
            if not isna(first):
                return maybe_unbox_numpy_scalar(first)

        if not self._is_multi and self.hasnans:
            # Take advantage of cache
            mask = self._isnan
            if not skipna or mask.all():
                return self._na_value

        if not self._is_multi and not isinstance(self._values, np.ndarray):
            return self._values._reduce(name="min", skipna=skipna)

        return maybe_unbox_numpy_scalar(nanops.nanmin(self._values, skipna=skipna))
```



---

### Функция ID: 82

**Исходный код:**
```typescript
function useStateLike<S>(
  name: string,
  initialState: (() => S) | S
): [S, (update: ((prevState: S) => S) | S) => void] {
  const stateRef = useRefLike(
    name,
    // @ts-expect-error S type should never be function, but there's no way to tell that to TypeScript
    typeof initialState === 'function' ? initialState() : initialState
  );
  const setState = (update: ((prevState: S) => S) | S) => {
    // @ts-expect-error S type should never be function, but there's no way to tell that to TypeScript
    stateRef.current = typeof update === 'function' ? update(stateRef.current) : update;
    triggerUpdate();
  };
  return [stateRef.current, setState];
}
```



---

### Функция ID: 83

**Исходный код:**
```python
def append(self, other: Index | Sequence[Index]) -> Index:
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices
            Single Index or a collection of indices, which can be either a list or a
            tuple.

        Returns
        -------
        Index
            Returns a new Index object resulting from appending the provided other
            indices to the original Index.

        See Also
        --------
        Index.insert : Make new Index inserting new item at location.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.append(pd.Index([4]))
        Index([1, 2, 3, 4], dtype='int64')
        """
        to_concat = [self]

        if isinstance(other, (list, tuple)):
            to_concat += list(other)
        else:
            # error: Argument 1 to "append" of "list" has incompatible type
            # "Union[Index, Sequence[Index]]"; expected "Index"
            to_concat.append(other)  # type: ignore[arg-type]

        for obj in to_concat:
            if not isinstance(obj, Index):
                raise TypeError("all inputs must be Index")

        names = {obj.name for obj in to_concat}
        name = None if len(names) > 1 else self.name

        return self._concat(to_concat, name)
```



---

### Функция ID: 84

**Исходный код:**
```typescript
function deriveActualIndentationFromList(list: readonly Node[], index: number, sourceFile: SourceFile, options: EditorSettings): number {

        Debug.assert(index >= 0 && index < list.length);

        const node = list[index];



        // walk toward the start of the list starting from current node and check if the line is the same for all items.

        // if end line for item [i - 1] differs from the start line for item [i] - find column of the first non-whitespace character on the line of item [i]

        let lineAndCharacter = getStartLineAndCharacterForNode(node, sourceFile);

        for (let i = index - 1; i >= 0; i--) {

            if (list[i].kind === SyntaxKind.CommaToken) {

                continue;

            }

            // skip list items that ends on the same line with the current list element

            const prevEndLine = sourceFile.getLineAndCharacterOfPosition(list[i].end).line;

            if (prevEndLine !== lineAndCharacter.line) {

                return findColumnForFirstNonWhitespaceCharacterInLine(lineAndCharacter, sourceFile, options);

            }



            lineAndCharacter = getStartLineAndCharacterForNode(list[i], sourceFile);

        }

        return Value.Unknown;

    }
```



---

### Функция ID: 85

**Исходный код:**
```python
def _justify(
    head: list[Sequence[str]], tail: list[Sequence[str]]
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """
    Justify items in head and tail, so they are right-aligned when stacked.

    Parameters
    ----------
    head : list-like of list-likes of strings
    tail : list-like of list-likes of strings

    Returns
    -------
    tuple of list of tuples of strings
        Same as head and tail, but items are right aligned when stacked
        vertically.

    Examples
    --------
    >>> _justify([["a", "b"]], [["abc", "abcd"]])
    ([('  a', '   b')], [('abc', 'abcd')])
    """
    combined = head + tail

    # For each position for the sequences in ``combined``,
    # find the length of the largest string.
    max_length = [0] * len(combined[0])
    for inner_seq in combined:
        length = [len(item) for item in inner_seq]
        max_length = [max(x, y) for x, y in zip(max_length, length, strict=True)]

    # justify each item in each list-like in head and tail using max_length
    head_tuples = [
        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length, strict=True))
        for seq in head
    ]
    tail_tuples = [
        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length, strict=True))
        for seq in tail
    ]
    return head_tuples, tail_tuples
```



---

### Функция ID: 86

**Исходный код:**
```python
def at_time(self, time, asof: bool = False, axis: Axis | None = None) -> Self:
        """
        Select values at particular time of day (e.g., 9:30AM).

        Parameters
        ----------
        time : datetime.time or str
            The values to select.
        asof : bool, default False
            This parameter is currently not supported.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            For `Series` this parameter is unused and defaults to 0.

        Returns
        -------
        Series or DataFrame
            The values with the specified time.

        Raises
        ------
        TypeError
            If the index is not  a :class:`DatetimeIndex`

        See Also
        --------
        between_time : Select values between particular times of the day.
        first : Select initial periods of time series based on a date offset.
        last : Select final periods of time series based on a date offset.
        DatetimeIndex.indexer_at_time : Get just the index locations for
            values at particular time of the day.

        Examples
        --------
        >>> i = pd.date_range("2018-04-09", periods=4, freq="12h")
        >>> ts = pd.DataFrame({"A": [1, 2, 3, 4]}, index=i)
        >>> ts
                             A
        2018-04-09 00:00:00  1
        2018-04-09 12:00:00  2
        2018-04-10 00:00:00  3
        2018-04-10 12:00:00  4

        >>> ts.at_time("12:00")
                             A
        2018-04-09 12:00:00  2
        2018-04-10 12:00:00  4
        """
        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)

        index = self._get_axis(axis)

        if not isinstance(index, DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")

        indexer = index.indexer_at_time(time, asof=asof)
        return self.take(indexer, axis=axis)
```



---

### Функция ID: 87

**Исходный код:**
```python
def equals(self, other) -> bool:
        """
        Return if another array is equivalent to this array.

        Equivalent means that both arrays have the same shape and dtype, and
        all values compare equal. Missing values in the same location are
        considered equal (in contrast with normal equality).

        Parameters
        ----------
        other : ExtensionArray
            Array to compare to this Array.

        Returns
        -------
        boolean
            Whether the arrays are equivalent.

        See Also
        --------
        numpy.array_equal : Equivalent method for numpy array.
        Series.equals : Equivalent method for Series.
        DataFrame.equals : Equivalent method for DataFrame.

        Examples
        --------
        >>> arr1 = pd.array([1, 2, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        True

        >>> arr1 = pd.array([1, 3, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        False
        """
        if type(self) != type(other):
            return False
        if other.dtype != self.dtype:
            return False

        # GH#44382 if e.g. self[1] is np.nan and other[1] is pd.NA, we are NOT
        #  equal.
        if not np.array_equal(self._mask, other._mask):
            return False

        left = self._data[~self._mask]
        right = other._data[~other._mask]
        return array_equivalent(left, right, strict_nan=True, dtype_equal=True)
```



---

### Функция ID: 88

**Исходный код:**
```python
def nanany(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> bool:
    """
    Check if any elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2])
    >>> nanops.nanany(s.values)
    np.True_

    >>> from pandas.core import nanops
    >>> s = pd.Series([np.nan])
    >>> nanops.nanany(s.values)
    np.False_
    """
    if values.dtype.kind in "iub" and mask is None:
        # GH#26032 fastpath
        # error: Incompatible return value type (got "Union[bool_, ndarray]",
        # expected "bool")
        return values.any(axis)  # type: ignore[return-value]

    if values.dtype.kind == "M":
        # GH#34479
        raise TypeError("datetime64 type does not support operation 'any'")

    values, _ = _get_values(values, skipna, fill_value=False, mask=mask)

    # For object type, any won't necessarily return
    # boolean values (numpy/numpy#4352)
    if values.dtype == object:
        values = values.astype(bool)

    # error: Incompatible return value type (got "Union[bool_, ndarray]", expected
    # "bool")
    return values.any(axis)  # type: ignore[return-value]
```



---

### Функция ID: 89

**Исходный код:**
```python
def set_caption(self, caption: str | tuple | list) -> Styler:
        """
        Set the text added to a ``<caption>`` HTML element.

        Parameters
        ----------
        caption : str, tuple, list
            For HTML output either the string input is used or the first element of the
            tuple. For LaTeX the string input provides a caption and the additional
            tuple input allows for full captions and short captions, in that order.

        Returns
        -------
        Styler
            Instance of class with text set for ``<caption>`` HTML element.

        See Also
        --------
        Styler.set_td_classes : Set the ``class`` attribute of ``<td>`` HTML elements.
        Styler.set_tooltips : Set the DataFrame of strings on ``Styler`` generating
            ``:hover`` tooltips.
        Styler.set_uuid : Set the uuid applied to ``id`` attributes of HTML elements.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.set_caption("test")  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        msg = "`caption` must be either a string or 2-tuple of strings."
        if isinstance(caption, (list, tuple)):
            if (
                len(caption) != 2
                or not isinstance(caption[0], str)
                or not isinstance(caption[1], str)
            ):
                raise ValueError(msg)
        elif not isinstance(caption, str):
            raise ValueError(msg)
        self.caption = caption
        return self
```



---

### Функция ID: 90

**Исходный код:**
```typescript
function getActualIndentationForListItem(node: Node, sourceFile: SourceFile, options: EditorSettings, listIndentsChild: boolean): number {

        if (node.parent && node.parent.kind === SyntaxKind.VariableDeclarationList) {

            // VariableDeclarationList has no wrapping tokens

            return Value.Unknown;

        }

        const containingList = getContainingList(node, sourceFile);

        if (containingList) {

            const index = containingList.indexOf(node);

            if (index !== -1) {

                const result = deriveActualIndentationFromList(containingList, index, sourceFile, options);

                if (result !== Value.Unknown) {

                    return result;

                }

            }

            return getActualIndentationForListStartLine(containingList, sourceFile, options) + (listIndentsChild ? options.indentSize! : 0); // TODO: GH#18217

        }

        return Value.Unknown;

    }
```



---

### Функция ID: 91

**Исходный код:**
```python
def iterrows(self) -> Iterable[tuple[Hashable, Series]]:
        """
        Iterate over DataFrame rows as (index, Series) pairs.

        Yields
        ------
        index : label or tuple of label
            The index of the row. A tuple for a `MultiIndex`.
        data : Series
            The data of the row as a Series.

        See Also
        --------
        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples of the values.
        DataFrame.items : Iterate over (column name, Series) pairs.

        Notes
        -----
        1. Because ``iterrows`` returns a Series for each row,
           it does **not** preserve dtypes across the rows (dtypes are
           preserved across columns for DataFrames).

           To preserve dtypes while iterating over the rows, it is better
           to use :meth:`itertuples` which returns namedtuples of the values
           and which is generally faster than ``iterrows``.

        2. You should **never modify** something you are iterating over.
           This is not guaranteed to work in all cases. Depending on the
           data types, the iterator returns a copy and not a view, and writing
           to it will have no effect.

        Examples
        --------

        >>> df = pd.DataFrame([[1, 1.5]], columns=["int", "float"])
        >>> row = next(df.iterrows())[1]
        >>> row
        int      1.0
        float    1.5
        Name: 0, dtype: float64
        >>> print(row["int"].dtype)
        float64
        >>> print(df["int"].dtype)
        int64
        """
        columns = self.columns
        klass = self._constructor_sliced
        for k, v in zip(self.index, self.values, strict=True):
            s = klass(v, index=columns, name=k).__finalize__(self)
            if self._mgr.is_single_block:
                s._mgr.add_references(self._mgr)
            yield k, s
```



---

### Функция ID: 92

**Исходный код:**
```python
def mask_zero_div_zero(x, y, result: np.ndarray) -> np.ndarray:
    """
    Set results of  0 // 0 to np.nan, regardless of the dtypes
    of the numerator or the denominator.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    result : ndarray

    Returns
    -------
    ndarray
        The filled result.

    Examples
    --------
    >>> x = np.array([1, 0, -1], dtype=np.int64)
    >>> x
    array([ 1,  0, -1])
    >>> y = 0  # int 0; numpy behavior is different with float
    >>> result = x // y
    >>> result  # raw numpy result does not fill division by zero
    array([0, 0, 0])
    >>> mask_zero_div_zero(x, y, result)
    array([ inf,  nan, -inf])
    """

    if not hasattr(y, "dtype"):
        # e.g. scalar, tuple
        y = np.array(y)
    if not hasattr(x, "dtype"):
        # e.g scalar, tuple
        x = np.array(x)

    zmask = y == 0

    if zmask.any():
        # Flip sign if necessary for -0.0
        zneg_mask = zmask & np.signbit(y)
        zpos_mask = zmask & ~zneg_mask

        x_lt0 = x < 0
        x_gt0 = x > 0
        nan_mask = zmask & (x == 0)
        neginf_mask = (zpos_mask & x_lt0) | (zneg_mask & x_gt0)
        posinf_mask = (zpos_mask & x_gt0) | (zneg_mask & x_lt0)

        if nan_mask.any() or neginf_mask.any() or posinf_mask.any():
            # Fill negative/0 with -inf, positive/0 with +inf, 0/0 with NaN
            result = result.astype("float64", copy=False)

            result[nan_mask] = np.nan
            result[posinf_mask] = np.inf
            result[neginf_mask] = -np.inf

    return result
```



---

### Функция ID: 93

**Исходный код:**
```python
def drop(
        self,
        labels: Index | np.ndarray | Iterable[Hashable],
        errors: IgnoreRaise = "raise",
    ) -> Index:
        """
        Make new Index with passed list of labels deleted.

        Parameters
        ----------
        labels : array-like or scalar
            Array-like object or a scalar value, representing the labels to be removed
            from the Index.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and existing labels are dropped.

        Returns
        -------
        Index
            Will be same type as self, except for RangeIndex.

        Raises
        ------
        KeyError
            If not all of the labels are found in the selected axis

        See Also
        --------
        Index.dropna : Return Index without NA/NaN values.
        Index.drop_duplicates : Return Index with duplicate values removed.

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.drop(["a"])
        Index(['b', 'c'], dtype='object')
        """
        if not isinstance(labels, Index):
            # avoid materializing e.g. RangeIndex
            arr_dtype = "object" if self.dtype == "object" else None
            labels = com.index_labels_to_array(labels, dtype=arr_dtype)

        indexer = self.get_indexer_for(labels)
        mask = indexer == -1
        if mask.any():
            if errors != "ignore":
                raise KeyError(f"{labels[mask].tolist()} not found in axis")
            indexer = indexer[~mask]
        return self.delete(indexer)
```



---

### Функция ID: 94

**Исходный код:**
```python
def sem(
        self, ddof: int = 1, numeric_only: bool = False, skipna: bool = True
    ) -> NDFrameT:
        """
        Compute standard error of the mean of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        skipna : bool, default True
            Exclude NA/null values. If an entire group is NA, the result will be NA.

            .. versionadded:: 3.0.0

        Returns
        -------
        Series or DataFrame
            Standard error of the mean of values within each group.

        See Also
        --------
        DataFrame.sem : Return unbiased standard error of the mean over requested axis.
        Series.sem : Return unbiased standard error of the mean over requested axis.

        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ["a", "a", "b", "b"]
        >>> ser = pd.Series([5, 10, 8, 14], index=lst)
        >>> ser
        a     5
        a    10
        b     8
        b    14
        dtype: int64
        >>> ser.groupby(level=0).sem()
        a    2.5
        b    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = [[1, 12, 11], [1, 15, 2], [2, 5, 8], [2, 6, 12]]
        >>> df = pd.DataFrame(
        ...     data,
        ...     columns=["a", "b", "c"],
        ...     index=["tuna", "salmon", "catfish", "goldfish"],
        ... )
        >>> df
                   a   b   c
            tuna   1  12  11
          salmon   1  15   2
         catfish   2   5   8
        goldfish   2   6  12
        >>> df.groupby("a").sem()
              b  c
        a
        1    1.5  4.5
        2    0.5  2.0

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 8],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").sem()
        2023-01-01    0.577350
        2023-02-01    1.527525
        Freq: MS, dtype: float64
        """
        if numeric_only and self.obj.ndim == 1 and not is_numeric_dtype(self.obj.dtype):
            raise TypeError(
                f"{type(self).__name__}.sem called with "
                f"numeric_only={numeric_only} and dtype {self.obj.dtype}"
            )
        return self._cython_agg_general(
            "sem",
            alt=lambda x: Series(x, copy=False).sem(ddof=ddof, skipna=skipna),
            numeric_only=numeric_only,
            ddof=ddof,
            skipna=skipna,
        )
```



---

### Функция ID: 95

**Исходный код:**
```java
private void addBucketToResult(long index, long count, boolean isPositive) {
        if (resultAlreadyReturned) {
            // we cannot modify the result anymore, create a new one
            reallocateResultWithCapacity(result.getCapacity(), true);
        }
        assert resultAlreadyReturned == false;
        boolean sufficientCapacity = result.tryAddBucket(index, count, isPositive);
        if (sufficientCapacity == false) {
            int newCapacity = Math.max(result.getCapacity() * 2, DEFAULT_ESTIMATED_BUCKET_COUNT);
            reallocateResultWithCapacity(newCapacity, true);
            boolean bucketAdded = result.tryAddBucket(index, count, isPositive);
            assert bucketAdded : "Output histogram should have enough capacity";
        }
    }
```



---

### Функция ID: 96

**Исходный код:**
```python
def filter(self, func, dropna: bool = True, *args, **kwargs):
        """
        Filter elements from groups that don't satisfy a criterion.

        Elements from groups are filtered if they do not satisfy the
        boolean criterion specified by func.

        Parameters
        ----------
        func : function
            Criterion to apply to each group. Should return True or False.
        dropna : bool, optional
            Drop groups that do not pass the filter. True by default; if False,
            groups that evaluate False are filled with NaNs.
        *args : tuple
            Optional positional arguments to pass to `func`.
        **kwargs : dict
            Optional keyword arguments to pass to `func`.

        Returns
        -------
        Series
            The filtered subset of the original Series.

        See Also
        --------
        Series.filter: Filter elements of ungrouped Series.
        DataFrameGroupBy.filter : Filter elements from groups base on criterion.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "A": ["foo", "bar", "foo", "bar", "foo", "bar"],
        ...         "B": [1, 2, 3, 4, 5, 6],
        ...         "C": [2.0, 5.0, 8.0, 1.0, 2.0, 9.0],
        ...     }
        ... )
        >>> grouped = df.groupby("A")
        >>> df.groupby("A").B.filter(lambda x: x.mean() > 3.0)
        1    2
        3    4
        5    6
        Name: B, dtype: int64
        """
        if isinstance(func, str):
            wrapper = lambda x: getattr(x, func)(*args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)

        # Interpret np.nan as False.
        def true_and_notna(x) -> bool:
            b = wrapper(x)
            return notna(b) and b

        try:
            indices = [
                self._get_index(name)
                for name, group in self._grouper.get_iterator(self._obj_with_exclusions)
                if true_and_notna(group)
            ]
        except (ValueError, TypeError) as err:
            raise TypeError("the filter must return a boolean result") from err

        filtered = self._apply_filter(indices, dropna)
        return filtered
```



---

### Функция ID: 97

**Исходный код:**
```java
static List<String> getStackFrameList(final Throwable throwable) {
        final String stackTrace = getStackTrace(throwable);
        final String linebreak = System.lineSeparator();
        final StringTokenizer frames = new StringTokenizer(stackTrace, linebreak);
        final List<String> list = new ArrayList<>();
        boolean traceStarted = false;
        while (frames.hasMoreTokens()) {
            final String token = frames.nextToken();
            // Determine if the line starts with "<whitespace>at"
            final int at = token.indexOf("at");
            if (at != NOT_FOUND && token.substring(0, at).trim().isEmpty()) {
                traceStarted = true;
                list.add(token);
            } else if (traceStarted) {
                break;
            }
        }
        return list;
    }
```



---

### Функция ID: 98

**Исходный код:**
```java
@Override
	public String toString() {
		if (this.resource instanceof FileSystemResource || this.resource instanceof FileUrlResource) {
			try {
				return "file [" + this.resource.getFile() + "]";
			}
			catch (IOException ex) {
				// Ignore
			}
		}
		return this.resource.toString();
	}
```



---

### Функция ID: 99

**Исходный код:**
```python
def ngroup(self, ascending: bool = True):
        """
        Number each group from 0 to the number of groups - 1.

        This is the enumerative complement of cumcount.  Note that the
        numbers given to the groups match the order in which the groups
        would be seen when iterating over the groupby object, not the
        order they are first observed.

        Groups with missing keys (where `pd.isna()` is True) will be labeled with `NaN`
        and will be skipped from the count.

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from number of group - 1 to 0.

        Returns
        -------
        Series
            Unique numbers for each group.

        See Also
        --------
        .cumcount : Number the rows in each group.

        Examples
        --------
        >>> df = pd.DataFrame({"color": ["red", None, "red", "blue", "blue", "red"]})
        >>> df
          color
        0   red
        1   NaN
        2   red
        3  blue
        4  blue
        5   red
        >>> df.groupby("color").ngroup()
        0    1.0
        1    NaN
        2    1.0
        3    0.0
        4    0.0
        5    1.0
        dtype: float64
        >>> df.groupby("color", dropna=False).ngroup()
        0    1
        1    2
        2    1
        3    0
        4    0
        5    1
        dtype: int64
        >>> df.groupby("color", dropna=False).ngroup(ascending=False)
        0    1
        1    0
        2    1
        3    2
        4    2
        5    1
        dtype: int64
        """
        obj = self._obj_with_exclusions
        index = obj.index
        comp_ids = self._grouper.ids

        dtype: type
        if self._grouper.has_dropped_na:
            comp_ids = np.where(comp_ids == -1, np.nan, comp_ids)
            dtype = np.float64
        else:
            dtype = np.int64

        if any(ping._passed_categorical for ping in self._grouper.groupings):
            # comp_ids reflect non-observed groups, we need only observed
            comp_ids = rank_1d(comp_ids, ties_method="dense") - 1

        result = self._obj_1d_constructor(comp_ids, index, dtype=dtype)
        if not ascending:
            result = self.ngroups - 1 - result
        return result
```



---

### Функция ID: 100

**Исходный код:**
```python
def itertuples(
        self, index: bool = True, name: str | None = "Pandas"
    ) -> Iterable[tuple[Any, ...]]:
        """
        Iterate over DataFrame rows as namedtuples.

        Parameters
        ----------
        index : bool, default True
            If True, return the index as the first element of the tuple.
        name : str or None, default "Pandas"
            The name of the returned namedtuples or None to return regular
            tuples.

        Returns
        -------
        iterator
            An object to iterate over namedtuples for each row in the
            DataFrame with the first field possibly being the index and
            following fields being the column values.

        See Also
        --------
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series)
            pairs.
        DataFrame.items : Iterate over (column name, Series) pairs.

        Notes
        -----
        The column names will be renamed to positional names if they are
        invalid Python identifiers, repeated, or start with an underscore.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"num_legs": [4, 2], "num_wings": [0, 2]}, index=["dog", "hawk"]
        ... )
        >>> df
              num_legs  num_wings
        dog          4          0
        hawk         2          2
        >>> for row in df.itertuples():
        ...     print(row)
        Pandas(Index='dog', num_legs=4, num_wings=0)
        Pandas(Index='hawk', num_legs=2, num_wings=2)

        By setting the `index` parameter to False we can remove the index
        as the first element of the tuple:

        >>> for row in df.itertuples(index=False):
        ...     print(row)
        Pandas(num_legs=4, num_wings=0)
        Pandas(num_legs=2, num_wings=2)

        With the `name` parameter set we set a custom name for the yielded
        namedtuples:

        >>> for row in df.itertuples(name="Animal"):
        ...     print(row)
        Animal(Index='dog', num_legs=4, num_wings=0)
        Animal(Index='hawk', num_legs=2, num_wings=2)
        """
        arrays = []
        fields = list(self.columns)
        if index:
            arrays.append(self.index)
            fields.insert(0, "Index")

        # use integer indexing because of possible duplicate column names
        arrays.extend(self.iloc[:, k] for k in range(len(self.columns)))

        if name is not None:
            # https://github.com/python/mypy/issues/9046
            # error: namedtuple() expects a string literal as the first argument
            itertuple = collections.namedtuple(  # type: ignore[misc]
                name, fields, rename=True
            )
            return map(itertuple._make, zip(*arrays, strict=True))

        # fallback to regular tuples
        return zip(*arrays, strict=True)
```



---

### Функция ID: 101

**Исходный код:**
```python
def zfill(self, width: int):
        """
        Pad strings in the Series/Index by prepending '0' characters.

        Strings in the Series/Index are padded with '0' characters on the
        left of the string to reach a total string length  `width`. Strings
        in the Series/Index with length greater or equal to `width` are
        unchanged.

        Parameters
        ----------
        width : int
            Minimum length of resulting string; strings with length less
            than `width` be prepended with '0' characters.

        Returns
        -------
        Series/Index of objects.
            A Series or Index where the strings are prepended with '0' characters.

        See Also
        --------
        Series.str.rjust : Fills the left side of strings with an arbitrary
            character.
        Series.str.ljust : Fills the right side of strings with an arbitrary
            character.
        Series.str.pad : Fills the specified sides of strings with an arbitrary
            character.
        Series.str.center : Fills both sides of strings with an arbitrary
            character.

        Notes
        -----
        Differs from :meth:`str.zfill` which has special handling
        for '+'/'-' in the string.

        Examples
        --------
        >>> s = pd.Series(["-1", "1", "1000", 10, np.nan])
        >>> s
        0      -1
        1       1
        2    1000
        3      10
        4     NaN
        dtype: object

        Note that ``10`` and ``NaN`` are not strings, therefore they are
        converted to ``NaN``. The minus sign in ``'-1'`` is treated as a
        special character and the zero is added to the right of it
        (:meth:`str.zfill` would have moved it to the left). ``1000``
        remains unchanged as it is longer than `width`.

        >>> s.str.zfill(3)
        0     -01
        1     001
        2    1000
        3     NaN
        4     NaN
        dtype: object
        """
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_zfill(width)
        return self._wrap_result(result)
```



---

### Функция ID: 102

**Исходный код:**
```python
def to_pickle(
    obj: Any,
    filepath_or_buffer: FilePath | WriteBuffer[bytes],
    compression: CompressionOptions = "infer",
    protocol: int = pickle.HIGHEST_PROTOCOL,
    storage_options: StorageOptions | None = None,
) -> None:
    """
    Pickle (serialize) object to file.

    Parameters
    ----------
    obj : any object
        Any python object.
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``write()`` function.
        Also accepts URL. URL has to be of S3 or GCS.
    {compression_options}

    protocol : int
        Int which indicates which protocol should be used by the pickler,
        default HIGHEST_PROTOCOL (see [1], paragraph 12.1.2). The possible
        values for this parameter depend on the version of Python. For Python
        2.x, possible values are 0, 1, 2. For Python>=3.0, 3 is a valid value.
        For Python >= 3.4, 4 is a valid value. A negative value for the
        protocol parameter is equivalent to setting its value to
        HIGHEST_PROTOCOL.

    {storage_options}

        .. [1] https://docs.python.org/3/library/pickle.html

    See Also
    --------
    read_pickle : Load pickled pandas object (or any object) from file.
    DataFrame.to_hdf : Write DataFrame to an HDF5 file.
    DataFrame.to_sql : Write DataFrame to a SQL database.
    DataFrame.to_parquet : Write a DataFrame to the binary parquet format.

    Examples
    --------
    >>> original_df = pd.DataFrame(
    ...     {{"foo": range(5), "bar": range(5, 10)}}
    ... )  # doctest: +SKIP
    >>> original_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> pd.to_pickle(original_df, "./dummy.pkl")  # doctest: +SKIP

    >>> unpickled_df = pd.read_pickle("./dummy.pkl")  # doctest: +SKIP
    >>> unpickled_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    """
    if protocol < 0:
        protocol = pickle.HIGHEST_PROTOCOL

    with get_handle(
        filepath_or_buffer,
        "wb",
        compression=compression,
        is_text=False,
        storage_options=storage_options,
    ) as handles:
        # letting pickle write directly to the buffer is more memory-efficient
        pickle.dump(obj, handles.handle, protocol=protocol)
```



---

### Функция ID: 103

**Исходный код:**
```python
def recode_for_categories(
    codes: np.ndarray,
    old_categories,
    new_categories,
    *,
    copy: bool = True,
    warn: bool = False,
) -> np.ndarray:
    """
    Convert a set of codes for to a new set of categories

    Parameters
    ----------
    codes : np.ndarray
    old_categories, new_categories : Index
    copy: bool, default True
        Whether to copy if the codes are unchanged.
    warn : bool, default False
        Whether to warn on silent-NA mapping.

    Returns
    -------
    new_codes : np.ndarray[np.int64]

    Examples
    --------
    >>> old_cat = pd.Index(["b", "a", "c"])
    >>> new_cat = pd.Index(["a", "b"])
    >>> codes = np.array([0, 1, 1, 2])
    >>> recode_for_categories(codes, old_cat, new_cat, copy=True)
    array([ 1,  0,  0, -1], dtype=int8)
    """
    if len(old_categories) == 0:
        # All null anyway, so just retain the nulls
        if copy:
            return codes.copy()
        return codes
    elif new_categories.equals(old_categories):
        # Same categories, so no need to actually recode
        if copy:
            return codes.copy()
        return codes

    codes_in_old_cats = new_categories.get_indexer_for(old_categories)
    if warn:
        wrong = codes_in_old_cats == -1
        if wrong.any():
            warnings.warn(
                "Constructing a Categorical with a dtype and values containing "
                "non-null entries not in that dtype's categories is deprecated "
                "and will raise in a future version.",
                Pandas4Warning,
                stacklevel=find_stack_level(),
            )
    indexer = coerce_indexer_dtype(codes_in_old_cats, new_categories)
    new_codes = take_nd(indexer, codes, fill_value=-1)
    return new_codes
```



---

### Функция ID: 104

**Исходный код:**
```python
def poke(self, context: Context):
        """
        Check subscribed queue for messages and write them to xcom with the ``messages`` key.

        :param context: the context object
        :return: ``True`` if message is available or ``False``
        """
        message_batch: list[Any] = []

        # perform multiple SQS call to retrieve messages in series
        for _ in range(self.num_batches):
            response = self.poll_sqs(sqs_conn=self.hook.conn)
            messages = process_response(
                response,
                self.message_filtering,
                self.message_filtering_match_values,
                self.message_filtering_config,
            )

            if not messages:
                continue

            message_batch.extend(messages)

            if self.delete_message_on_reception:
                self.log.info("Deleting %d messages", len(messages))

                entries = [
                    {"Id": message["MessageId"], "ReceiptHandle": message["ReceiptHandle"]}
                    for message in messages
                ]
                response = self.hook.conn.delete_message_batch(QueueUrl=self.sqs_queue, Entries=entries)

                if "Successful" not in response:
                    raise AirflowException(f"Delete SQS Messages failed {response} for messages {messages}")
        if message_batch:
            context["ti"].xcom_push(key="messages", value=message_batch)
            return True
        return False
```



---

### Функция ID: 105

**Исходный код:**
```java
private @Nullable String extractMessage(@Nullable HttpEntity entity) {
		if (entity != null) {
			try {
				JSONObject error = getContentAsJson(entity);
				if (error.has("message")) {
					return error.getString("message");
				}
			}
			catch (Exception ex) {
				// Ignore
			}
		}
		return null;
	}
```



---

### Функция ID: 106

**Исходный код:**
```java
public String getShortPathName(String path) {
        String longPath = "\\\\?\\" + path;
        // first we get the length of the buffer needed
        final int length = kernel.GetShortPathNameW(longPath, null, 0);
        if (length == 0) {
            logger.warn("failed to get short path name: {}", kernel.GetLastError());
            return path;
        }
        final char[] shortPath = new char[length];
        // knowing the length of the buffer, now we get the short name
        if (kernel.GetShortPathNameW(longPath, shortPath, length) > 0) {
            assert shortPath[length - 1] == '\0';
            return new String(shortPath, 0, length - 1);
        } else {
            logger.warn("failed to get short path name: {}", kernel.GetLastError());
            return path;
        }
    }
```



---

### Функция ID: 107

**Исходный код:**
```python
def take(
        self,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> Self:
        """
        Return a new Index of the values selected by the indices.

        For internal compatibility with numpy arrays.

        Parameters
        ----------
        indices : array-like
            Indices to be taken.
        axis : {0 or 'index'}, optional
            The axis over which to select values, always 0 or 'index'.
        allow_fill : bool, default True
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any
              other negative values raise a ``ValueError``.

        fill_value : scalar, default None
            If allow_fill=True and fill_value is not None, indices specified by
            -1 are regarded as NA. If Index doesn't hold NA, raise ValueError.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        Index
            An index formed of elements at the given indices. Will be the same
            type as self, except for RangeIndex.

        See Also
        --------
        numpy.ndarray.take: Return an array formed from the
            elements of a at the given indices.

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.take([2, 2, 1, 2])
        Index(['c', 'c', 'b', 'c'], dtype='str')
        """
        if kwargs:
            nv.validate_take((), kwargs)
        if is_scalar(indices):
            raise TypeError("Expected indices to be array-like")
        indices = ensure_platform_int(indices)
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)

        if indices.ndim == 1 and lib.is_range_indexer(indices, len(self)):
            return self.copy()

        # Note: we discard fill_value and use self._na_value, only relevant
        #  in the case where allow_fill is True and fill_value is not None
        values = self._values
        if isinstance(values, np.ndarray):
            taken = algos.take(
                values, indices, allow_fill=allow_fill, fill_value=self._na_value
            )
        else:
            # algos.take passes 'axis' keyword which not all EAs accept
            taken = values.take(
                indices, allow_fill=allow_fill, fill_value=self._na_value
            )
        return self._constructor._simple_new(taken, name=self.name)
```



---

### Функция ID: 108

**Исходный код:**
```python
def between_time(
        self,
        start_time,
        end_time,
        inclusive: IntervalClosedType = "both",
        axis: Axis | None = None,
    ) -> Self:
        """
        Select values between particular times of the day (e.g., 9:00-9:30 AM).

        By setting ``start_time`` to be later than ``end_time``,
        you can get the times that are *not* between the two times.

        Parameters
        ----------
        start_time : datetime.time or str
            Initial time as a time filter limit.
        end_time : datetime.time or str
            End time as a time filter limit.
        inclusive : {"both", "neither", "left", "right"}, default "both"
            Include boundaries; whether to set each bound as closed or open.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Determine range time on index or columns value.
            For `Series` this parameter is unused and defaults to 0.

        Returns
        -------
        Series or DataFrame
            Data from the original object filtered to the specified dates range.

        Raises
        ------
        TypeError
            If the index is not  a :class:`DatetimeIndex`

        See Also
        --------
        at_time : Select values at a particular time of the day.
        first : Select initial periods of time series based on a date offset.
        last : Select final periods of time series based on a date offset.
        DatetimeIndex.indexer_between_time : Get just the index locations for
            values between particular times of the day.

        Examples
        --------
        >>> i = pd.date_range("2018-04-09", periods=4, freq="1D20min")
        >>> ts = pd.DataFrame({"A": [1, 2, 3, 4]}, index=i)
        >>> ts
                             A
        2018-04-09 00:00:00  1
        2018-04-10 00:20:00  2
        2018-04-11 00:40:00  3
        2018-04-12 01:00:00  4

        >>> ts.between_time("0:15", "0:45")
                             A
        2018-04-10 00:20:00  2
        2018-04-11 00:40:00  3

        You get the times that are *not* between two times by setting
        ``start_time`` later than ``end_time``:

        >>> ts.between_time("0:45", "0:15")
                             A
        2018-04-09 00:00:00  1
        2018-04-12 01:00:00  4
        """
        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)

        index = self._get_axis(axis)
        if not isinstance(index, DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")

        left_inclusive, right_inclusive = validate_inclusive(inclusive)
        indexer = index.indexer_between_time(
            start_time,
            end_time,
            include_start=left_inclusive,
            include_end=right_inclusive,
        )
        return self.take(indexer, axis=axis)
```



---

### Функция ID: 109

**Исходный код:**
```python
def get_schema_defaults(cls, object_type: str) -> dict[str, Any]:
        """
        Extract default values from JSON schema for any object type.

        :param object_type: The object type to get defaults for (e.g., "operator", "dag")
        :return: Dictionary of field name -> default value
        """
        # Load schema if needed (handles lazy loading)
        schema_loader = cls._json_schema

        if schema_loader is None:
            return {}

        # Access the schema definitions (trigger lazy loading)
        schema_data = schema_loader.schema
        object_def = schema_data.get("definitions", {}).get(object_type, {})
        properties = object_def.get("properties", {})

        defaults = {}
        for field_name, field_def in properties.items():
            if isinstance(field_def, dict) and "default" in field_def:
                defaults[field_name] = field_def["default"]

        return defaults
```



---

### Функция ID: 110

**Исходный код:**
```java
private void setBucket(long index, long count, boolean isPositive) {
        if (count < 1) {
            throw new IllegalArgumentException("Bucket count must be at least 1");
        }
        if (negativeBuckets == null && positiveBuckets == null) {
            // so far, all received buckets were in order, try to directly build the result
            if (result == null) {
                // Initialize the result buffer if required
                reallocateResultWithCapacity(estimatedBucketCount, false);
            }
            if ((isPositive && result.wasLastAddedBucketPositive() == false)
                || (isPositive == result.wasLastAddedBucketPositive() && index > result.getLastAddedBucketIndex())) {
                // the new bucket is in order too, we can directly add the bucket
                addBucketToResult(index, count, isPositive);
                return;
            }
        }
        // fallback to TreeMap if a bucket is received out of order
        initializeBucketTreeMapsIfNeeded();
        if (isPositive) {
            positiveBuckets.put(index, count);
        } else {
            negativeBuckets.put(index, count);
        }
    }
```



---

### Функция ID: 111

**Исходный код:**
```python
def first(
        self, numeric_only: bool = False, min_count: int = -1, skipna: bool = True
    ) -> NDFrameT:
        """
        Compute the first entry of each column within each group.

        Defaults to skipping NA elements.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` valid values are present the result will be NA.
        skipna : bool, default True
            Exclude NA/null values. If an entire group is NA, the result will be NA.

            .. versionadded:: 2.2.1

        Returns
        -------
        Series or DataFrame
            First values within each group.

        See Also
        --------
        DataFrame.groupby : Apply a function groupby to each row or column of a
            DataFrame.
        core.groupby.DataFrameGroupBy.last : Compute the last non-null entry
            of each column.
        core.groupby.DataFrameGroupBy.nth : Take the nth row from each group.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     dict(
        ...         A=[1, 1, 3],
        ...         B=[None, 5, 6],
        ...         C=[1, 2, 3],
        ...         D=["3/11/2000", "3/12/2000", "3/13/2000"],
        ...     )
        ... )
        >>> df["D"] = pd.to_datetime(df["D"])
        >>> df.groupby("A").first()
             B  C          D
        A
        1  5.0  1 2000-03-11
        3  6.0  3 2000-03-13
        >>> df.groupby("A").first(min_count=2)
            B    C          D
        A
        1 NaN  1.0 2000-03-11
        3 NaN  NaN        NaT
        >>> df.groupby("A").first(numeric_only=True)
             B  C
        A
        1  5.0  1
        3  6.0  3
        """

        def first_compat(obj: NDFrameT):
            def first(x: Series):
                """Helper function for first item that isn't NA."""
                arr = x.array[notna(x.array)]
                if not len(arr):
                    return x.array.dtype.na_value
                return arr[0]

            if isinstance(obj, DataFrame):
                return obj.apply(first)
            elif isinstance(obj, Series):
                return first(obj)
            else:  # pragma: no cover
                raise TypeError(type(obj))

        return self._agg_general(
            numeric_only=numeric_only,
            min_count=min_count,
            alias="first",
            npfunc=first_compat,
            skipna=skipna,
        )
```



---

### Функция ID: 112

**Исходный код:**
```python
def symmetric_difference(
        self,
        other,
        result_name: abc.Hashable | None = None,
        sort: bool | None = None,
    ):
        """
        Compute the symmetric difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
            Index or an array-like object with elements to compute the symmetric
            difference with the original Index.
        result_name : str
            A string representing the name of the resulting Index, if desired.
        sort : bool or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index
            Returns a new Index object containing elements that appear in either the
            original Index or the `other` Index, but not both.

        See Also
        --------
        Index.difference : Return a new Index with elements of index not in other.
        Index.union : Form the union of two Index objects.
        Index.intersection : Form the intersection of two Index objects.

        Notes
        -----
        ``symmetric_difference`` contains elements that appear in either
        ``idx1`` or ``idx2`` but not both. Equivalent to the Index created by
        ``idx1.difference(idx2) | idx2.difference(idx1)`` with duplicates
        dropped.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([2, 3, 4, 5])
        >>> idx1.symmetric_difference(idx2)
        Index([1, 5], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name_update = self._convert_can_do_setop(other)
        if result_name is None:
            result_name = result_name_update

        if self.dtype != other.dtype:
            self, other = self._dti_setop_align_tzs(other, "symmetric_difference")

        if not self._should_compare(other):
            return self.union(other, sort=sort).rename(result_name)

        elif self.dtype != other.dtype:
            dtype = self._find_common_type_compat(other)
            this = self.astype(dtype, copy=False)
            that = other.astype(dtype, copy=False)
            return this.symmetric_difference(that, sort=sort).rename(result_name)

        this = self.unique()
        other = other.unique()
        indexer = this.get_indexer_for(other)

        # {this} minus {other}
        common_indexer = indexer.take((indexer != -1).nonzero()[0])
        left_indexer = np.setdiff1d(
            np.arange(this.size), common_indexer, assume_unique=True
        )
        left_diff = this.take(left_indexer)

        # {other} minus {this}
        right_indexer = (indexer == -1).nonzero()[0]
        right_diff = other.take(right_indexer)

        res_values = left_diff.append(right_diff)
        result = _maybe_try_sort(res_values, sort)

        if not self._is_multi:
            return Index(result, name=result_name, dtype=res_values.dtype)
        else:
            left_diff = cast("MultiIndex", left_diff)
            if len(result) == 0:
                # result might be an Index, if other was an Index
                return left_diff.remove_unused_levels().set_names(result_name)
            return result.set_names(result_name)
```



---

### Функция ID: 113

**Исходный код:**
```python
def count(self) -> NDFrameT:
        """
        Compute count of group, excluding missing values.

        Returns
        -------
        Series or DataFrame
            Count of values within each group.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, np.nan], index=lst)
        >>> ser
        a    1.0
        a    2.0
        b    NaN
        dtype: float64
        >>> ser.groupby(level=0).count()
        a    2
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, np.nan, 3], [1, np.nan, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["cow", "horse", "bull"]
        ... )
        >>> df
                a	  b	c
        cow     1	NaN	3
        horse	1	NaN	6
        bull	7	8.0	9
        >>> df.groupby("a").count()
            b   c
        a
        1   0   2
        7   1   1

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").count()
        2023-01-01    2
        2023-02-01    2
        Freq: MS, dtype: int64
        """
        data = self._get_data_to_aggregate()
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        mask = ids != -1

        is_series = data.ndim == 1

        def hfunc(bvalues: ArrayLike) -> ArrayLike:
            # TODO(EA2D): reshape would not be necessary with 2D EAs
            if bvalues.ndim == 1:
                # EA
                masked = mask & ~isna(bvalues).reshape(1, -1)
            else:
                masked = mask & ~isna(bvalues)

            counted = lib.count_level_2d(masked, labels=ids, max_bin=ngroups)
            if isinstance(bvalues, BaseMaskedArray):
                return IntegerArray(
                    counted[0], mask=np.zeros(counted.shape[1], dtype=np.bool_)
                )
            elif isinstance(bvalues, ArrowExtensionArray) and not isinstance(
                bvalues.dtype, StringDtype
            ):
                dtype = pandas_dtype("int64[pyarrow]")
                return type(bvalues)._from_sequence(counted[0], dtype=dtype)
            if is_series:
                assert counted.ndim == 2
                assert counted.shape[0] == 1
                return counted[0]
            return counted

        new_mgr = data.grouped_reduce(hfunc)
        new_obj = self._wrap_agged_manager(new_mgr)
        result = self._wrap_aggregated_output(new_obj)

        return result
```



---

### Функция ID: 114

**Исходный код:**
```python
def get_filesystem_type(filepath: str):
    """
    Determine the type of filesystem used - we might want to use different parameters if tmpfs is used.
    :param filepath: path to check
    :return: type of filesystem
    """
    # We import it locally so that click autocomplete works
    try:
        import psutil
    except ImportError:
        return "unknown"

    root_type = "unknown"
    for part in psutil.disk_partitions(all=True):
        if part.mountpoint == "/":
            root_type = part.fstype
        elif filepath.startswith(part.mountpoint):
            return part.fstype

    return root_type
```



---

### Функция ID: 115

**Исходный код:**
```python
def size(self) -> DataFrame | Series:
        """
        Compute group sizes.

        Returns
        -------
        DataFrame or Series
            Number of rows in each group as a Series if as_index is True
            or a DataFrame if as_index is False.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a     1
        a     2
        b     3
        dtype: int64
        >>> ser.groupby(level=0).size()
        a    2
        b    1
        dtype: int64

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["owl", "toucan", "eagle"]
        ... )
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby("a").size()
        a
        1    2
        7    1
        dtype: int64

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 2, 3],
        ...     index=pd.DatetimeIndex(["2023-01-01", "2023-01-15", "2023-02-01"]),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        dtype: int64
        >>> ser.resample("MS").size()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        """
        result = self._grouper.size()
        dtype_backend: None | Literal["pyarrow", "numpy_nullable"] = None
        if isinstance(self.obj, Series):
            if isinstance(self.obj.array, ArrowExtensionArray):
                if isinstance(self.obj.array, ArrowStringArray):
                    if self.obj.array.dtype.na_value is np.nan:
                        dtype_backend = None
                    else:
                        dtype_backend = "numpy_nullable"
                else:
                    dtype_backend = "pyarrow"
            elif isinstance(self.obj.array, BaseMaskedArray):
                dtype_backend = "numpy_nullable"
        # TODO: For DataFrames what if columns are mixed arrow/numpy/masked?

        # GH28330 preserve subclassed Series/DataFrames through calls
        if isinstance(self.obj, Series):
            result = self._obj_1d_constructor(result, name=self.obj.name)
        else:
            result = self._obj_1d_constructor(result)

        if dtype_backend is not None:
            result = result.convert_dtypes(
                infer_objects=False,
                convert_string=False,
                convert_boolean=False,
                convert_floating=False,
                dtype_backend=dtype_backend,
            )

        if not self.as_index:
            result = result.rename("size").reset_index()
        return result
```



---

### Функция ID: 116

**Исходный код:**
```python
def lreshape(data: DataFrame, groups: dict, dropna: bool = True) -> DataFrame:
    """
    Reshape wide-format data to long. Generalized inverse of DataFrame.pivot.

    Accepts a dictionary, ``groups``, in which each key is a new column name
    and each value is a list of old column names that will be "melted" under
    the new column name as part of the reshape.

    Parameters
    ----------
    data : DataFrame
        The wide-format DataFrame.
    groups : dict
        {new_name : list_of_columns}.
    dropna : bool, default True
        Do not include columns whose entries are all NaN.

    Returns
    -------
    DataFrame
        Reshaped DataFrame.

    See Also
    --------
    melt : Unpivot a DataFrame from wide to long format, optionally leaving
        identifiers set.
    pivot : Create a spreadsheet-style pivot table as a DataFrame.
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.

    Examples
    --------
    >>> data = pd.DataFrame(
    ...     {
    ...         "hr1": [514, 573],
    ...         "hr2": [545, 526],
    ...         "team": ["Red Sox", "Yankees"],
    ...         "year1": [2007, 2007],
    ...         "year2": [2008, 2008],
    ...     }
    ... )
    >>> data
       hr1  hr2     team  year1  year2
    0  514  545  Red Sox   2007   2008
    1  573  526  Yankees   2007   2008

    >>> pd.lreshape(data, {"year": ["year1", "year2"], "hr": ["hr1", "hr2"]})
          team  year   hr
    0  Red Sox  2007  514
    1  Yankees  2007  573
    2  Red Sox  2008  545
    3  Yankees  2008  526
    """
    mdata = {}
    pivot_cols = []
    all_cols: set[Hashable] = set()
    K = len(next(iter(groups.values())))
    for target, names in groups.items():
        if len(names) != K:
            raise ValueError("All column lists must be same length")
        to_concat = [data[col]._values for col in names]

        mdata[target] = concat_compat(to_concat)
        pivot_cols.append(target)
        all_cols = all_cols.union(names)

    id_cols = list(data.columns.difference(all_cols))
    for col in id_cols:
        mdata[col] = np.tile(data[col]._values, K)

    if dropna:
        mask = np.ones(len(mdata[pivot_cols[0]]), dtype=bool)
        for c in pivot_cols:
            mask &= notna(mdata[c])
        if not mask.all():
            mdata = {k: v[mask] for k, v in mdata.items()}

    return data._constructor(mdata, columns=id_cols + pivot_cols)
```



---

### Функция ID: 117

**Исходный код:**
```python
def argmax(
        self, axis: AxisInt | None = None, skipna: bool = True, *args, **kwargs
    ) -> int:
        """
        Return int position of the {value} value in the Series.

        If the {op}imum is achieved in multiple locations,
        the first row position is returned.

        Parameters
        ----------
        axis : {{None}}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, or if ``skipna=False``
            and there is an NA value, this method will raise a ``ValueError``.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        int
            Row position of the {op}imum value.

        See Also
        --------
        Series.arg{op} : Return position of the {op}imum value.
        Series.arg{oppose} : Return position of the {oppose}imum value.
        numpy.ndarray.arg{op} : Equivalent method for numpy arrays.
        Series.idxmax : Return index label of the maximum values.
        Series.idxmin : Return index label of the minimum values.

        Examples
        --------
        Consider dataset containing cereal calories

        >>> s = pd.Series(
        ...     [100.0, 110.0, 120.0, 110.0],
        ...     index=[
        ...         "Corn Flakes",
        ...         "Almond Delight",
        ...         "Cinnamon Toast Crunch",
        ...         "Cocoa Puff",
        ...     ],
        ... )
        >>> s
        Corn Flakes              100.0
        Almond Delight           110.0
        Cinnamon Toast Crunch    120.0
        Cocoa Puff               110.0
        dtype: float64

        >>> s.argmax()
        np.int64(2)
        >>> s.argmin()
        np.int64(0)

        The maximum cereal calories is the third element and
        the minimum cereal calories is the first element,
        since series is zero-indexed.
        """
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)

        if isinstance(delegate, ExtensionArray):
            return delegate.argmax(skipna=skipna)
        else:
            result = nanops.nanargmax(delegate, skipna=skipna)
            # error: Incompatible return value type (got "Union[int, ndarray]", expected
            # "int")
            return result  # type: ignore[return-value]
```



---

### Функция ID: 118

**Исходный код:**
```java
private void maybeFailWithError() {
        if (!hasError()) {
            return;
        }
        // for ProducerFencedException, do not wrap it as a KafkaException
        // but create a new instance without the call trace since it was not thrown because of the current call
        if (lastError instanceof ProducerFencedException) {
            throw new ProducerFencedException("Producer with transactionalId '" + transactionalId
                    + "' and " + producerIdAndEpoch + " has been fenced by another producer " +
                    "with the same transactionalId");
        }
        if (lastError instanceof InvalidProducerEpochException) {
            throw new InvalidProducerEpochException("Producer with transactionalId '" + transactionalId
                    + "' and " + producerIdAndEpoch + " attempted to produce with an old epoch");
        }
        if (lastError instanceof IllegalStateException) {
            throw new IllegalStateException("Producer with transactionalId '" + transactionalId
                    + "' and " + producerIdAndEpoch + " cannot execute transactional method because of previous invalid state transition attempt", lastError);
        }
        throw new KafkaException("Cannot execute transactional method because we are in an error state", lastError);
    }
```



---

### Функция ID: 119

**Исходный код:**
```java
private String[] removeDebugFlags(String[] args) {
		List<String> rtn = new ArrayList<>(args.length);
		boolean appArgsDetected = false;
		for (String arg : args) {
			// Allow apps to have a --debug argument
			appArgsDetected |= "--".equals(arg);
			if ("--debug".equals(arg) && !appArgsDetected) {
				continue;
			}
			rtn.add(arg);
		}
		return StringUtils.toStringArray(rtn);
	}
```



---

### Функция ID: 120

**Исходный код:**
```python
def astype(self, dtype: Dtype, copy: bool = True):
        """
        Create an Index with values cast to dtypes.

        The class of a new Index is determined by dtype. When conversion is
        impossible, a TypeError exception is raised.

        Parameters
        ----------
        dtype : numpy dtype or pandas type
            Note that any signed integer `dtype` is treated as ``'int64'``,
            and any unsigned integer `dtype` is treated as ``'uint64'``,
            regardless of the size.
        copy : bool, default True
            By default, astype always returns a newly allocated object.
            If copy is set to False and internal requirements on dtype are
            satisfied, the original data is used to create a new Index
            or the original Index is returned.

        Returns
        -------
        Index
            Index with values cast to specified dtype.

        See Also
        --------
        Index.dtype: Return the dtype object of the underlying data.
        Index.dtypes: Return the dtype object of the underlying data.
        Index.convert_dtypes: Convert columns to the best possible dtypes.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.astype("float")
        Index([1.0, 2.0, 3.0], dtype='float64')
        """
        if dtype is not None:
            dtype = pandas_dtype(dtype)

        if self.dtype == dtype:
            # Ensure that self.astype(self.dtype) is self
            return self.copy() if copy else self

        values = self._data
        if isinstance(values, ExtensionArray):
            with rewrite_exception(type(values).__name__, type(self).__name__):
                new_values = values.astype(dtype, copy=copy)

        elif isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            # Note: for RangeIndex and CategoricalDtype self vs self._values
            #  behaves differently here.
            new_values = cls._from_sequence(self, dtype=dtype, copy=copy)

        else:
            # GH#13149 specifically use astype_array instead of astype
            new_values = astype_array(values, dtype=dtype, copy=copy)

        # pass copy=False because any copying will be done in the astype above
        result = Index(new_values, name=self.name, dtype=new_values.dtype, copy=False)
        if (
            not copy
            and self._references is not None
            and astype_is_view(self.dtype, dtype)
        ):
            result._references = self._references
            result._references.add_index_reference(result)
        return result
```



---

### Функция ID: 121

**Исходный код:**
```java
private void initializeBucketTreeMapsIfNeeded() {
        if (negativeBuckets == null) {
            negativeBuckets = new TreeMap<>();
            positiveBuckets = new TreeMap<>();
            // copy existing buckets to the maps
            if (result != null) {
                BucketIterator it = result.negativeBuckets().iterator();
                while (it.hasNext()) {
                    negativeBuckets.put(it.peekIndex(), it.peekCount());
                    it.advance();
                }
                it = result.positiveBuckets().iterator();
                while (it.hasNext()) {
                    positiveBuckets.put(it.peekIndex(), it.peekCount());
                    it.advance();
                }
            }
        }
    }
```



---

### Функция ID: 122

**Исходный код:**
```python
def is_wsl2() -> bool:
    """
    Check if the current platform is WSL2. This method will exit with error printing appropriate
    message if WSL1 is detected as WSL1 is not supported.

    :return: True if the current platform is WSL2, False otherwise (unless it's WSL1 then it exits).
    """
    if not sys.platform.startswith("linux"):
        return False
    release_name = platform.uname().release
    has_wsl_interop = _exists_no_permission_error("/proc/sys/fs/binfmt_misc/WSLInterop")
    microsoft_in_release = "microsoft" in release_name.lower()
    wsl_conf = _exists_no_permission_error("/etc/wsl.conf")
    if not has_wsl_interop and not microsoft_in_release and not wsl_conf:
        return False
    if microsoft_in_release:
        # Release name WSL1 detection
        if "Microsoft" in release_name:
            message_on_wsl1_detected(release_name=release_name, kernel_version=None)
            sys.exit(1)
        return True

    # Kernel WSL1 detection
    kernel_version: tuple[int, ...] = (0, 0)
    if len(parts := release_name.split(".", 2)[:2]) == 2:
        with contextlib.suppress(TypeError, ValueError):
            kernel_version = tuple(map(int, parts))
    if kernel_version < (4, 19):
        message_on_wsl1_detected(release_name=None, kernel_version=kernel_version)
        sys.exit(1)
    return True
```



---

### Функция ID: 123

**Исходный код:**
```python
def render_dag(dag: DAG | SerializedDAG, tis: list[TaskInstance] | None = None) -> graphviz.Digraph:
    """
    Render the DAG object to the DOT object.

    If an task instance list is passed, the nodes will be painted according to task statuses.

    :param dag: DAG that will be rendered.
    :param tis: List of task instances
    :return: Graphviz object
    """
    if not graphviz:
        raise AirflowException(
            "Could not import graphviz. Install the graphviz python package to fix this error."
        )
    dot = graphviz.Digraph(
        dag.dag_id,
        graph_attr={
            "rankdir": "LR",
            "labelloc": "t",
            "label": dag.dag_id,
        },
    )
    states_by_task_id = None
    if tis is not None:
        states_by_task_id = {ti.task_id: ti.state for ti in tis}

    _draw_nodes(dag.task_group, dot, states_by_task_id)

    for edge in dag_edges(dag):
        # Gets an optional label for the edge; this will be None if none is specified.
        label = dag.get_edge_info(edge["source_id"], edge["target_id"]).get("label")
        # Add the edge to the graph with optional label
        # (we can just use the maybe-None label variable directly)
        dot.edge(edge["source_id"], edge["target_id"], label)

    return dot
```



---

### Функция ID: 124

**Исходный код:**
```java
protected @Nullable String pendingToString() {
    // TODO(diamondm) consider moving this into addPendingString so it's always in the output
    if (this instanceof ScheduledFuture) {
      return "remaining delay=[" + ((ScheduledFuture) this).getDelay(MILLISECONDS) + " ms]";
    }
    return null;
  }
```



---

### Функция ID: 125

**Исходный код:**
```javascript
function replace() {
      var args = arguments,
          string = toString(args[0]);

      return args.length < 3 ? string : string.replace(args[1], args[2]);
    }
```



---

### Функция ID: 126

**Исходный код:**
```python
def take(
    arr,
    indices: TakeIndexer,
    axis: AxisInt = 0,
    allow_fill: bool = False,
    fill_value=None,
):
    """
    Take elements from an array.

    Parameters
    ----------
    arr : numpy.ndarray, ExtensionArray, Index, or Series
        Input array.
    indices : sequence of int or one-dimensional np.ndarray of int
        Indices to be taken.
    axis : int, default 0
        The axis over which to select values.
    allow_fill : bool, default False
        How to handle negative values in `indices`.

        * False: negative values in `indices` indicate positional indices
          from the right (the default). This is similar to :func:`numpy.take`.

        * True: negative values in `indices` indicate
          missing values. These values are set to `fill_value`. Any other
          negative values raise a ``ValueError``.

    fill_value : any, optional
        Fill value to use for NA-indices when `allow_fill` is True.
        This may be ``None``, in which case the default NA value for
        the type (``self.dtype.na_value``) is used.

        For multi-dimensional `arr`, each *element* is filled with
        `fill_value`.

    Returns
    -------
    ndarray or ExtensionArray
        Same type as the input.

    Raises
    ------
    IndexError
        When `indices` is out of bounds for the array.
    ValueError
        When the indexer contains negative values other than ``-1``
        and `allow_fill` is True.

    Notes
    -----
    When `allow_fill` is False, `indices` may be whatever dimensionality
    is accepted by NumPy for `arr`.

    When `allow_fill` is True, `indices` should be 1-D.

    See Also
    --------
    numpy.take : Take elements from an array along an axis.

    Examples
    --------
    >>> import pandas as pd

    With the default ``allow_fill=False``, negative numbers indicate
    positional indices from the right.

    >>> pd.api.extensions.take(np.array([10, 20, 30]), [0, 0, -1])
    array([10, 10, 30])

    Setting ``allow_fill=True`` will place `fill_value` in those positions.

    >>> pd.api.extensions.take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True)
    array([10., 10., nan])

    >>> pd.api.extensions.take(
    ...     np.array([10, 20, 30]), [0, 0, -1], allow_fill=True, fill_value=-10
    ... )
    array([ 10,  10, -10])
    """
    if not isinstance(
        arr,
        (np.ndarray, ABCExtensionArray, ABCIndex, ABCSeries, ABCNumpyExtensionArray),
    ):
        # GH#52981
        raise TypeError(
            "pd.api.extensions.take requires a numpy.ndarray, ExtensionArray, "
            f"Index, Series, or NumpyExtensionArray got {type(arr).__name__}."
        )

    indices = ensure_platform_int(indices)

    if allow_fill:
        # Pandas style, -1 means NA
        validate_indices(indices, arr.shape[axis])
        # error: Argument 1 to "take_nd" has incompatible type
        # "ndarray[Any, Any] | ExtensionArray | Index | Series"; expected
        # "ndarray[Any, Any]"
        result = take_nd(
            arr,  # type: ignore[arg-type]
            indices,
            axis=axis,
            allow_fill=True,
            fill_value=fill_value,
        )
    else:
        # NumPy style
        # error: Unexpected keyword argument "axis" for "take" of "ExtensionArray"
        result = arr.take(indices, axis=axis)  # type: ignore[call-arg,assignment]
    return result
```



---

### Функция ID: 127

**Исходный код:**
```python
def get_loc(self, key) -> int | slice | np.ndarray:
        """
        Get integer location, slice or boolean mask for requested label.

        The `get_loc` method is used to retrieve the integer index, a slice for
        slicing objects, or a boolean mask indicating the presence of the label
        in the `IntervalIndex`.

        Parameters
        ----------
        key : label
            The value or range to find in the IntervalIndex.

        Returns
        -------
        int if unique index, slice if monotonic index, else mask
            The position or positions found. This could be a single
            number, a range, or an array of true/false values
            indicating the position(s) of the label.

        See Also
        --------
        IntervalIndex.get_indexer_non_unique : Compute indexer and
            mask for new index given the current index.
        Index.get_loc : Similar method in the base Index class.

        Examples
        --------
        >>> i1, i2 = pd.Interval(0, 1), pd.Interval(1, 2)
        >>> index = pd.IntervalIndex([i1, i2])
        >>> index.get_loc(1)
        0

        You can also supply a point inside an interval.

        >>> index.get_loc(1.5)
        1

        If a label is in several intervals, you get the locations of all the
        relevant intervals.

        >>> i3 = pd.Interval(0, 2)
        >>> overlapping_index = pd.IntervalIndex([i1, i2, i3])
        >>> overlapping_index.get_loc(0.5)
        array([ True, False,  True])

        Only exact matches will be returned if an interval is provided.

        >>> index.get_loc(pd.Interval(0, 1))
        0
        """
        self._check_indexing_error(key)

        if isinstance(key, Interval):
            if self.closed != key.closed:
                raise KeyError(key)
            mask = (self.left == key.left) & (self.right == key.right)
        elif is_valid_na_for_dtype(key, self.dtype):
            mask = self.isna()
        else:
            # assume scalar
            op_left = le if self.closed_left else lt
            op_right = le if self.closed_right else lt
            try:
                mask = op_left(self.left, key) & op_right(key, self.right)
            except TypeError as err:
                # scalar is not comparable to II subtype --> invalid label
                raise KeyError(key) from err

        matches = mask.sum()
        if matches == 0:
            raise KeyError(key)
        if matches == 1:
            return mask.argmax()

        res = lib.maybe_booleans_to_slice(mask.view("u1"))
        if isinstance(res, slice) and res.stop is None:
            # TODO: DO this in maybe_booleans_to_slice?
            res = slice(res.start, len(self), res.step)
        return res
```



---

### Функция ID: 128

**Исходный код:**
```java
public synchronized boolean hasAllFetchPositions() {
        // Since this is in the hot-path for fetching, we do this instead of using java.util.stream API
        Iterator<TopicPartitionState> it = assignment.stateIterator();
        while (it.hasNext()) {
            if (!it.next().hasValidPosition()) {
                return false;
            }
        }
        return true;
    }
```



---

### Функция ID: 129

**Исходный код:**
```python
def select(
        self,
        key: str,
        where=None,
        start=None,
        stop=None,
        columns=None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ):
        """
        Retrieve pandas object stored in file, optionally based on where criteria.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
            Object being retrieved from file.
        where : list or None
            List of Term (or convertible) objects, optional.
        start : int or None
            Row number to start selection.
        stop : int, default None
            Row number to stop selection.
        columns : list or None
            A list of columns that if not None, will limit the return columns.
        iterator : bool or False
            Returns an iterator.
        chunksize : int or None
            Number or rows to include in iteration, return an iterator.
        auto_close : bool or False
            Should automatically close the store when finished.

        Returns
        -------
        object
            Retrieved object from file.

        See Also
        --------
        HDFStore.select_as_coordinates : Returns the selection as an index.
        HDFStore.select_column : Returns a single column from the table.
        HDFStore.select_as_multiple : Retrieves pandas objects from multiple tables.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        ['/data1', '/data2']
        >>> store.select("/data1")  # doctest: +SKIP
           A  B
        0  1  2
        1  3  4
        >>> store.select("/data1", where="columns == A")  # doctest: +SKIP
           A
        0  1
        1  3
        >>> store.close()  # doctest: +SKIP
        """
        group = self.get_node(key)
        if group is None:
            raise KeyError(f"No object named {key} in the file")

        # create the storer and axes
        where = _ensure_term(where, scope_level=1)
        s = self._create_storer(group)
        s.infer_axes()

        # function to call on iteration
        def func(_start, _stop, _where):
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)

        # create the iterator
        it = TableIterator(
            self,
            s,
            func,
            where=where,
            nrows=s.nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )

        return it.get_result()
```



---

### Функция ID: 130

**Исходный код:**
```python
def fillna(
        self,
        value: object | ArrayLike,
        limit: int | None = None,
        copy: bool = True,
    ) -> Self:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like "value" can be given. It's expected
            that the array-like have the same length as 'self'.
        limit : int, default None
            The maximum number of entries where NA values will be filled.
        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.

        Returns
        -------
        ExtensionArray
            With NA/NaN filled.

        See Also
        --------
        api.extensions.ExtensionArray.dropna : Return ExtensionArray without
            NA values.
        api.extensions.ExtensionArray.isna : A 1-D array indicating if
            each value is missing.

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr.fillna(0)
        <IntegerArray>
        [0, 0, 2, 3, 0, 0]
        Length: 6, dtype: Int64
        """
        mask = self.isna()
        if limit is not None and limit < len(self):
            # isna can return an ExtensionArray, we're assuming that comparisons
            # are implemented.
            # mypy doesn't like that mask can be an EA which need not have `cumsum`
            modify = mask.cumsum() > limit  # type: ignore[union-attr]
            if modify.any():
                # Only copy mask if necessary
                mask = mask.copy()
                mask[modify] = False
        # error: Argument 2 to "check_value_size" has incompatible type
        # "ExtensionArray"; expected "ndarray"
        value = missing.check_value_size(
            value,
            mask,  # type: ignore[arg-type]
            len(self),
        )

        if mask.any():
            # fill with value
            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
            new_values[mask] = value
        else:
            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
        return new_values
```



---

### Функция ID: 131

**Исходный код:**
```java
public static Method getAccessibleMethod(final Class<?> cls, final Method method) {
        if (!MemberUtils.isPublic(method)) {
            return null;
        }
        // If the declaring class is public, we are done
        if (ClassUtils.isPublic(cls)) {
            return method;
        }
        final String methodName = method.getName();
        final Class<?>[] parameterTypes = method.getParameterTypes();
        // Check the implemented interfaces and subinterfaces
        final Method method2 = getAccessibleMethodFromInterfaceNest(cls, methodName, parameterTypes);
        // Check the superclass chain
        return method2 != null ? method2 : getAccessibleMethodFromSuperclass(cls, methodName, parameterTypes);
    }
```



---

### Функция ID: 132

**Исходный код:**
```java
public static double asin(double value) {
        boolean negateResult;
        if (value < 0.0) {
            value = -value;
            negateResult = true;
        } else {
            negateResult = false;
        }
        if (value <= ASIN_MAX_VALUE_FOR_TABS) {
            int index = (int) (value * ASIN_INDEXER + 0.5);
            double delta = value - index * ASIN_DELTA;
            double result = asinTab[index] + delta * (asinDer1DivF1Tab[index] + delta * (asinDer2DivF2Tab[index] + delta
                * (asinDer3DivF3Tab[index] + delta * asinDer4DivF4Tab[index])));
            return negateResult ? -result : result;
        } else if (value <= ASIN_MAX_VALUE_FOR_POWTABS) {
            int index = (int) (FastMath.powFast(value * ASIN_POWTABS_ONE_DIV_MAX_VALUE, ASIN_POWTABS_POWER) * ASIN_POWTABS_SIZE_MINUS_ONE
                + 0.5);
            double delta = value - asinParamPowTab[index];
            double result = asinPowTab[index] + delta * (asinDer1DivF1PowTab[index] + delta * (asinDer2DivF2PowTab[index] + delta
                * (asinDer3DivF3PowTab[index] + delta * asinDer4DivF4PowTab[index])));
            return negateResult ? -result : result;
        } else { // value > ASIN_MAX_VALUE_FOR_TABS, or value is NaN
            // This part is derived from fdlibm.
            if (value < 1.0) {
                double t = (1.0 - value) * 0.5;
                double p = t * (ASIN_PS0 + t * (ASIN_PS1 + t * (ASIN_PS2 + t * (ASIN_PS3 + t * (ASIN_PS4 + t * ASIN_PS5)))));
                double q = 1.0 + t * (ASIN_QS1 + t * (ASIN_QS2 + t * (ASIN_QS3 + t * ASIN_QS4)));
                double s = Math.sqrt(t);
                double z = s + s * (p / q);
                double result = ASIN_PIO2_HI - ((z + z) - ASIN_PIO2_LO);
                return negateResult ? -result : result;
            } else { // value >= 1.0, or value is NaN
                if (value == 1.0) {
                    return negateResult ? -M_HALF_PI : M_HALF_PI;
                } else {
                    return Double.NaN;
                }
            }
        }
    }
```



---

### Функция ID: 133

**Исходный код:**
```python
def dot(self, other: AnyArrayLike | DataFrame) -> Series | np.ndarray:
        """
        Compute the dot product between the Series and the columns of other.

        This method computes the dot product between the Series and another
        one, or the Series and each columns of a DataFrame, or the Series and
        each columns of an array.

        It can also be called using `self @ other`.

        Parameters
        ----------
        other : Series, DataFrame or array-like
            The other object to compute the dot product with its columns.

        Returns
        -------
        scalar, Series or numpy.ndarray
            Return the dot product of the Series and other if other is a
            Series, the Series of the dot product of Series and each rows of
            other if other is a DataFrame or a numpy.ndarray between the Series
            and each columns of the numpy array.

        See Also
        --------
        DataFrame.dot: Compute the matrix product with the DataFrame.
        Series.mul: Multiplication of series and other, element-wise.

        Notes
        -----
        The Series and other has to share the same index if other is a Series
        or a DataFrame.

        Examples
        --------
        >>> s = pd.Series([0, 1, 2, 3])
        >>> other = pd.Series([-1, 2, -3, 4])
        >>> s.dot(other)
        8
        >>> s @ other
        8
        >>> df = pd.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])
        >>> s.dot(df)
        0    24
        1    14
        dtype: int64
        >>> arr = np.array([[0, 1], [-2, 3], [4, -5], [6, 7]])
        >>> s.dot(arr)
        array([24, 14])
        """
        if isinstance(other, (Series, ABCDataFrame)):
            common = self.index.union(other.index)
            if len(common) > len(self.index) or len(common) > len(other.index):
                raise ValueError("matrices are not aligned")

            left = self.reindex(index=common)
            right = other.reindex(index=common)
            lvals = left.values
            rvals = right.values
        else:
            lvals = self.values
            rvals = np.asarray(other)
            if lvals.shape[0] != rvals.shape[0]:
                raise Exception(
                    f"Dot product shape mismatch, {lvals.shape} vs {rvals.shape}"
                )

        if isinstance(other, ABCDataFrame):
            common_type = find_common_type([self.dtypes] + list(other.dtypes))
            return self._constructor(
                np.dot(lvals, rvals), index=other.columns, copy=False, dtype=common_type
            ).__finalize__(self, method="dot")
        elif isinstance(other, Series):
            return np.dot(lvals, rvals)
        elif isinstance(rvals, np.ndarray):
            return np.dot(lvals, rvals)
        else:  # pragma: no cover
            raise TypeError(f"unsupported type: {type(other)}")
```



---

### Функция ID: 134

**Исходный код:**
```java
@Override
    public long write(ByteBuffer[] srcs, int offset, int length) throws IOException {
        if ((offset < 0) || (length < 0) || (offset > srcs.length - length))
            throw new IndexOutOfBoundsException();
        int totalWritten = 0;
        int i = offset;
        while (i < offset + length) {
            if (srcs[i].hasRemaining() || hasPendingWrites()) {
                int written = write(srcs[i]);
                if (written > 0) {
                    totalWritten += written;
                }
            }
            if (!srcs[i].hasRemaining() && !hasPendingWrites()) {
                i++;
            } else {
                // if we are unable to write the current buffer to socketChannel we should break,
                // as we might have reached max socket send buffer size.
                break;
            }
        }
        return totalWritten;
    }
```



---

### Функция ID: 135

**Исходный код:**
```java
static long load64Safely(byte[] input, int offset, int length) {
    long result = 0;
    // Due to the way we shift, we can stop iterating once we've run out of data, the rest
    // of the result already being filled with zeros.

    // This loop is critical to performance, so please check HashBenchmark if altering it.
    int limit = min(length, 8);
    for (int i = 0; i < limit; i++) {
      // Shift value left while iterating logically through the array.
      result |= (input[offset + i] & 0xFFL) << (i * 8);
    }
    return result;
  }
```



---

### Функция ID: 136

**Исходный код:**
```python
def read_all_dags(cls, session: Session = NEW_SESSION) -> dict[str, SerializedDAG]:
        """
        Read all DAGs in serialized_dag table.

        :param session: ORM Session
        :returns: a dict of DAGs read from database
        """
        latest_serialized_dag_subquery = (
            select(cls.dag_id, func.max(cls.created_at).label("max_created")).group_by(cls.dag_id).subquery()
        )
        serialized_dags = session.scalars(
            select(cls).join(
                latest_serialized_dag_subquery,
                (cls.dag_id == latest_serialized_dag_subquery.c.dag_id)
                and (cls.created_at == latest_serialized_dag_subquery.c.max_created),
            )
        )

        dags = {}
        for row in serialized_dags:
            log.debug("Deserializing DAG: %s", row.dag_id)
            dag = row.dag

            # Coherence check
            if dag.dag_id == row.dag_id:
                dags[row.dag_id] = dag
            else:
                log.warning(
                    "dag_id Mismatch in DB: Row with dag_id '%s' has Serialised DAG with '%s' dag_id",
                    row.dag_id,
                    dag.dag_id,
                )
        return dags
```



---

### Функция ID: 137

**Исходный код:**
```python
def nanmean(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute the mean of the element along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanmean(s.values)
    np.float64(1.5)
    """
    if values.dtype == object and len(values) > 1_000 and mask is None:
        # GH#54754 if we are going to fail, try to fail-fast
        nanmean(values[:1000], axis=axis, skipna=skipna)

    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    dtype_count = np.dtype(np.float64)

    # not using needs_i8_conversion because that includes period
    if dtype.kind in "mM":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind in "iu":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind == "f":
        dtype_sum = dtype
        dtype_count = dtype

    count = _get_counts(values.shape, mask, axis, dtype=dtype_count)
    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _ensure_numeric(the_sum)

    if axis is not None and getattr(the_sum, "ndim", False):
        count = cast(np.ndarray, count)
        with np.errstate(all="ignore"):
            # suppress division by zero warnings
            the_mean = the_sum / count
        ct_mask = count == 0
        if ct_mask.any():
            the_mean[ct_mask] = np.nan
    else:
        the_mean = the_sum / count if count > 0 else np.nan

    return the_mean
```



---

### Функция ID: 138

**Исходный код:**
```java
@Nullable
    static Database getIpinfoDatabase(final String databaseType) {
        // for ipinfo the database selection is more along the lines of user-agent sniffing than
        // string-based dispatch. the specific database_type strings could change in the future,
        // hence the somewhat loose nature of this checking.

        final String cleanedType = ipinfoTypeCleanup(databaseType);

        // early detection on any of the 'extended' types
        if (databaseType.contains("extended")) {
            // which are not currently supported
            logger.trace("returning null for unsupported database_type [{}]", databaseType);
            return null;
        }

        // early detection on 'country_asn' so the 'country' and 'asn' checks don't get faked out
        if (cleanedType.contains("country_asn")) {
            // but it's not currently supported
            logger.trace("returning null for unsupported database_type [{}]", databaseType);
            return null;
        }

        if (cleanedType.contains("asn")) {
            return Database.AsnV2;
        } else if (cleanedType.contains("country")) {
            return Database.CountryV2;
        } else if (cleanedType.contains("location")) { // note: catches 'location' and 'geolocation' ;)
            return Database.CityV2;
        } else if (cleanedType.contains("privacy")) {
            return Database.PrivacyDetection;
        } else {
            // no match was found
            logger.trace("returning null for unsupported database_type [{}]", databaseType);
            return null;
        }
    }
```



---

### Функция ID: 139

**Исходный код:**
```java
@Override
    public boolean ready(Node node, long now) {
        if (node.isEmpty())
            throw new IllegalArgumentException("Cannot connect to empty node " + node);

        if (isReady(node, now))
            return true;

        if (connectionStates.canConnect(node.idString(), now))
            // if we are interested in sending to a node and we don't have a connection to it, initiate one
            initiateConnect(node, now);

        return false;
    }
```



---

### Функция ID: 140

**Исходный код:**
```python
def sort_values(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: NaPosition = "last",
        key: Callable | None = None,
    ) -> Self | tuple[Self, np.ndarray]:
        """
        Return a sorted copy of the index.

        Return a sorted copy of the index, and optionally return the indices
        that sorted the index itself.

        Parameters
        ----------
        return_indexer : bool, default False
            Should the indices that would sort the index be returned.
        ascending : bool, default True
            Should the index values be sorted in an ascending order.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.
        key : callable, optional
            If not None, apply the key function to the index values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect an
            ``Index`` and return an ``Index`` of the same shape.

        Returns
        -------
        sorted_index : pandas.Index
            Sorted copy of the index.
        indexer : numpy.ndarray, optional
            The indices that the index itself was sorted by.

        See Also
        --------
        Series.sort_values : Sort values of a Series.
        DataFrame.sort_values : Sort values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([10, 100, 1, 1000])
        >>> idx
        Index([10, 100, 1, 1000], dtype='int64')

        Sort values in ascending order (default behavior).

        >>> idx.sort_values()
        Index([1, 10, 100, 1000], dtype='int64')

        Sort values in descending order, and also get the indices `idx` was
        sorted by.

        >>> idx.sort_values(ascending=False, return_indexer=True)
        (Index([1000, 100, 10, 1], dtype='int64'), array([3, 1, 0, 2]))
        """
        if key is None and (
            (ascending and self.is_monotonic_increasing)
            or (not ascending and self.is_monotonic_decreasing)
        ):
            if return_indexer:
                indexer = np.arange(len(self), dtype=np.intp)
                return self.copy(), indexer
            else:
                return self.copy()

        # GH 35584. Sort missing values according to na_position kwarg
        # ignore na_position for MultiIndex
        if not isinstance(self, ABCMultiIndex):
            _as = nargsort(
                items=self, ascending=ascending, na_position=na_position, key=key
            )
        else:
            idx = cast(Index, ensure_key_mapped(self, key))
            _as = idx.argsort(na_position=na_position)
            if not ascending:
                _as = _as[::-1]

        sorted_index = self.take(_as)

        if return_indexer:
            return sorted_index, _as
        else:
            return sorted_index
```



---

### Функция ID: 141

**Исходный код:**
```python
def update(self, other: Series | Sequence | Mapping) -> None:
        """
        Modify Series in place using values from passed Series.

        Uses non-NA values from passed Series to make updates. Aligns
        on index.

        Parameters
        ----------
        other : Series, or object coercible into Series
            Other Series that provides values to update the current Series.

        See Also
        --------
        Series.combine : Perform element-wise operation on two Series
            using a given function.
        Series.transform: Modify a Series using a function.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, 5, 6]))
        >>> s
        0    4
        1    5
        2    6
        dtype: int64

        >>> s = pd.Series(["a", "b", "c"])
        >>> s.update(pd.Series(["d", "e"], index=[0, 2]))
        >>> s
        0    d
        1    b
        2    e
        dtype: object

        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, 5, 6, 7, 8]))
        >>> s
        0    4
        1    5
        2    6
        dtype: int64

        If ``other`` contains NaNs the corresponding values are not updated
        in the original Series.

        >>> s = pd.Series([1, 2, 3])
        >>> s.update(pd.Series([4, np.nan, 6]))
        >>> s
        0    4
        1    2
        2    6
        dtype: int64

        ``other`` can also be a non-Series object type
        that is coercible into a Series

        >>> s = pd.Series([1, 2, 3])
        >>> s.update([4, np.nan, 6])
        >>> s
        0    4
        1    2
        2    6
        dtype: int64

        >>> s = pd.Series([1, 2, 3])
        >>> s.update({1: 9})
        >>> s
        0    1
        1    9
        2    3
        dtype: int64
        """
        if not CHAINED_WARNING_DISABLED:
            if sys.getrefcount(
                self
            ) <= REF_COUNT_METHOD and not com.is_local_in_caller_frame(self):
                warnings.warn(
                    _chained_assignment_method_msg,
                    ChainedAssignmentError,
                    stacklevel=2,
                )

        if not isinstance(other, Series):
            other = Series(other)

        other = other.reindex_like(self)
        mask = notna(other)

        self._mgr = self._mgr.putmask(mask=mask, new=other)
```



---

### Функция ID: 142

**Исходный код:**
```python
def sanitize_conn_id(conn_id: str | None, max_length=CONN_ID_MAX_LEN) -> str | None:
    r"""
    Sanitizes the connection id and allows only specific characters to be within.

    Namely, it allows alphanumeric characters plus the symbols #,!,-,_,.,:,\,/ and () from 1 and up to
    250 consecutive matches. If desired, the max length can be adjusted by setting `max_length`.

    You can try to play with the regex here: https://regex101.com/r/69033B/1

    The character selection is such that it prevents the injection of javascript or
    executable bits to avoid any awkward behaviour in the front-end.

    :param conn_id: The connection id to sanitize.
    :param max_length: The max length of the connection ID, by default it is 250.
    :return: the sanitized string, `None` otherwise.
    """
    # check if `conn_id` or our match group is `None` and the `conn_id` is within the specified length.
    if (not isinstance(conn_id, str) or len(conn_id) > max_length) or (
        res := re.match(RE_SANITIZE_CONN_ID, conn_id)
    ) is None:
        return None

    # if we reach here, then we matched something, return the first match
    return res.group(0)
```



---

### Функция ID: 143

**Исходный код:**
```java
private boolean startsWithArgumentClassName(String message, @Nullable Object argument) {
			if (argument == null) {
				return false;
			}
			Class<?> argumentType = argument.getClass();
			// On Java 8, the message starts with the class name: "java.lang.String cannot
			// be cast..."
			if (message.startsWith(argumentType.getName())) {
				return true;
			}
			// On Java 11, the message starts with "class ..." a.k.a. Class.toString()
			if (message.startsWith(argumentType.toString())) {
				return true;
			}
			// On Java 9, the message used to contain the module name:
			// "java.base/java.lang.String cannot be cast..."
			int moduleSeparatorIndex = message.indexOf('/');
			if (moduleSeparatorIndex != -1 && message.startsWith(argumentType.getName(), moduleSeparatorIndex + 1)) {
				return true;
			}
			if (CLASS_GET_MODULE != null && MODULE_GET_NAME != null) {
				Object module = ReflectionUtils.invokeMethod(CLASS_GET_MODULE, argumentType);
				Object moduleName = ReflectionUtils.invokeMethod(MODULE_GET_NAME, module);
				return message.startsWith(moduleName + "/" + argumentType.getName());
			}
			return false;
		}
```



---

### Функция ID: 144

**Исходный код:**
```python
def pandas_dtype(dtype) -> DtypeObj:
    """
    Convert input into a pandas only dtype object or a numpy dtype object.

    Parameters
    ----------
    dtype : object
        The object to be converted into a dtype.

    Returns
    -------
    np.dtype or a pandas dtype
        The converted dtype, which can be either a numpy dtype or a pandas dtype.

    Raises
    ------
    TypeError if not a dtype

    See Also
    --------
    api.types.is_dtype : Return true if the condition is satisfied for the arr_or_dtype.

    Examples
    --------
    >>> pd.api.types.pandas_dtype(int)
    dtype('int64')
    """
    # short-circuit
    if isinstance(dtype, np.ndarray):
        return dtype.dtype
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        return dtype

    # builtin aliases
    if dtype is str and using_string_dtype():
        from pandas.core.arrays.string_ import StringDtype

        return StringDtype(na_value=np.nan)

    # registered extension types
    result = registry.find(dtype)
    if result is not None:
        if isinstance(result, type):
            # GH 31356, GH 54592
            warnings.warn(
                f"Instantiating {result.__name__} without any arguments."
                f"Pass a {result.__name__} instance to silence this warning.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            result = result()
        return result

    # try a numpy dtype
    # raise a consistent TypeError if failed
    try:
        with warnings.catch_warnings():
            # TODO: warnings.catch_warnings can be removed when numpy>2.3.0
            # is the minimum version
            # GH#51523 - Series.astype(np.integer) doesn't show
            # numpy deprecation warning of np.integer
            # Hence enabling DeprecationWarning
            warnings.simplefilter("always", DeprecationWarning)
            npdtype = np.dtype(dtype)
    except SyntaxError as err:
        # np.dtype uses `eval` which can raise SyntaxError
        raise TypeError(f"data type '{dtype}' not understood") from err

    # Any invalid dtype (such as pd.Timestamp) should raise an error.
    # np.dtype(invalid_type).kind = 0 for such objects. However, this will
    # also catch some valid dtypes such as object, np.object_ and 'object'
    # which we safeguard against by catching them earlier and returning
    # np.dtype(valid_dtype) before this condition is evaluated.
    if is_hashable(dtype) and dtype in [
        object,
        np.object_,
        "object",
        "O",
        "object_",
    ]:
        # check hashability to avoid errors/DeprecationWarning when we get
        # here and `dtype` is an array
        return npdtype
    elif npdtype.kind == "O":
        raise TypeError(f"dtype '{dtype}' not understood")

    return npdtype
```



---

### Функция ID: 145

**Исходный код:**
```python
def to_typst(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        encoding: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
    ) -> str | None:
        """
        Write Styler to a file, buffer or string in Typst format.

        .. versionadded:: 3.0.0

        Parameters
        ----------
        %(buf)s
        %(encoding)s
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns`` value.
        max_rows : int, optional
            The maximum number of rows that will be rendered. Defaults to
            ``pandas.options.styler.render.max_rows``, which is None.
        max_columns : int, optional
            The maximum number of columns that will be rendered. Defaults to
            ``pandas.options.styler.render.max_columns``, which is None.

            Rows and columns may be reduced if the number of total elements is
            large. This value is set to ``pandas.options.styler.render.max_elements``,
            which is 262144 (18 bit browser rendering).

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        DataFrame.to_typst : Write a DataFrame to a file,
            buffer or string in Typst format.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.style.to_typst()  # doctest: +SKIP

        .. code-block:: typst

            #table(
              columns: 3,
              [], [A], [B],

              [0], [1], [3],
              [1], [2], [4],
            )
        """
        obj = self._copy(deepcopy=True)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        text = obj._render_typst(
            sparse_columns=sparse_columns,
            sparse_index=sparse_index,
            max_rows=max_rows,
            max_cols=max_columns,
        )
        return save_to_buffer(
            text, buf=buf, encoding=(encoding if buf is not None else None)
        )
```



---

### Функция ID: 146

**Исходный код:**
```python
def get_loc_level(self, key, level: IndexLabel = 0, drop_level: bool = True):
        """
        Get location and sliced index for requested label(s)/level(s).

        The `get_loc_level` method is a more advanced form of `get_loc`, allowing
        users to specify not just a label or sequence of labels, but also the level(s)
        in which to search. This method is useful when you need to isolate particular
        sections of a MultiIndex, either for further analysis or for slicing and
        dicing the data. The method provides flexibility in terms of maintaining
        or dropping levels from the resulting index based on the `drop_level`
        parameter.

        Parameters
        ----------
        key : label or sequence of labels
            The label(s) for which to get the location.
        level : int/level name or list thereof, optional
            The level(s) in the MultiIndex to consider. If not provided, defaults
            to the first level.
        drop_level : bool, default True
            If ``False``, the resulting index will not drop any level.

        Returns
        -------
        tuple
            A 2-tuple where the elements :

            Element 0: int, slice object or boolean array.

            Element 1: The resulting sliced multiindex/index. If the key
            contains all levels, this will be ``None``.

        See Also
        --------
        MultiIndex.get_loc  : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list("abb"), list("def")], names=["A", "B"])

        >>> mi.get_loc_level("b")
        (slice(1, 3, None), Index(['e', 'f'], dtype='object', name='B'))

        >>> mi.get_loc_level("e", level="B")
        (array([False,  True, False]), Index(['b'], dtype='object', name='A'))

        >>> mi.get_loc_level(["b", "e"])
        (1, None)
        """
        if not isinstance(level, (range, list, tuple)):
            level = self._get_level_number(level)
        else:
            level = [self._get_level_number(lev) for lev in level]

        loc, mi = self._get_loc_level(key, level=level)
        if not drop_level:
            if lib.is_integer(loc):
                # Slice index must be an integer or None
                mi = self[loc : loc + 1]
            else:
                mi = self[loc]
        return loc, mi
```



---

### Функция ID: 147

**Исходный код:**
```python
def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert the DataFrame to a NumPy array.

        By default, the dtype of the returned array will be the common NumPy
        dtype of all types in the DataFrame. For example, if the dtypes are
        ``float16`` and ``float32``, the results dtype will be ``float32``.
        This may require copying data and coercing values, which may be
        expensive.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the dtypes of the DataFrame columns.

        Returns
        -------
        numpy.ndarray
            The NumPy array representing the values in the DataFrame.

        See Also
        --------
        Series.to_numpy : Similar method for Series.

        Examples
        --------
        >>> pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_numpy()
        array([[1, 3],
               [2, 4]])

        With heterogeneous data, the lowest common type will have to
        be used.

        >>> df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.5]})
        >>> df.to_numpy()
        array([[1. , 3. ],
               [2. , 4.5]])

        For a mix of numeric and non-numeric types, the output array will
        have object dtype.

        >>> df["C"] = pd.date_range("2000", periods=2)
        >>> df.to_numpy()
        array([[1, 3.0, Timestamp('2000-01-01 00:00:00')],
               [2, 4.5, Timestamp('2000-01-02 00:00:00')]], dtype=object)
        """
        if dtype is not None:
            dtype = np.dtype(dtype)
        result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
        if result.dtype is not dtype:
            result = np.asarray(result, dtype=dtype)

        return result
```



---

### Функция ID: 148

**Исходный код:**
```python
def groups(self) -> dict[Hashable, Index]:
        """
        Dict {group name -> group labels}.

        This property provides a dictionary representation of the groupings formed
        during a groupby operation, where each key represents a unique group value from
        the specified column(s), and each value is a list of index labels
        that belong to that group.

        See Also
        --------
        core.groupby.DataFrameGroupBy.get_group : Retrieve group from a
            ``DataFrameGroupBy`` object with provided name.
        core.groupby.SeriesGroupBy.get_group : Retrieve group from a
            ``SeriesGroupBy`` object with provided name.
        core.resample.Resampler.get_group : Retrieve group from a
            ``Resampler`` object with provided name.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).groups
        {'a': ['a', 'a'], 'b': ['b']}

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"])
        >>> df
           a  b  c
        0  1  2  3
        1  1  5  6
        2  7  8  9
        >>> df.groupby(by="a").groups
        {1: [0, 1], 7: [2]}

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").groups
        {Timestamp('2023-01-01 00:00:00'): np.int64(2),
         Timestamp('2023-02-01 00:00:00'): np.int64(4)}
        """
        if isinstance(self.keys, list) and len(self.keys) == 1:
            warnings.warn(
                "`groups` by one element list returns scalar is deprecated "
                "and will be removed. In a future version `groups` by one element "
                "list will return tuple. Use ``df.groupby(by='a').groups`` "
                "instead of ``df.groupby(by=['a']).groups`` to avoid this warning",
                Pandas4Warning,
                stacklevel=find_stack_level(),
            )
        return self._grouper.groups
```



---

### Функция ID: 149

**Исходный код:**
```python
def mean(
        self,
        numeric_only: bool = False,
        skipna: bool = True,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None`` and defaults to ``False``.

        skipna : bool, default True
            Exclude NA/null values. If an entire group is NA, the result will be NA.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Mean of values within each group. Same object type as the caller.
        %(see_also)s
        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"A": [1, 1, 2, 1, 2], "B": [np.nan, 2, 3, 4, 5], "C": [1, 2, 1, 1, 2]},
        ...     columns=["A", "B", "C"],
        ... )

        Groupby one column and return the mean of the remaining columns in
        each group.

        >>> df.groupby("A").mean()
             B         C
        A
        1  3.0  1.333333
        2  4.0  1.500000

        Groupby two columns and return the mean of the remaining column.

        >>> df.groupby(["A", "B"]).mean()
                 C
        A B
        1 2.0  2.0
          4.0  1.0
        2 3.0  1.0
          5.0  2.0

        Groupby one column and return the mean of only particular column in
        the group.

        >>> df.groupby("A")["B"].mean()
        A
        1    3.0
        2    4.0
        Name: B, dtype: float64
        """

        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_mean

            return self._numba_agg_general(
                grouped_mean,
                executor.float_dtype_mapping,
                engine_kwargs,
                min_periods=0,
                skipna=skipna,
            )
        else:
            result = self._cython_agg_general(
                "mean",
                alt=lambda x: Series(x, copy=False).mean(
                    numeric_only=numeric_only, skipna=skipna
                ),
                numeric_only=numeric_only,
                skipna=skipna,
            )
            return result.__finalize__(self.obj, method="groupby")
```



---

### Функция ID: 150

**Исходный код:**
```python
def get_commented_out_prs_from_provider_changelogs() -> list[int]:
    """
    Returns list of PRs that are commented out in the changelog.
    :return: list of PR numbers that appear only in comments in changelog.rst files in "providers" dir
    """
    pr_matcher = re.compile(r".*\(#([0-9]+)\).*")
    commented_prs = set()

    # Get all provider distributions
    provider_distributions_metadata = get_provider_distributions_metadata()

    for provider_id in provider_distributions_metadata.keys():
        provider_details = get_provider_details(provider_id)
        changelog_path = provider_details.changelog_path
        print(f"[info]Checking changelog {changelog_path} for PRs to be excluded automatically.")
        if not changelog_path.exists():
            continue

        changelog_lines = changelog_path.read_text().splitlines()
        in_excluded_section = False

        for line in changelog_lines:
            # Check if we're entering an excluded/commented section
            if line.strip().startswith(
                ".. Below changes are excluded from the changelog"
            ) or line.strip().startswith(".. Review and move the new changes"):
                in_excluded_section = True
                continue
            # Check if we're exiting the excluded section (new version header or regular content)
            # Version headers are lines that contain only dots (like "4.10.1" followed by "......")
            # Or lines that start with actual content sections like "Misc", "Features", etc.
            if (
                in_excluded_section
                and line
                and not line.strip().startswith("..")
                and not line.strip().startswith("*")
            ):
                # end excluded section with empty line
                if line.strip() == "":
                    in_excluded_section = False

            # Extract PRs from excluded sections
            if in_excluded_section and line.strip().startswith("*"):
                match_result = pr_matcher.search(line)
                if match_result:
                    commented_prs.add(int(match_result.group(1)))

    return sorted(commented_prs)
```



---

### Функция ID: 151

**Исходный код:**
```python
def one_hot(
    x: Array,
    /,
    num_classes: int,
    *,
    dtype: DType | None = None,
    axis: int = -1,
    xp: ModuleType | None = None,
) -> Array:
    """
    One-hot encode the given indices.

    Each index in the input `x` is encoded as a vector of zeros of length `num_classes`
    with the element at the given index set to one.

    Parameters
    ----------
    x : array
        An array with integral dtype whose values are between `0` and `num_classes - 1`.
    num_classes : int
        Number of classes in the one-hot dimension.
    dtype : DType, optional
        The dtype of the return value.  Defaults to the default float dtype (usually
        float64).
    axis : int, optional
        Position in the expanded axes where the new axis is placed. Default: -1.
    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer.

    Returns
    -------
    array
        An array having the same shape as `x` except for a new axis at the position
        given by `axis` having size `num_classes`.  If `axis` is unspecified, it
        defaults to -1, which appends a new axis.

        If ``x < 0`` or ``x >= num_classes``, then the result is undefined, may raise
        an exception, or may even cause a bad state.  `x` is not checked.

    Examples
    --------
    >>> import array_api_extra as xpx
    >>> import array_api_strict as xp
    >>> xpx.one_hot(xp.asarray([1, 2, 0]), 3)
    Array([[0., 1., 0.],
          [0., 0., 1.],
          [1., 0., 0.]], dtype=array_api_strict.float64)
    """
    # Validate inputs.
    if xp is None:
        xp = array_namespace(x)
    if not xp.isdtype(x.dtype, "integral"):
        msg = "x must have an integral dtype."
        raise TypeError(msg)
    if dtype is None:
        dtype = _funcs.default_dtype(xp, device=get_device(x))
    # Delegate where possible.
    if is_jax_namespace(xp):
        from jax.nn import one_hot as jax_one_hot

        return jax_one_hot(x, num_classes, dtype=dtype, axis=axis)
    if is_torch_namespace(xp):
        from torch.nn.functional import one_hot as torch_one_hot

        x = xp.astype(x, xp.int64)  # PyTorch only supports int64 here.
        try:
            out = torch_one_hot(x, num_classes)
        except RuntimeError as e:
            raise IndexError from e
    else:
        out = _funcs.one_hot(x, num_classes, xp=xp)
    out = xp.astype(out, dtype, copy=False)
    if axis != -1:
        out = xp.moveaxis(out, -1, axis)
    return out
```



---

### Функция ID: 152

**Исходный код:**
```python
def var(
        self,
        ddof: int = 1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        numeric_only: bool = False,
        skipna: bool = True,
    ):
        """
        Compute variance of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        skipna : bool, default True
            Exclude NA/null values. If an entire group is NA, the result will be NA.

            .. versionadded:: 3.0.0

        Returns
        -------
        Series or DataFrame
            Variance of values within each group.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ["a", "a", "a", "b", "b", "b"]
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).var()
        a    10.333333
        b     0.333333
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {"a": [1, 3, 5, 7, 7, 8, 3], "b": [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(
        ...     data, index=["dog", "dog", "dog", "mouse", "mouse", "mouse", "mouse"]
        ... )
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).var()
                      a          b
        dog    4.000000  12.333333
        mouse  4.916667   2.250000
        """
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_var

            return self._numba_agg_general(
                grouped_var,
                executor.float_dtype_mapping,
                engine_kwargs,
                min_periods=0,
                ddof=ddof,
                skipna=skipna,
            )
        else:
            return self._cython_agg_general(
                "var",
                alt=lambda x: Series(x, copy=False).var(ddof=ddof, skipna=skipna),
                numeric_only=numeric_only,
                ddof=ddof,
                skipna=skipna,
            )
```



---

### Функция ID: 153

**Исходный код:**
```java
public static String join(final Iterator<?> iterator, final char separator) {
        // handle null, zero and one elements before building a buffer
        if (iterator == null) {
            return null;
        }
        if (!iterator.hasNext()) {
            return EMPTY;
        }
        return Streams.of(iterator).collect(LangCollectors.joining(ObjectUtils.toString(String.valueOf(separator)), EMPTY, EMPTY, ObjectUtils::toString));
    }
```



---

### Функция ID: 154

**Исходный код:**
```python
def _set_axis_name(
        self, name, axis: Axis = 0, *, inplace: bool = False
    ) -> Self | None:
        """
        Set the name(s) of the axis.

        Parameters
        ----------
        name : str or list of str
            Name(s) to set.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to set the label. The value 0 or 'index' specifies index,
            and the value 1 or 'columns' specifies columns.
        inplace : bool, default False
            If `True`, do operation inplace and return None.

        Returns
        -------
        Series, DataFrame, or None
            The same type as the caller or `None` if `inplace` is `True`.

        See Also
        --------
        DataFrame.rename : Alter the axis labels of :class:`DataFrame`.
        Series.rename : Alter the index labels or set the index name
            of :class:`Series`.
        Index.rename : Set the name of :class:`Index` or :class:`MultiIndex`.

        Examples
        --------
        >>> df = pd.DataFrame({"num_legs": [4, 4, 2]}, ["dog", "cat", "monkey"])
        >>> df
                num_legs
        dog            4
        cat            4
        monkey         2
        >>> df._set_axis_name("animal")
                num_legs
        animal
        dog            4
        cat            4
        monkey         2
        >>> df.index = pd.MultiIndex.from_product(
        ...     [["mammal"], ["dog", "cat", "monkey"]]
        ... )
        >>> df._set_axis_name(["type", "name"])
                       num_legs
        type   name
        mammal dog        4
               cat        4
               monkey     2
        """
        axis = self._get_axis_number(axis)
        idx = self._get_axis(axis).set_names(name)

        inplace = validate_bool_kwarg(inplace, "inplace")
        renamed = self if inplace else self.copy(deep=False)
        if axis == 0:
            renamed.index = idx
        else:
            renamed.columns = idx

        if not inplace:
            return renamed
        return None
```



---

### Функция ID: 155

**Исходный код:**
```python
def aggregate(self, func=None, axis: Axis = 0, *args, **kwargs):
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a Series or when passed to Series.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
            - dict of axis labels -> functions, function names or list of such.
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        scalar, Series or DataFrame
            The return can be:

            * scalar : when Series.agg is called with single function
            * Series : when DataFrame.agg is called with a single function
            * DataFrame : when DataFrame.agg is called with several functions

        See Also
        --------
        Series.apply : Invoke function on a Series.
        Series.transform : Transform function producing a Series with like indexes.

        Notes
        -----
        The aggregation operations are always performed over an axis, either the
        index (default) or the column axis. This behavior is different from
        `numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,
        `var`), where the default is to compute the aggregation of the flattened
        array, e.g., ``numpy.mean(arr_2d)`` as opposed to
        ``numpy.mean(arr_2d, axis=0)``.

        `agg` is an alias for `aggregate`. Use the alias.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        A passed user-defined-function will be passed a Series for evaluation.

        If ``func`` defines an index relabeling, ``axis`` must be ``0`` or ``index``.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.agg("min")
        1

        >>> s.agg(["min", "max"])
        min   1
        max   4
        dtype: int64
        """

        # Validate the axis parameter
        self._get_axis_number(axis)

        # if func is None, will switch to user-provided "named aggregation" kwargs
        if func is None:
            func = dict(kwargs.items())

        op = SeriesApply(self, func, args=args, kwargs=kwargs)
        result = op.agg()
        return result
```



---

### Функция ID: 156

**Исходный код:**
```python
def dot(self, other: AnyArrayLike | DataFrame) -> DataFrame | Series:
        """
        Compute the matrix multiplication between the DataFrame and other.

        This method computes the matrix product between the DataFrame and the
        values of an other Series, DataFrame or a numpy array.

        It can also be called using ``self @ other``.

        Parameters
        ----------
        other : Series, DataFrame or array-like
            The other object to compute the matrix product with.

        Returns
        -------
        Series or DataFrame
            If other is a Series, return the matrix product between self and
            other as a Series. If other is a DataFrame or a numpy.array, return
            the matrix product of self and other in a DataFrame of a np.array.

        See Also
        --------
        Series.dot: Similar method for Series.

        Notes
        -----
        The dimensions of DataFrame and other must be compatible in order to
        compute the matrix multiplication. In addition, the column names of
        DataFrame and the index of other must contain the same values, as they
        will be aligned prior to the multiplication.

        The dot method for Series computes the inner product, instead of the
        matrix product here.

        Examples
        --------
        Here we multiply a DataFrame with a Series.

        >>> df = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
        >>> s = pd.Series([1, 1, 2, 1])
        >>> df.dot(s)
        0    -4
        1     5
        dtype: int64

        Here we multiply a DataFrame with another DataFrame.

        >>> other = pd.DataFrame([[0, 1], [1, 2], [-1, -1], [2, 0]])
        >>> df.dot(other)
            0   1
        0   1   4
        1   2   2

        Note that the dot method give the same result as @

        >>> df @ other
            0   1
        0   1   4
        1   2   2

        The dot method works also if other is an np.array.

        >>> arr = np.array([[0, 1], [1, 2], [-1, -1], [2, 0]])
        >>> df.dot(arr)
            0   1
        0   1   4
        1   2   2

        Note how shuffling of the objects does not change the result.

        >>> s2 = s.reindex([1, 0, 2, 3])
        >>> df.dot(s2)
        0    -4
        1     5
        dtype: int64
        """
        if isinstance(other, (Series, DataFrame)):
            common = self.columns.union(other.index)
            if len(common) > len(self.columns) or len(common) > len(other.index):
                raise ValueError("matrices are not aligned")

            left = self.reindex(columns=common)
            right = other.reindex(index=common)
            lvals = left.values
            rvals = right._values
        else:
            left = self
            lvals = self.values
            rvals = np.asarray(other)
            if lvals.shape[1] != rvals.shape[0]:
                raise ValueError(
                    f"Dot product shape mismatch, {lvals.shape} vs {rvals.shape}"
                )

        if isinstance(other, DataFrame):
            common_type = find_common_type(list(self.dtypes) + list(other.dtypes))
            return self._constructor(
                np.dot(lvals, rvals),
                index=left.index,
                columns=other.columns,
                copy=False,
                dtype=common_type,
            )
        elif isinstance(other, Series):
            common_type = find_common_type(list(self.dtypes) + [other.dtypes])
            return self._constructor_sliced(
                np.dot(lvals, rvals), index=left.index, copy=False, dtype=common_type
            )
        elif isinstance(rvals, (np.ndarray, Index)):
            result = np.dot(lvals, rvals)
            if result.ndim == 2:
                return self._constructor(result, index=left.index, copy=False)
            else:
                return self._constructor_sliced(result, index=left.index, copy=False)
        else:  # pragma: no cover
            raise TypeError(f"unsupported type: {type(other)}")
```



---

### Функция ID: 157

**Исходный код:**
```python
def _flush_logs_out_of_heap(
    heap: list[tuple[int, StructuredLogMessage]],
    flush_size: int,
    last_log_container: list[StructuredLogMessage | None],
) -> Generator[StructuredLogMessage, None, None]:
    """
    Flush logs out of the heap, deduplicating them based on the last log.

    :param heap: heap to flush logs from
    :param flush_size: number of logs to flush
    :param last_log_container: a container to store the last log, to avoid duplicate logs
    :return: a generator that yields deduplicated logs
    """
    last_log = last_log_container[0]
    for _ in range(flush_size):
        sort_key, line = heapq.heappop(heap)
        if line != last_log or _is_sort_key_with_default_timestamp(sort_key):  # dedupe
            yield line
        last_log = line
    # update the last log container with the last log
    last_log_container[0] = last_log
```



---

### Функция ID: 158

**Исходный код:**
```java
Object parseValue(ConfigKey key, Object value, boolean isSet) {
        Object parsedValue;
        if (isSet) {
            parsedValue = parseType(key.name, value, key.type);
        // props map doesn't contain setting, the key is required because no default value specified - its an error
        } else if (NO_DEFAULT_VALUE.equals(key.defaultValue)) {
            throw new ConfigException("Missing required configuration \"" + key.name + "\" which has no default value.");
        } else {
            // otherwise assign setting its default value
            parsedValue = key.defaultValue;
        }
        if (key.validator instanceof ValidList && parsedValue instanceof List) {
            List<?> originalListValue = (List<?>) parsedValue;
            parsedValue = originalListValue.stream().distinct().collect(Collectors.toList());
            if (originalListValue.size() != ((List<?>) parsedValue).size()) {
                LOGGER.warn("Configuration key \"{}\" contains duplicate values. Duplicates will be removed. The original value " +
                        "is: {}, the updated value is: {}", key.name, originalListValue, parsedValue);
            }
        }
        if (key.validator != null) {
            key.validator.ensureValid(key.name, parsedValue);
        }
        return parsedValue;
    }
```



---

### Функция ID: 159

**Исходный код:**
```python
def nanvar(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask=None,
):
    """
    Compute the variance along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanvar(s.values)
    1.0
    """
    dtype = values.dtype
    mask = _maybe_get_mask(values, skipna, mask)
    if dtype.kind in "iu":
        values = values.astype("f8")
        if mask is not None:
            values[mask] = np.nan

    if values.dtype.kind == "f":
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    else:
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)

    # xref GH10242
    # Compute variance via two-pass algorithm, which is stable against
    # cancellation errors and relatively accurate for small numbers of
    # observations.
    #
    # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    avg = _ensure_numeric(values.sum(axis=axis, dtype=np.float64)) / count
    if axis is not None:
        avg = np.expand_dims(avg, axis)
    if values.dtype.kind == "c":
        # Need to use absolute value for complex numbers.
        sqr = _ensure_numeric(abs(avg - values) ** 2)
    else:
        sqr = _ensure_numeric((avg - values) ** 2)
    if mask is not None:
        np.putmask(sqr, mask, 0)
    result = sqr.sum(axis=axis, dtype=np.float64) / d

    # Return variance as np.float64 (the datatype used in the accumulator),
    # unless we were dealing with a float array, in which case use the same
    # precision as the original values array.
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)
    return result
```



---

### Функция ID: 160

**Исходный код:**
```python
def to_frame(
        self,
        index: bool = True,
        name=lib.no_default,
        allow_duplicates: bool = False,
    ) -> DataFrame:
        """
        Create a DataFrame with the levels of the MultiIndex as columns.

        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original MultiIndex.

        name : list / sequence of str, optional
            The passed names should substitute index level names.

        allow_duplicates : bool, optional default False
            Allow duplicate column labels to be created.

        Returns
        -------
        DataFrame
            DataFrame representation of the MultiIndex, with levels as columns.

        See Also
        --------
        DataFrame : Two-dimensional, size-mutable, potentially heterogeneous
            tabular data.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a", "b"], ["c", "d"]])
        >>> mi
        MultiIndex([('a', 'c'),
                    ('b', 'd')],
                   )

        >>> df = mi.to_frame()
        >>> df
             0  1
        a c  a  c
        b d  b  d

        >>> df = mi.to_frame(index=False)
        >>> df
           0  1
        0  a  c
        1  b  d

        >>> df = mi.to_frame(name=["x", "y"])
        >>> df
             x  y
        a c  a  c
        b d  b  d
        """
        from pandas import DataFrame

        if name is not lib.no_default:
            if not is_list_like(name):
                raise TypeError("'name' must be a list / sequence of column names.")

            if len(name) != len(self.levels):
                raise ValueError(
                    "'name' should have same length as number of levels on index."
                )
            idx_names = name
        else:
            idx_names = self._get_level_names()

        if not allow_duplicates and len(set(idx_names)) != len(idx_names):
            raise ValueError(
                "Cannot create duplicate column labels if allow_duplicates is False"
            )

        # Guarantee resulting column order - PY36+ dict maintains insertion order
        result = DataFrame(
            {level: self._get_level_values(level) for level in range(len(self.levels))},
            copy=False,
        )
        result.columns = idx_names

        if index:
            result.index = self
        return result
```



---

### Функция ID: 161

**Исходный код:**
```python
def infer_freq(
    index: DatetimeIndex | TimedeltaIndex | Series | DatetimeLikeArrayMixin,
) -> str | None:
    """
    Infer the most likely frequency given the input index.

    This method attempts to deduce the most probable frequency (e.g., 'D' for daily,
    'H' for hourly) from a sequence of datetime-like objects. It is particularly useful
    when the frequency of a time series is not explicitly set or known but can be
    inferred from its values.

    Parameters
    ----------
    index : DatetimeIndex, TimedeltaIndex, Series or array-like
      If passed a Series will use the values of the series (NOT THE INDEX).

    Returns
    -------
    str or None
        None if no discernible frequency.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If there are fewer than three values.

    See Also
    --------
    date_range : Return a fixed frequency DatetimeIndex.
    timedelta_range : Return a fixed frequency TimedeltaIndex with day as the default.
    period_range : Return a fixed frequency PeriodIndex.
    DatetimeIndex.freq : Return the frequency object if it is set, otherwise None.

    Examples
    --------
    >>> idx = pd.date_range(start="2020/12/01", end="2020/12/30", periods=30)
    >>> pd.infer_freq(idx)
    'D'
    """
    from pandas.core.api import DatetimeIndex

    if isinstance(index, ABCSeries):
        values = index._values

        if isinstance(index.dtype, ArrowDtype):
            import pyarrow as pa

            if pa.types.is_timestamp(values.dtype.pyarrow_dtype):
                # GH#58403
                values = values._to_datetimearray()

        if not (
            lib.is_np_dtype(values.dtype, "mM")
            or isinstance(values.dtype, DatetimeTZDtype)
            or values.dtype == object
        ):
            raise TypeError(
                "cannot infer freq from a non-convertible dtype "
                f"on a Series of {index.dtype}"
            )
        index = values

    inferer: _FrequencyInferer

    if not hasattr(index, "dtype"):
        pass
    elif isinstance(index.dtype, PeriodDtype):
        raise TypeError(
            "PeriodIndex given. Check the `freq` attribute instead of using infer_freq."
        )
    elif lib.is_np_dtype(index.dtype, "m"):
        # Allow TimedeltaIndex and TimedeltaArray
        inferer = _TimedeltaFrequencyInferer(index)
        return inferer.get_freq()

    elif is_numeric_dtype(index.dtype):
        raise TypeError(
            f"cannot infer freq from a non-convertible index of dtype {index.dtype}"
        )

    if not isinstance(index, DatetimeIndex):
        index = DatetimeIndex(index)

    inferer = _FrequencyInferer(index)
    return inferer.get_freq()
```



---

### Функция ID: 162

**Исходный код:**
```python
def idxmin(
        self, axis: Axis = 0, skipna: bool = True, numeric_only: bool = False
    ) -> Series:
        """
        Return index of first occurrence of minimum over requested axis.

        NA/null values are excluded.

        Parameters
        ----------
        axis : {{0 or 'index', 1 or 'columns'}}, default 0
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
        skipna : bool, default True
            Exclude NA/null values. If the entire DataFrame is NA,
            or if ``skipna=False`` and there is an NA value, this method
            will raise a ``ValueError``.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

        Returns
        -------
        Series
            Indexes of minima along the specified axis.

        Raises
        ------
        ValueError
            * If the row/column is empty

        See Also
        --------
        Series.idxmin : Return index of the minimum element.

        Notes
        -----
        This method is the DataFrame version of ``ndarray.argmin``.

        Examples
        --------
        Consider a dataset containing food consumption in Argentina.

        >>> df = pd.DataFrame(
        ...     {
        ...         {
        ...             "consumption": [10.51, 103.11, 55.48],
        ...             "co2_emissions": [37.2, 19.66, 1712],
        ...         }
        ...     },
        ...     index=["Pork", "Wheat Products", "Beef"],
        ... )

        >>> df
                        consumption  co2_emissions
        Pork                  10.51         37.20
        Wheat Products       103.11         19.66
        Beef                  55.48       1712.00

        By default, it returns the index for the minimum value in each column.

        >>> df.idxmin()
        consumption                Pork
        co2_emissions    Wheat Products
        dtype: object

        To return the index for the minimum value in each row, use ``axis="columns"``.

        >>> df.idxmin(axis="columns")
        Pork                consumption
        Wheat Products    co2_emissions
        Beef                consumption
        dtype: object
        """
        axis = self._get_axis_number(axis)

        if self.empty and len(self.axes[axis]):
            axis_dtype = self.axes[axis].dtype
            return self._constructor_sliced(dtype=axis_dtype)

        if numeric_only:
            data = self._get_numeric_data()
        else:
            data = self

        res = data._reduce(
            nanops.nanargmin, "argmin", axis=axis, skipna=skipna, numeric_only=False
        )
        indices = res._values
        # indices will always be np.ndarray since axis is not N

        if (indices == -1).any():
            if skipna:
                msg = "Encountered all NA values"
            else:
                msg = "Encountered an NA values with skipna=False"
            raise ValueError(msg)

        index = data._get_axis(axis)
        result = algorithms.take(
            index._values, indices, allow_fill=True, fill_value=index._na_value
        )
        final_result = data._constructor_sliced(result, index=data._get_agg_axis(axis))
        return final_result.__finalize__(self, method="idxmin")
```



---

### Функция ID: 163

**Исходный код:**
```java
@Override
    public boolean equals(final Object obj) {
        if (!(obj instanceof FastDateFormat)) {
            return false;
        }
        final FastDateFormat other = (FastDateFormat) obj;
        // no need to check parser, as it has same invariants as printer
        return printer.equals(other.printer);
    }
```



---

### Функция ID: 164

**Исходный код:**
```java
@CanIgnoreReturnValue
  public static long copy(Readable from, Appendable to) throws IOException {
    // The most common case is that from is a Reader (like InputStreamReader or StringReader) so
    // take advantage of that.
    if (from instanceof Reader) {
      // optimize for common output types which are optimized to deal with char[]
      if (to instanceof StringBuilder) {
        return copyReaderToBuilder((Reader) from, (StringBuilder) to);
      } else {
        return copyReaderToWriter((Reader) from, asWriter(to));
      }
    }

    checkNotNull(from);
    checkNotNull(to);
    long total = 0;
    CharBuffer buf = createBuffer();
    while (from.read(buf) != -1) {
      Java8Compatibility.flip(buf);
      to.append(buf);
      total += buf.remaining();
      Java8Compatibility.clear(buf);
    }
    return total;
  }
```



---

### Функция ID: 165

**Исходный код:**
```python
def walk(self, where: str = "/") -> Iterator[tuple[str, list[str], list[str]]]:
        """
        Walk the pytables group hierarchy for pandas objects.

        This generator will yield the group path, subgroups and pandas object
        names for each group.

        Any non-pandas PyTables objects that are not a group will be ignored.

        The `where` group itself is listed first (preorder), then each of its
        child groups (following an alphanumerical order) is also traversed,
        following the same procedure.

        Parameters
        ----------
        where : str, default "/"
            Group where to start walking.

        Yields
        ------
        path : str
            Full path to a group (without trailing '/').
        groups : list
            Names (strings) of the groups contained in `path`.
        leaves : list
            Names (strings) of the pandas objects contained in `path`.

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df1, format="table")  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["A", "B"])
        >>> store.append("data", df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        >>> for group in store.walk():  # doctest: +SKIP
        ...     print(group)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        _tables()
        self._check_if_open()
        assert self._handle is not None  # for mypy
        assert _table_mod is not None  # for mypy

        for g in self._handle.walk_groups(where):
            if getattr(g._v_attrs, "pandas_type", None) is not None:
                continue

            groups = []
            leaves = []
            for child in g._v_children.values():
                pandas_type = getattr(child._v_attrs, "pandas_type", None)
                if pandas_type is None:
                    if isinstance(child, _table_mod.group.Group):
                        groups.append(child._v_name)
                else:
                    leaves.append(child._v_name)

            yield (g._v_pathname.rstrip("/"), groups, leaves)
```



---

### Функция ID: 166

**Исходный код:**
```python
def isin(self, values, level: str_t | int | None = None) -> npt.NDArray[np.bool_]:
        """
        Return a boolean array where the index values are in `values`.

        Compute boolean array of whether each index value is found in the
        passed set of values. The length of the returned boolean array matches
        the length of the index.

        Parameters
        ----------
        values : set or list-like
            Sought values.
        level : str or int, optional
            Name or position of the index level to use (if the index is a
            `MultiIndex`).

        Returns
        -------
        np.ndarray[bool]
            NumPy array of boolean values.

        See Also
        --------
        Series.isin : Same for Series.
        DataFrame.isin : Same method for DataFrames.

        Notes
        -----
        In the case of `MultiIndex` you must either specify `values` as a
        list-like object containing tuples that are the same length as the
        number of levels, or specify `level`. Otherwise it will raise a
        ``ValueError``.

        If `level` is specified:

        - if it is the name of one *and only one* index level, use that level;
        - otherwise it should be a number indicating level position.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')

        Check whether each index value in a list of values.

        >>> idx.isin([1, 4])
        array([ True, False, False])

        >>> midx = pd.MultiIndex.from_arrays(
        ...     [[1, 2, 3], ["red", "blue", "green"]], names=["number", "color"]
        ... )
        >>> midx
        MultiIndex([(1,   'red'),
                    (2,  'blue'),
                    (3, 'green')],
                   names=['number', 'color'])

        Check whether the strings in the 'color' level of the MultiIndex
        are in a list of colors.

        >>> midx.isin(["red", "orange", "yellow"], level="color")
        array([ True, False, False])

        To check across the levels of a MultiIndex, pass a list of tuples:

        >>> midx.isin([(1, "red"), (3, "red")])
        array([ True, False, False])
        """
        if level is not None:
            self._validate_index_level(level)
        return algos.isin(self._values, values)
```



---

### Функция ID: 167

**Исходный код:**
```java
public <T> StrBuilder appendAll(@SuppressWarnings("unchecked") final T... array) {
        /*
         * @SuppressWarnings used to hide warning about vararg usage. We cannot
         * use @SafeVarargs, since this method is not final. Using @SuppressWarnings
         * is fine, because it isn't inherited by subclasses, so each subclass must
         * vouch for itself whether its use of 'array' is safe.
         */
        if (ArrayUtils.isNotEmpty(array)) {
            for (final Object element : array) {
                append(element);
            }
        }
        return this;
    }
```



---

### Функция ID: 168

**Исходный код:**
```python
def maybe_cast_to_integer_array(arr: list | np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Takes any dtype and returns the casted version, raising for when data is
    incompatible with integer/unsigned integer dtypes.

    Parameters
    ----------
    arr : np.ndarray or list
        The array to cast.
    dtype : np.dtype
        The integer dtype to cast the array to.

    Returns
    -------
    ndarray
        Array of integer or unsigned integer dtype.

    Raises
    ------
    OverflowError : the dtype is incompatible with the data
    ValueError : loss of precision has occurred during casting

    Examples
    --------
    If you try to coerce negative values to unsigned integers, it raises:

    >>> pd.Series([-1], dtype="uint64")
    Traceback (most recent call last):
        ...
    OverflowError: Trying to coerce negative values to unsigned integers

    Also, if you try to coerce float values to integers, it raises:

    >>> maybe_cast_to_integer_array([1, 2, 3.5], dtype=np.dtype("int64"))
    Traceback (most recent call last):
        ...
    ValueError: Trying to coerce float values to integers
    """
    assert dtype.kind in "iu"

    try:
        if not isinstance(arr, np.ndarray):
            with warnings.catch_warnings():
                # We already disallow dtype=uint w/ negative numbers
                # (test_constructor_coercion_signed_to_unsigned) so safe to ignore.
                warnings.filterwarnings(
                    "ignore",
                    "NumPy will stop allowing conversion of out-of-bound Python int",
                    DeprecationWarning,
                )
                casted = np.asarray(arr, dtype=dtype)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                casted = arr.astype(dtype, copy=False)
    except OverflowError as err:
        raise OverflowError(
            "The elements provided in the data cannot all be "
            f"casted to the dtype {dtype}"
        ) from err

    if isinstance(arr, np.ndarray) and arr.dtype == dtype:
        # avoid expensive array_equal check
        return casted

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore", "elementwise comparison failed", FutureWarning
        )
        if np.array_equal(arr, casted):
            return casted

    # We do this casting to allow for proper
    # data and dtype checking.
    #
    # We didn't do this earlier because NumPy
    # doesn't handle `uint64` correctly.
    arr = np.asarray(arr)

    if np.issubdtype(arr.dtype, str):
        # TODO(numpy-2.0 min): This case will raise an OverflowError above
        if (casted.astype(str) == arr).all():
            return casted
        raise ValueError(f"string values cannot be losslessly cast to {dtype}")

    if dtype.kind == "u" and (arr < 0).any():
        # TODO: can this be hit anymore after numpy 2.0?
        raise OverflowError("Trying to coerce negative values to unsigned integers")

    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all():
            raise IntCastingNaNError(
                "Cannot convert non-finite values (NA or inf) to integer"
            )
        raise ValueError("Trying to coerce float values to integers")
    if arr.dtype == object:
        raise ValueError("Trying to coerce object values to integers")

    if casted.dtype < arr.dtype:
        # TODO: Can this path be hit anymore with numpy > 2
        # GH#41734 e.g. [1, 200, 923442] and dtype="int8" -> overflows
        raise ValueError(
            f"Values are too large to be losslessly converted to {dtype}. "
            f"To cast anyway, use pd.Series(values).astype({dtype})"
        )

    if arr.dtype.kind in "mM":
        # test_constructor_maskedarray_nonfloat
        raise TypeError(
            f"Constructing a Series or DataFrame from {arr.dtype} values and "
            f"dtype={dtype} is not supported. Use values.view({dtype}) instead."
        )

    # No known cases that get here, but raising explicitly to cover our bases.
    raise ValueError(f"values cannot be losslessly cast to {dtype}")
```



---

### Функция ID: 169

**Исходный код:**
```python
def insert(self, loc: int, item) -> Index:
        """
        Make new Index inserting new item at location.

        Follows Python numpy.insert semantics for negative values.

        Parameters
        ----------
        loc : int
            The integer location where the new item will be inserted.
        item : object
            The new item to be inserted into the Index.

        Returns
        -------
        Index
            Returns a new Index object resulting from inserting the specified item at
            the specified location within the original Index.

        See Also
        --------
        Index.append : Append a collection of Indexes together.

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.insert(1, "x")
        Index(['a', 'x', 'b', 'c'], dtype='str')
        """
        item = lib.item_from_zerodim(item)
        if is_valid_na_for_dtype(item, self.dtype) and self.dtype != object:
            item = self._na_value

        arr = self._values

        if using_string_dtype() and len(self) == 0 and self.dtype == np.object_:
            # special case: if we are an empty object-dtype Index, also
            # take into account the inserted item for the resulting dtype
            # (https://github.com/pandas-dev/pandas/pull/60797)
            dtype = self._find_common_type_compat(item)
            if dtype != self.dtype:
                return self.astype(dtype).insert(loc, item)

        try:
            if isinstance(arr, ExtensionArray):
                res_values = arr.insert(loc, item)
                return type(self)._simple_new(res_values, name=self.name)
            else:
                item = self._validate_fill_value(item)
        except (TypeError, ValueError, LossySetitemError):
            # e.g. trying to insert an integer into a DatetimeIndex
            #  We cannot keep the same dtype, so cast to the (often object)
            #  minimal shared dtype before doing the insert.
            dtype = self._find_common_type_compat(item)
            if dtype == self.dtype:
                # EA's might run into recursion errors if loc is invalid
                raise
            return self.astype(dtype).insert(loc, item)

        if arr.dtype != object or not isinstance(
            item, (tuple, np.datetime64, np.timedelta64)
        ):
            # with object-dtype we need to worry about numpy incorrectly casting
            # dt64/td64 to integer, also about treating tuples as sequences
            # special-casing dt64/td64 https://github.com/numpy/numpy/issues/12550
            casted = arr.dtype.type(item)
            new_values = np.insert(arr, loc, casted)

        else:
            # error: No overload variant of "insert" matches argument types
            # "ndarray[Any, Any]", "int", "None"
            new_values = np.insert(arr, loc, None)  # type: ignore[call-overload]
            loc = loc if loc >= 0 else loc - 1
            new_values[loc] = item

        # GH#51363 stopped doing dtype inference here
        out = Index(new_values, dtype=new_values.dtype, name=self.name)
        return out
```



---

### Функция ID: 170

**Исходный код:**
```java
public Fraction negate() {
        // the positive range is one smaller than the negative range of an int.
        if (numerator == Integer.MIN_VALUE) {
            throw new ArithmeticException("overflow: too large to negate");
        }
        return new Fraction(-numerator, denominator);
    }
```



---

### Функция ID: 171

**Исходный код:**
```python
def prepare_code_snippet(file_path: Path, line_no: int, context_lines_count: int = 5) -> str:
    """
    Prepare code snippet with line numbers and  a specific line marked.

    :param file_path: File name
    :param line_no: Line number
    :param context_lines_count: The number of lines that will be cut before and after.
    :return: str
    """
    code_lines = file_path.read_text().splitlines()
    # Prepend line number
    code_lines = [
        f">{lno:3} | {line}" if line_no == lno else f"{lno:4} | {line}"
        for lno, line in enumerate(code_lines, 1)
    ]
    # # Cut out the snippet
    start_line_no = max(0, line_no - context_lines_count - 1)
    end_line_no = line_no + context_lines_count
    code_lines = code_lines[start_line_no:end_line_no]
    # Join lines
    code = "\n".join(code_lines)
    return code
```



---

### Функция ID: 172

**Исходный код:**
```python
def take(
        self: MultiIndex,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> MultiIndex:
        """
        Return a new MultiIndex of the values selected by the indices.

        For internal compatibility with numpy arrays.

        Parameters
        ----------
        indices : array-like
            Indices to be taken.
        axis : {0 or 'index'}, optional
            The axis over which to select values, always 0 or 'index'.
        allow_fill : bool, default True
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
            from the right (the default). This is similar to
            :func:`numpy.take`.

            * True: negative values in `indices` indicate
            missing values. These values are set to `fill_value`. Any other
            other negative values raise a ``ValueError``.

        fill_value : scalar, default None
            If allow_fill=True and fill_value is not None, indices specified by
            -1 are regarded as NA. If Index doesn't hold NA, raise ValueError.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        Index
            An index formed of elements at the given indices. Will be the same
            type as self, except for RangeIndex.

        See Also
        --------
        numpy.ndarray.take: Return an array formed from the
            elements of a at the given indices.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays([["a", "b", "c"], [1, 2, 3]])
        >>> idx
        MultiIndex([('a', 1),
                    ('b', 2),
                    ('c', 3)],
                   )
        >>> idx.take([2, 2, 1, 0])
        MultiIndex([('c', 3),
                    ('c', 3),
                    ('b', 2),
                    ('a', 1)],
                   )
        """
        nv.validate_take((), kwargs)
        indices = ensure_platform_int(indices)

        # only fill if we are passing a non-None fill_value
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)

        if indices.ndim == 1 and lib.is_range_indexer(indices, len(self)):
            return self.copy()

        na_value = -1

        taken = [lab.take(indices) for lab in self.codes]
        if allow_fill:
            mask = indices == -1
            if mask.any():
                masked = []
                for new_label in taken:
                    label_values = new_label
                    label_values[mask] = na_value
                    masked.append(np.asarray(label_values))
                taken = masked

        return MultiIndex(
            levels=self.levels, codes=taken, names=self.names, verify_integrity=False
        )
```



---

### Функция ID: 173

**Исходный код:**
```python
def _log_stream_to_parsed_log_stream(
    log_stream: RawLogStream,
) -> ParsedLogStream:
    """
    Turn a str log stream into a generator of parsed log lines.

    :param log_stream: The stream to parse.
    :return: A generator of parsed log lines.
    """
    from airflow._shared.timezones.timezone import coerce_datetime

    timestamp = None
    next_timestamp = None
    idx = 0
    for line in log_stream:
        if line:
            try:
                log = StructuredLogMessage.model_validate_json(line)
            except ValidationError:
                with suppress(Exception):
                    # If we can't parse the timestamp, don't attach one to the row
                    if isinstance(line, str):
                        next_timestamp = _parse_timestamp(line)
                log = StructuredLogMessage(event=str(line), timestamp=next_timestamp)
            if log.timestamp:
                log.timestamp = coerce_datetime(log.timestamp)
                timestamp = log.timestamp
            yield timestamp, idx, log
        idx += 1
```



---

### Функция ID: 174

**Исходный код:**
```python
def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Assuming that `self` is sorted:

        ======  ================================
        `side`  returned index `i` satisfies
        ======  ================================
        left    ``self[i-1] < value <= self[i]``
        right   ``self[i-1] <= value < self[i]``
        ======  ================================

        Parameters
        ----------
        value : array-like, list or scalar
            Value(s) to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        array of ints or int
            If value is array-like, array of insertion points.
            If value is scalar, a single integer.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 5])
        >>> arr.searchsorted([4])
        array([3])
        """
        # Note: the base tests provided by pandas only test the basics.
        # We do not test
        # 1. Values outside the range of the `data_for_sorting` fixture
        # 2. Values between the values in the `data_for_sorting` fixture
        # 3. Missing values.
        arr = self.astype(object)
        if isinstance(value, ExtensionArray):
            value = value.astype(object)
        return arr.searchsorted(value, side=side, sorter=sorter)
```



---

### Функция ID: 175

**Исходный код:**
```python
def diff(
        self,
        periods: int = 1,
    ) -> NDFrameT:
        """
        First discrete difference of element.

        Calculates the difference of each element compared with another
        element in the group (default is element in previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative values.

        Returns
        -------
        Series or DataFrame
            First differences.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ["a", "a", "a", "b", "b", "b"]
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).diff()
        a    NaN
        a   -5.0
        a    6.0
        b    NaN
        b   -1.0
        b    0.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {"a": [1, 3, 5, 7, 7, 8, 3], "b": [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(
        ...     data, index=["dog", "dog", "dog", "mouse", "mouse", "mouse", "mouse"]
        ... )
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).diff()
                 a    b
          dog  NaN  NaN
          dog  2.0  3.0
          dog  2.0  4.0
        mouse  NaN  NaN
        mouse  0.0  0.0
        mouse  1.0 -2.0
        mouse -5.0 -1.0
        """
        obj = self._obj_with_exclusions
        shifted = self.shift(periods=periods)

        # GH45562 - to retain existing behavior and match behavior of Series.diff(),
        # int8 and int16 are coerced to float32 rather than float64.
        dtypes_to_f32 = ["int8", "int16"]
        if obj.ndim == 1:
            if obj.dtype in dtypes_to_f32:
                shifted = shifted.astype("float32")
        else:
            to_coerce = [c for c, dtype in obj.dtypes.items() if dtype in dtypes_to_f32]
            if to_coerce:
                shifted = shifted.astype(dict.fromkeys(to_coerce, "float32"))

        return obj - shifted
```



---

### Функция ID: 176

**Исходный код:**
```python
def nunique(self, dropna: bool = True) -> Series | DataFrame:
        """
        Return number of unique elements in the group.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        Series
            Number of unique values within each group.

        See Also
        --------
        core.resample.Resampler.nunique : Method nunique for Resampler.

        Examples
        --------
        >>> lst = ["a", "a", "b", "b"]
        >>> ser = pd.Series([1, 2, 3, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    3
        dtype: int64
        >>> ser.groupby(level=0).nunique()
        a    2
        b    1
        dtype: int64
        """
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        val = self.obj._values
        codes, uniques = algorithms.factorize(val, use_na_sentinel=dropna, sort=False)

        if self._grouper.has_dropped_na:
            mask = ids >= 0
            ids = ids[mask]
            codes = codes[mask]

        group_index = get_group_index(
            labels=[ids, codes],
            shape=(ngroups, len(uniques)),
            sort=False,
            xnull=dropna,
        )

        if dropna:
            mask = group_index >= 0
            if (~mask).any():
                ids = ids[mask]
                group_index = group_index[mask]

        mask = duplicated(group_index, "first")
        res = np.bincount(ids[~mask], minlength=ngroups)
        res = ensure_int64(res)

        ri = self._grouper.result_index
        result: Series | DataFrame = self.obj._constructor(
            res, index=ri, name=self.obj.name
        )
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result
```



---

### Функция ID: 177

**Исходный код:**
```java
static Object removeAt(final Object array, final BitSet indices) {
        if (array == null) {
            return null;
        }
        final int srcLength = getLength(array);
        // No need to check maxIndex here, because method only currently called from removeElements()
        // which guarantee to generate only valid bit entries.
//        final int maxIndex = indices.length();
//        if (maxIndex > srcLength) {
//            throw new IndexOutOfBoundsException("Index: " + (maxIndex-1) + ", Length: " + srcLength);
//        }
        final int removals = indices.cardinality(); // true bits are items to remove
        final Object result = Array.newInstance(array.getClass().getComponentType(), srcLength - removals);
        int srcIndex = 0;
        int destIndex = 0;
        int count;
        int set;
        while ((set = indices.nextSetBit(srcIndex)) != -1) {
            count = set - srcIndex;
            if (count > 0) {
                System.arraycopy(array, srcIndex, result, destIndex, count);
                destIndex += count;
            }
            srcIndex = indices.nextClearBit(set);
        }
        count = srcLength - srcIndex;
        if (count > 0) {
            System.arraycopy(array, srcIndex, result, destIndex, count);
        }
        return result;
    }
```



---

### Функция ID: 178

**Исходный код:**
```python
def kurt(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | Any:
        """
        Return unbiased kurtosis over requested axis.

        Kurtosis obtained using Fisher's definition of
        kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or scalar
            Unbiased kurtosis over requested axis.

        See Also
        --------
        Dataframe.kurtosis : Returns unbiased kurtosis over requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 2, 3], index=["cat", "dog", "dog", "mouse"])
        >>> s
        cat    1
        dog    2
        dog    2
        mouse  3
        dtype: int64
        >>> s.kurt()
        1.5

        With a DataFrame

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2, 2, 3], "b": [3, 4, 4, 4]},
        ...     index=["cat", "dog", "dog", "mouse"],
        ... )
        >>> df
               a   b
          cat  1   3
          dog  2   4
          dog  2   4
        mouse  3   4
        >>> df.kurt()
        a   1.5
        b   4.0
        dtype: float64

        With axis=None

        >>> df.kurt(axis=None).round(6)
        -0.988693

        Using axis=1

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2], "b": [3, 4], "c": [3, 4], "d": [1, 2]},
        ...     index=["cat", "dog"],
        ... )
        >>> df.kurt(axis=1)
        cat   -6.0
        dog   -6.0
        dtype: float64
        """
        result = super().kurt(
            axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )
        if isinstance(result, Series):
            result = result.__finalize__(self, method="kurt")
        return result
```



---

### Функция ID: 179

**Исходный код:**
```typescript
function getIndentationForNodeWorker(

        current: Node,

        currentStart: LineAndCharacter,

        ignoreActualIndentationRange: TextRange | undefined,

        indentationDelta: number,

        sourceFile: SourceFile,

        isNextChild: boolean,

        options: EditorSettings,

    ): number {

        let parent = current.parent;



        // Walk up the tree and collect indentation for parent-child node pairs. Indentation is not added if

        // * parent and child nodes start on the same line, or

        // * parent is an IfStatement and child starts on the same line as an 'else clause'.

        while (parent) {

            let useActualIndentation = true;

            if (ignoreActualIndentationRange) {

                const start = current.getStart(sourceFile);

                useActualIndentation = start < ignoreActualIndentationRange.pos || start > ignoreActualIndentationRange.end;

            }



            const containingListOrParentStart = getContainingListOrParentStart(parent, current, sourceFile);

            const parentAndChildShareLine = containingListOrParentStart.line === currentStart.line ||

                childStartsOnTheSameLineWithElseInIfStatement(parent, current, currentStart.line, sourceFile);



            if (useActualIndentation) {

                // check if current node is a list item - if yes, take indentation from it

                const firstListChild = getContainingList(current, sourceFile)?.[0];

                // A list indents its children if the children begin on a later line than the list itself:

                //

                // f1(               L0 - List start

                //   {               L1 - First child start: indented, along with all other children

                //     prop: 0

                //   },

                //   {

                //     prop: 1

                //   }

                // )

                //

                // f2({             L0 - List start and first child start: children are not indented.

                //   prop: 0             Object properties are indented only one level, because the list

                // }, {                  itself contributes nothing.

                //   prop: 1        L3 - The indentation of the second object literal is best understood by

                // })                    looking at the relationship between the list and *first* list item.

                const listIndentsChild = !!firstListChild && getStartLineAndCharacterForNode(firstListChild, sourceFile).line > containingListOrParentStart.line;

                let actualIndentation = getActualIndentationForListItem(current, sourceFile, options, listIndentsChild);

                if (actualIndentation !== Value.Unknown) {

                    return actualIndentation + indentationDelta;

                }



                // try to fetch actual indentation for current node from source text

                actualIndentation = getActualIndentationForNode(current, parent, currentStart, parentAndChildShareLine, sourceFile, options);

                if (actualIndentation !== Value.Unknown) {

                    return actualIndentation + indentationDelta;

                }

            }



            // increase indentation if parent node wants its content to be indented and parent and child nodes don't start on the same line

            if (shouldIndentChildNode(options, parent, current, sourceFile, isNextChild) && !parentAndChildShareLine) {

                indentationDelta += options.indentSize!;

            }



            // In our AST, a call argument's `parent` is the call-expression, not the argument list.

            // We would like to increase indentation based on the relationship between an argument and its argument-list,

            // so we spoof the starting position of the (parent) call-expression to match the (non-parent) argument-list.

            // But, the spoofed start-value could then cause a problem when comparing the start position of the call-expression

            // to *its* parent (in the case of an iife, an expression statement), adding an extra level of indentation.

            //

            // Instead, when at an argument, we unspoof the starting position of the enclosing call expression

            // *after* applying indentation for the argument.



            const useTrueStart = isArgumentAndStartLineOverlapsExpressionBeingCalled(parent, current, currentStart.line, sourceFile);



            current = parent;

            parent = current.parent;

            currentStart = useTrueStart ? sourceFile.getLineAndCharacterOfPosition(current.getStart(sourceFile)) : containingListOrParentStart;

        }



        return indentationDelta + getBaseIndentation(options);

    }
```



---

### Функция ID: 180

**Исходный код:**
```python
def _list_all(self, api_call: Callable, response_key: str, verbose: bool) -> list:
        """
        Repeatedly call a provided boto3 API Callable and collates the responses into a List.

        :param api_call: The api command to execute.
        :param response_key: Which dict key to collect into the final list.
        :param verbose: Provides additional logging if set to True.  Defaults to False.
        :return: A List of the combined results of the provided API call.
        """
        name_collection: list = []
        token: str | None = DEFAULT_PAGINATION_TOKEN

        while token is not None:
            response = api_call(nextToken=token)
            # If response list is not empty, append it to the running list.
            name_collection += filter(None, response.get(response_key))
            token = response.get("nextToken")

        self.log.info("Retrieved list of %s %s.", len(name_collection), response_key)
        if verbose:
            self.log.info("%s found: %s", response_key.title(), name_collection)

        return name_collection
```



---

### Функция ID: 181

**Исходный код:**
```python
def difference(self, other, sort: bool | None = None):
        """
        Return a new Index with elements of index not in `other`.

        This is the set difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
            Index object or an array-like object containing elements to be compared
            with the elements of the original Index.
        sort : bool or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index
            Returns a new Index object containing elements that are in the original
            Index but not in the `other` Index.

        See Also
        --------
        Index.symmetric_difference : Compute the symmetric difference of two Index
            objects.
        Index.intersection : Form the intersection of two Index objects.

        Examples
        --------
        >>> idx1 = pd.Index([2, 1, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.difference(idx2)
        Index([1, 2], dtype='int64')
        >>> idx1.difference(idx2, sort=False)
        Index([2, 1], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)

        # Note: we do NOT call _dti_setop_align_tzs here, as there
        #  is no requirement that .difference be commutative, so it does
        #  not cast to object.

        if self.equals(other):
            # Note: we do not (yet) sort even if sort=None GH#24959
            return self[:0].rename(result_name)

        if len(other) == 0:
            # Note: we do not (yet) sort even if sort=None GH#24959
            result = self.unique().rename(result_name)
            if sort is True:
                return result.sort_values()
            return result

        if not self._should_compare(other):
            # Nothing matches -> difference is everything
            result = self.unique().rename(result_name)
            if sort is True:
                return result.sort_values()
            return result

        result = self._difference(other, sort=sort)
        return self._wrap_difference_result(other, result)
```



---

### Функция ID: 182

**Исходный код:**
```java
public int runAndHandleErrors(String... args) {
		String[] argsWithoutDebugFlags = removeDebugFlags(args);
		boolean debug = argsWithoutDebugFlags.length != args.length;
		if (debug) {
			System.setProperty("debug", "true");
		}
		try {
			ExitStatus result = run(argsWithoutDebugFlags);
			// The caller will hang up if it gets a non-zero status
			if (result != null && result.isHangup()) {
				return (result.getCode() > 0) ? result.getCode() : 0;
			}
			return 0;
		}
		catch (NoArgumentsException ex) {
			showUsage();
			return 1;
		}
		catch (Exception ex) {
			return handleError(debug, ex);
		}
	}
```



---

### Функция ID: 183

**Исходный код:**
```java
public Map<String, Object> parse(Map<?, ?> props) {
        // Check all configurations are defined
        List<String> undefinedConfigKeys = undefinedDependentConfigs();
        if (!undefinedConfigKeys.isEmpty()) {
            String joined = undefinedConfigKeys.stream().map(String::toString).collect(Collectors.joining(","));
            throw new ConfigException("Some configurations in are referred in the dependents, but not defined: " + joined);
        }
        // parse all known keys
        Map<String, Object> values = new HashMap<>();
        for (ConfigKey key : configKeys.values())
            values.put(key.name, parseValue(key, props.get(key.name), props.containsKey(key.name)));
        return values;
    }
```



---

### Функция ID: 184

**Исходный код:**
```java
@Override
    public int write(ByteBuffer src) throws IOException {
        if (state == State.CLOSING)
            throw closingException();
        if (!ready())
            return 0;

        int written = 0;
        while (flush(netWriteBuffer) && src.hasRemaining()) {
            netWriteBuffer.clear();
            SSLEngineResult wrapResult = sslEngine.wrap(src, netWriteBuffer);
            netWriteBuffer.flip();

            // reject renegotiation if TLS < 1.3, key updates for TLS 1.3 are allowed
            if (wrapResult.getHandshakeStatus() != HandshakeStatus.NOT_HANDSHAKING &&
                    wrapResult.getStatus() == Status.OK &&
                    !sslEngine.getSession().getProtocol().equals(TLS13)) {
                throw renegotiationException();
            }

            if (wrapResult.getStatus() == Status.OK) {
                written += wrapResult.bytesConsumed();
            } else if (wrapResult.getStatus() == Status.BUFFER_OVERFLOW) {
                // BUFFER_OVERFLOW means that the last `wrap` call had no effect, so we expand the buffer and try again
                netWriteBuffer = Utils.ensureCapacity(netWriteBuffer, netWriteBufferSize());
                netWriteBuffer.position(netWriteBuffer.limit());
            } else if (wrapResult.getStatus() == Status.BUFFER_UNDERFLOW) {
                throw new IllegalStateException("SSL BUFFER_UNDERFLOW during write");
            } else if (wrapResult.getStatus() == Status.CLOSED) {
                throw new EOFException();
            }
        }
        return written;
    }
```



---

### Функция ID: 185

**Исходный код:**
```python
def reconstruct_func(
    func: AggFuncType | None, **kwargs
) -> tuple[bool, AggFuncType, tuple[str, ...] | None, npt.NDArray[np.intp] | None]:
    """
    This is the internal function to reconstruct func given if there is relabeling
    or not and also normalize the keyword to get new order of columns.

    If named aggregation is applied, `func` will be None, and kwargs contains the
    column and aggregation function information to be parsed;
    If named aggregation is not applied, `func` is either string (e.g. 'min') or
    Callable, or list of them (e.g. ['min', np.max]), or the dictionary of column name
    and str/Callable/list of them (e.g. {'A': 'min'}, or {'A': [np.min, lambda x: x]})

    If relabeling is True, will return relabeling, reconstructed func, column
    names, and the reconstructed order of columns.
    If relabeling is False, the columns and order will be None.

    Parameters
    ----------
    func: agg function (e.g. 'min' or Callable) or list of agg functions
        (e.g. ['min', np.max]) or dictionary (e.g. {'A': ['min', np.max]}).
    **kwargs: dict, kwargs used in is_multi_agg_with_relabel and
        normalize_keyword_aggregation function for relabelling

    Returns
    -------
    relabelling: bool, if there is relabelling or not
    func: normalized and mangled func
    columns: tuple of column names
    order: array of columns indices

    Examples
    --------
    >>> reconstruct_func(None, **{"foo": ("col", "min")})
    (True, defaultdict(<class 'list'>, {'col': ['min']}), ('foo',), array([0]))

    >>> reconstruct_func("min")
    (False, 'min', None, None)
    """
    from pandas.core.groupby.generic import NamedAgg

    relabeling = func is None and (
        is_multi_agg_with_relabel(**kwargs)
        or any(isinstance(v, NamedAgg) for v in kwargs.values())
    )

    columns: tuple[str, ...] | None = None
    order: npt.NDArray[np.intp] | None = None

    if not relabeling:
        if isinstance(func, list) and len(func) > len(set(func)):
            # GH 28426 will raise error if duplicated function names are used and
            # there is no reassigned name
            raise SpecificationError(
                "Function names must be unique if there is no new column names assigned"
            )
        if func is None:
            # nicer error message
            raise TypeError("Must provide 'func' or tuples of '(column, aggfunc).")

    if relabeling:
        # error: Incompatible types in assignment (expression has type
        # "MutableMapping[Hashable, list[Callable[..., Any] | str]]", variable has type
        # "Callable[..., Any] | str | list[Callable[..., Any] | str] |
        # MutableMapping[Hashable, Callable[..., Any] | str | list[Callable[..., Any] |
        # str]] | None")
        converted_kwargs = {}
        for key, val in kwargs.items():
            if isinstance(val, NamedAgg):
                aggfunc = val.aggfunc
                if val.args or val.kwargs:
                    aggfunc = lambda x, func=aggfunc, a=val.args, kw=val.kwargs: func(
                        x, *a, **kw
                    )
                converted_kwargs[key] = (val.column, aggfunc)
            else:
                converted_kwargs[key] = val

        func, columns, order = normalize_keyword_aggregation(  # type: ignore[assignment]
            converted_kwargs
        )

    assert func is not None

    return relabeling, func, columns, order
```



---

### Функция ID: 186

**Исходный код:**
```typescript
function shouldAddOverrideKeyword(): boolean {

        return !!(context.program.getCompilerOptions().noImplicitOverride && declaration && hasAbstractModifier(declaration));

    }
```



---

### Функция ID: 187

**Исходный код:**
```java
private JSONObject readObject() throws JSONException {
		JSONObject result = new JSONObject();

		/* Peek to see if this is the empty object. */
		int first = nextCleanInternal();
		if (first == '}') {
			return result;
		}
		else if (first != -1) {
			this.pos--;
		}

		while (true) {
			Object name = nextValue();
			if (!(name instanceof String)) {
				if (name == null) {
					throw syntaxError("Names cannot be null");
				}
				else {
					throw syntaxError(
							"Names must be strings, but " + name + " is of type " + name.getClass().getName());
				}
			}

			/*
			 * Expect the name/value separator to be either a colon ':', an equals sign
			 * '=', or an arrow "=>". The last two are bogus but we include them because
			 * that's what the original implementation did.
			 */
			int separator = nextCleanInternal();
			if (separator != ':' && separator != '=') {
				throw syntaxError("Expected ':' after " + name);
			}
			if (this.pos < this.in.length() && this.in.charAt(this.pos) == '>') {
				this.pos++;
			}

			result.put((String) name, nextValue());

			switch (nextCleanInternal()) {
				case '}':
					return result;
				case ';', ',':
					continue;
				default:
					throw syntaxError("Unterminated object");
			}
		}
	}
```



---

### Функция ID: 188

**Исходный код:**
```python
def explode(self, ignore_index: bool = False) -> Series:
        """
        Transform each element of a list-like to a row.

        Parameters
        ----------
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        Series
            Exploded lists to rows; index will be duplicated for these rows.

        See Also
        --------
        Series.str.split : Split string values on specified separator.
        Series.unstack : Unstack, a.k.a. pivot, Series with MultiIndex
            to produce DataFrame.
        DataFrame.melt : Unpivot a DataFrame from wide format to long format.
        DataFrame.explode : Explode a DataFrame from list-like
            columns to long format.

        Notes
        -----
        This routine will explode list-likes including lists, tuples, sets,
        Series, and np.ndarray. The result dtype of the subset rows will
        be object. Scalars will be returned unchanged, and empty list-likes will
        result in an np.nan for that row. In addition, the ordering of elements in
        the output will be non-deterministic when exploding sets.

        Reference :ref:`the user guide <reshaping.explode>` for more examples.

        Examples
        --------
        >>> s = pd.Series([[1, 2, 3], "foo", [], [3, 4]])
        >>> s
        0    [1, 2, 3]
        1          foo
        2           []
        3       [3, 4]
        dtype: object

        >>> s.explode()
        0      1
        0      2
        0      3
        1    foo
        2    NaN
        3      3
        3      4
        dtype: object
        """
        if isinstance(self.dtype, ExtensionDtype):
            values, counts = self._values._explode()
        elif len(self) and is_object_dtype(self.dtype):
            values, counts = reshape.explode(np.asarray(self._values))
        else:
            result = self.copy()
            return result.reset_index(drop=True) if ignore_index else result

        if ignore_index:
            index: Index = default_index(len(values))
        else:
            index = self.index.repeat(counts)

        return self._constructor(values, index=index, name=self.name, copy=False)
```



---

### Функция ID: 189

**Исходный код:**
```python
def get_indexer_non_unique(
        self, target
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        Compute indexer and mask for new index given the current index.

        The indexer should be then used as an input to ndarray.take to align the
        current data to the new index.

        Parameters
        ----------
        target : Index
            An iterable containing the values to be used for computing indexer.

        Returns
        -------
        indexer : np.ndarray[np.intp]
            Integers from 0 to n - 1 indicating that the index at these
            positions matches the corresponding target values. Missing values
            in the target are marked by -1.
        missing : np.ndarray[np.intp]
            An indexer into the target of the values not found.
            These correspond to the -1 in the indexer array.

        See Also
        --------
        Index.get_indexer : Computes indexer and mask for new index given
            the current index.
        Index.get_indexer_for : Returns an indexer even when non-unique.

        Examples
        --------
        >>> index = pd.Index(["c", "b", "a", "b", "b"])
        >>> index.get_indexer_non_unique(["b", "b"])
        (array([1, 3, 4, 1, 3, 4]), array([], dtype=int64))

        In the example below there are no matched values.

        >>> index = pd.Index(["c", "b", "a", "b", "b"])
        >>> index.get_indexer_non_unique(["q", "r", "t"])
        (array([-1, -1, -1]), array([0, 1, 2]))

        For this reason, the returned ``indexer`` contains only integers equal to -1.
        It demonstrates that there's no match between the index and the ``target``
        values at these positions. The mask [0, 1, 2] in the return value shows that
        the first, second, and third elements are missing.

        Notice that the return value is a tuple contains two items. In the example
        below the first item is an array of locations in ``index``. The second
        item is a mask shows that the first and third elements are missing.

        >>> index = pd.Index(["c", "b", "a", "b", "b"])
        >>> index.get_indexer_non_unique(["f", "b", "s"])
        (array([-1,  1,  3,  4, -1]), array([0, 2]))
        """
        target = self._maybe_cast_listlike_indexer(target)

        if not self._should_compare(target) and not self._should_partial_index(target):
            # _should_partial_index e.g. IntervalIndex with numeric scalars
            #  that can be matched to Interval scalars.
            return self._get_indexer_non_comparable(target, method=None, unique=False)

        pself, ptarget = self._maybe_downcast_for_indexing(target)
        if pself is not self or ptarget is not target:
            return pself.get_indexer_non_unique(ptarget)

        if self.dtype != target.dtype:
            # TODO: if object, could use infer_dtype to preempt costly
            #  conversion if still non-comparable?
            dtype = self._find_common_type_compat(target)

            this = self.astype(dtype, copy=False)
            that = target.astype(dtype, copy=False)
            return this.get_indexer_non_unique(that)

        # TODO: get_indexer has fastpaths for both Categorical-self and
        #  Categorical-target. Can we do something similar here?

        # Note: _maybe_downcast_for_indexing ensures we never get here
        #  with MultiIndex self and non-Multi target
        if self._is_multi and target._is_multi:
            engine = self._engine
            # Item "IndexEngine" of "Union[IndexEngine, ExtensionEngine]" has
            # no attribute "_extract_level_codes"
            tgt_values = engine._extract_level_codes(target)  # type: ignore[union-attr]
        else:
            tgt_values = target._get_engine_target()

        indexer, missing = self._engine.get_indexer_non_unique(tgt_values)
        return ensure_platform_int(indexer), ensure_platform_int(missing)
```



---

### Функция ID: 190

**Исходный код:**
```python
def pickle_flatten(
    obj: object, cls: type[T] | tuple[type[T], ...]
) -> tuple[list[T], FlattenRest]:
    """
    Use the pickle machinery to extract objects out of an arbitrary container.

    Unlike regular ``pickle.dumps``, this function always succeeds.

    Parameters
    ----------
    obj : object
        The object to pickle.
    cls : type | tuple[type, ...]
        One or multiple classes to extract from the object.
        The instances of these classes inside ``obj`` will not be pickled.

    Returns
    -------
    instances : list[cls]
        All instances of ``cls`` found inside ``obj`` (not pickled).
    rest
        Opaque object containing the pickled bytes plus all other objects where
        ``__reduce__`` / ``__reduce_ex__`` is either not implemented or raised.
        These are unpickleable objects, types, modules, and functions.

        This object is *typically* hashable save for fairly exotic objects
        that are neither pickleable nor hashable.

        This object is pickleable if everything except ``instances`` was pickleable
        in the input object.

    See Also
    --------
    pickle_unflatten : Reverse function.

    Examples
    --------
    >>> class A:
    ...     def __repr__(self):
    ...         return "<A>"
    >>> class NS:
    ...     def __repr__(self):
    ...         return "<NS>"
    ...     def __reduce__(self):
    ...         assert False, "not serializable"
    >>> obj = {1: A(), 2: [A(), NS(), A()]}
    >>> instances, rest = pickle_flatten(obj, A)
    >>> instances
    [<A>, <A>, <A>]
    >>> pickle_unflatten(instances, rest)
    {1: <A>, 2: [<A>, <NS>, <A>]}

    This can be also used to swap inner objects; the only constraint is that
    the number of objects in and out must be the same:

    >>> pickle_unflatten(["foo", "bar", "baz"], rest)
    {1: "foo", 2: ["bar", <NS>, "baz"]}
    """
    instances: list[T] = []
    rest: list[object] = []

    class Pickler(pickle.Pickler):  # numpydoc ignore=GL08
        """
        Use the `pickle.Pickler.persistent_id` hook to extract objects.
        """

        @override
        def persistent_id(
            self, obj: object
        ) -> Literal[0, 1, None]:  # numpydoc ignore=GL08
            if isinstance(obj, cls):
                instances.append(obj)  # type: ignore[arg-type]
                return 0

            typ_ = type(obj)
            if typ_ in _BASIC_PICKLED_TYPES:  # No subclasses!
                # If obj is a collection, recursively descend inside it
                return None
            if typ_ in _BASIC_REST_TYPES:
                rest.append(obj)
                return 1

            try:
                # Note: a class that defines __slots__ without defining __getstate__
                # cannot be pickled with __reduce__(), but can with __reduce_ex__(5)
                _ = obj.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
            except Exception:  # pylint: disable=broad-exception-caught
                rest.append(obj)
                return 1

            # Object can be pickled. Let the Pickler recursively descend inside it.
            return None

    f = io.BytesIO()
    p = Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
    p.dump(obj)
    return instances, (f.getvalue(), *rest)
```



---

### Функция ID: 191

**Исходный код:**
```python
def skew(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | Any:
        """
        Return unbiased skew over requested axis.

        Normalized by N-1.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or scalar
            Unbiased skew over requested axis.

        See Also
        --------
        Dataframe.kurt : Returns unbiased kurtosis over requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.skew()
        0.0

        With a DataFrame

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2, 3], "b": [2, 3, 4], "c": [1, 3, 5]},
        ...     index=["tiger", "zebra", "cow"],
        ... )
        >>> df
                a   b   c
        tiger   1   2   1
        zebra   2   3   3
        cow     3   4   5
        >>> df.skew()
        a   0.0
        b   0.0
        c   0.0
        dtype: float64

        Using axis=1

        >>> df.skew(axis=1)
        tiger   1.732051
        zebra  -1.732051
        cow     0.000000
        dtype: float64

        In this case, `numeric_only` should be set to `True` to avoid
        getting an error.

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2, 3], "b": ["T", "Z", "X"]}, index=["tiger", "zebra", "cow"]
        ... )
        >>> df.skew(numeric_only=True)
        a   0.0
        dtype: float64
        """
        result = super().skew(
            axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )
        if isinstance(result, Series):
            result = result.__finalize__(self, method="skew")
        return result
```



---

### Функция ID: 192

**Исходный код:**
```java
@Override
    public <B extends Appendable> B format(final Calendar calendar, final B buf) {
        // Don't edit the given Calendar, clone it only if needed.
        Calendar actual = calendar;
        if (!calendar.getTimeZone().equals(timeZone)) {
            actual = (Calendar) calendar.clone();
            actual.setTimeZone(timeZone);
        }
        return applyRules(actual, buf);
    }
```



---

### Функция ID: 193

**Исходный код:**
```java
private RequestFuture<Map<TopicPartition, OffsetAndMetadata>> sendOffsetFetchRequest(Set<TopicPartition> partitions) {
        Node coordinator = checkAndGetCoordinator();
        if (coordinator == null)
            return RequestFuture.coordinatorNotAvailable();

        log.debug("Fetching committed offsets for partitions: {}", partitions);

        // construct the request
        List<OffsetFetchRequestData.OffsetFetchRequestTopics> topics = partitions.stream()
            .collect(Collectors.groupingBy(TopicPartition::topic))
            .entrySet()
            .stream()
            .map(entry -> new OffsetFetchRequestData.OffsetFetchRequestTopics()
                .setName(entry.getKey())
                .setPartitionIndexes(entry.getValue().stream()
                    .map(TopicPartition::partition)
                    .collect(Collectors.toList())))
            .collect(Collectors.toList());

        OffsetFetchRequest.Builder requestBuilder = OffsetFetchRequest.Builder.forTopicNames(
            new OffsetFetchRequestData()
                .setRequireStable(true)
                .setGroups(List.of(
                    new OffsetFetchRequestData.OffsetFetchRequestGroup()
                        .setGroupId(this.rebalanceConfig.groupId)
                        .setTopics(topics))),
            throwOnFetchStableOffsetsUnsupported);

        // send the request with a callback
        return client.send(coordinator, requestBuilder)
                .compose(new OffsetFetchResponseHandler());
    }
```



---

### Функция ID: 194

**Исходный код:**
```typescript
function getActualIndentationForListStartLine(list: NodeArray<Node>, sourceFile: SourceFile, options: EditorSettings): number {

        if (!list) {

            return Value.Unknown;

        }

        return findColumnForFirstNonWhitespaceCharacterInLine(sourceFile.getLineAndCharacterOfPosition(list.pos), sourceFile, options);

    }
```



---

### Функция ID: 195

**Исходный код:**
```python
def get_loc(self, key) -> int:
        """
        Get integer location for requested label.

        Parameters
        ----------
        key : int or float
            Label to locate. Integer-like floats (e.g. 3.0) are accepted and
            treated as the corresponding integer. Non-integer floats and other
            non-integer labels are not valid and will raise KeyError or
            InvalidIndexError.

        Returns
        -------
        int
            Integer location of the label within the RangeIndex.

        Raises
        ------
        KeyError
            If the label is not present in the RangeIndex or the label is a
            non-integer value.
        InvalidIndexError
            If the label is of an invalid type for the RangeIndex.

        See Also
        --------
        RangeIndex.get_slice_bound : Calculate slice bound that corresponds to
            given label.
        RangeIndex.get_indexer : Computes indexer and mask for new index given
            the current index.
        RangeIndex.get_non_unique : Returns indexer and masks for new index given
            the current index.
        RangeIndex.get_indexer_for : Returns an indexer even when non-unique.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.get_loc(3)
        3

        >>> idx = pd.RangeIndex(2, 10, 2)  # values [2, 4, 6, 8]
        >>> idx.get_loc(6)
        2
        """
        if is_integer(key) or (is_float(key) and key.is_integer()):
            new_key = int(key)
            try:
                return self._range.index(new_key)
            except ValueError as err:
                raise KeyError(key) from err
        if isinstance(key, Hashable):
            raise KeyError(key)
        self._check_indexing_error(key)
        raise KeyError(key)
```



---

### Функция ID: 196

**Исходный код:**
```python
def droplevel(self, level: IndexLabel = 0):
        """
        Return index with requested level(s) removed.

        If resulting index has only 1 level left, the result will be
        of Index type, not MultiIndex. The original index is not modified inplace.

        Parameters
        ----------
        level : int, str, or list-like, default 0
            If a string is given, must be the name of a level
            If list-like, elements must be names or indexes of levels.

        Returns
        -------
        Index or MultiIndex
            Returns an Index or MultiIndex object, depending on the resulting index
            after removing the requested level(s).

        See Also
        --------
        Index.dropna : Return Index without NA/NaN values.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(
        ...     [[1, 2], [3, 4], [5, 6]], names=["x", "y", "z"]
        ... )
        >>> mi
        MultiIndex([(1, 3, 5),
                    (2, 4, 6)],
                   names=['x', 'y', 'z'])

        >>> mi.droplevel()
        MultiIndex([(3, 5),
                    (4, 6)],
                   names=['y', 'z'])

        >>> mi.droplevel(2)
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.droplevel("z")
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.droplevel(["x", "y"])
        Index([5, 6], dtype='int64', name='z')
        """
        if not isinstance(level, (tuple, list)):
            level = [level]

        levnums = sorted((self._get_level_number(lev) for lev in level), reverse=True)

        return self._drop_level_numbers(levnums)
```



---

### Функция ID: 197

**Исходный код:**
```python
def identical(self, other) -> bool:
        """
        Similar to equals, but checks that object attributes and types are also equal.

        Parameters
        ----------
        other : Index
            The Index object you want to compare with the current Index object.

        Returns
        -------
        bool
            If two Index objects have equal elements and same type True,
            otherwise False.

        See Also
        --------
        Index.equals: Determine if two Index object are equal.
        Index.has_duplicates: Check if the Index has duplicate values.
        Index.is_unique: Return if the index has unique values.

        Examples
        --------
        >>> idx1 = pd.Index(["1", "2", "3"])
        >>> idx2 = pd.Index(["1", "2", "3"])
        >>> idx2.identical(idx1)
        True

        >>> idx1 = pd.Index(["1", "2", "3"], name="A")
        >>> idx2 = pd.Index(["1", "2", "3"], name="B")
        >>> idx2.identical(idx1)
        False
        """
        return (
            self.equals(other)
            and all(
                getattr(self, c, None) == getattr(other, c, None)
                for c in self._comparables
            )
            and type(self) == type(other)
            and self.dtype == other.dtype
        )
```



---

### Функция ID: 198

**Исходный код:**
```python
def describe_option(pat: str = "", _print_desc: bool = True) -> str | None:
    """
    Print the description for one or more registered options.

    Call with no arguments to get a listing for all registered options.

    Parameters
    ----------
    pat : str, default ""
        String or string regexp pattern.
        Empty string will return all options.
        For regexp strings, all matching keys will have their description displayed.
    _print_desc : bool, default True
        If True (default) the description(s) will be printed to stdout.
        Otherwise, the description(s) will be returned as a string
        (for testing).

    Returns
    -------
    None
        If ``_print_desc=True``.
    str
        If the description(s) as a string if ``_print_desc=False``.

    See Also
    --------
    get_option : Retrieve the value of the specified option.
    set_option : Set the value of the specified option or options.
    reset_option : Reset one or more options to their default value.

    Notes
    -----
    For all available options, please view the
    :ref:`User Guide <options.available>`.

    Examples
    --------
    >>> pd.describe_option("display.max_columns")  # doctest: +SKIP
    display.max_columns : int
        If max_cols is exceeded, switch to truncate view...
    """
    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError(f"No such keys(s) for {pat=}")

    s = "\n".join([_build_option_description(k) for k in keys])

    if _print_desc:
        print(s)
        return None
    return s
```



---

### Функция ID: 199

**Исходный код:**
```python
def _memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of the values.

        Parameters
        ----------
        deep : bool, default False
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption.

        Returns
        -------
        bytes used
            Returns memory usage of the values in the Index in bytes.

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False or if used on PyPy

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.memory_usage()
        24
        """
        if hasattr(self.array, "memory_usage"):
            return self.array.memory_usage(  # pyright: ignore[reportAttributeAccessIssue]
                deep=deep,
            )

        v = self.array.nbytes
        if deep and is_object_dtype(self.dtype) and not PYPY:
            values = cast(np.ndarray, self._values)
            v += lib.memory_usage_of_objects(values)
        return v
```



---

### Функция ID: 200

**Исходный код:**
```java
RequestFuture<Void> sendOffsetCommitRequest(final Map<TopicPartition, OffsetAndMetadata> offsets) {
        if (offsets.isEmpty())
            return RequestFuture.voidSuccess();

        Node coordinator = checkAndGetCoordinator();
        if (coordinator == null)
            return RequestFuture.coordinatorNotAvailable();

        // create the offset commit request
        Map<String, OffsetCommitRequestData.OffsetCommitRequestTopic> requestTopicDataMap = new HashMap<>();
        for (Map.Entry<TopicPartition, OffsetAndMetadata> entry : offsets.entrySet()) {
            TopicPartition topicPartition = entry.getKey();
            OffsetAndMetadata offsetAndMetadata = entry.getValue();
            if (offsetAndMetadata.offset() < 0) {
                return RequestFuture.failure(new IllegalArgumentException("Invalid offset: " + offsetAndMetadata.offset()));
            }

            OffsetCommitRequestData.OffsetCommitRequestTopic topic = requestTopicDataMap
                    .getOrDefault(topicPartition.topic(),
                            new OffsetCommitRequestData.OffsetCommitRequestTopic()
                                    .setName(topicPartition.topic())
                    );

            topic.partitions().add(new OffsetCommitRequestData.OffsetCommitRequestPartition()
                    .setPartitionIndex(topicPartition.partition())
                    .setCommittedOffset(offsetAndMetadata.offset())
                    .setCommittedLeaderEpoch(offsetAndMetadata.leaderEpoch().orElse(RecordBatch.NO_PARTITION_LEADER_EPOCH))
                    .setCommittedMetadata(offsetAndMetadata.metadata())
            );
            requestTopicDataMap.put(topicPartition.topic(), topic);
        }

        final Generation generation;
        final String groupInstanceId;
        if (subscriptions.hasAutoAssignedPartitions()) {
            synchronized (ConsumerCoordinator.this) {
                generation = generationIfStable();
                groupInstanceId = rebalanceConfig.groupInstanceId.orElse(null);
                // if the generation is null, we are not part of an active group (and we expect to be).
                // the only thing we can do is fail the commit and let the user rejoin the group in poll().
                if (generation == null) {
                    log.info("Failing OffsetCommit request since the consumer is not part of an active group");

                    if (rebalanceInProgress()) {
                        // if the client knows it is already rebalancing, we can use RebalanceInProgressException instead of
                        // CommitFailedException to indicate this is not a fatal error
                        return RequestFuture.failure(new RebalanceInProgressException("Offset commit cannot be completed since the " +
                            "consumer is undergoing a rebalance for auto partition assignment. You can try completing the rebalance " +
                            "by calling poll() and then retry the operation."));
                    } else {
                        return RequestFuture.failure(new CommitFailedException("Offset commit cannot be completed since the " +
                            "consumer is not part of an active group for auto partition assignment; it is likely that the consumer " +
                            "was kicked out of the group."));
                    }
                }
            }
        } else {
            generation = Generation.NO_GENERATION;
            groupInstanceId = null;
        }

        OffsetCommitRequest.Builder builder = OffsetCommitRequest.Builder.forTopicNames(
                new OffsetCommitRequestData()
                        .setGroupId(this.rebalanceConfig.groupId)
                        .setGenerationIdOrMemberEpoch(generation.generationId)
                        .setMemberId(generation.memberId)
                        .setGroupInstanceId(groupInstanceId)
                        .setTopics(new ArrayList<>(requestTopicDataMap.values()))
        );

        log.trace("Sending OffsetCommit request with {} to coordinator {}", offsets, coordinator);

        return client.send(coordinator, builder)
                .compose(new OffsetCommitResponseHandler(offsets, generation));
    }
```



---

### Функция ID: 201

**Исходный код:**
```python
def _sizeof_fmt(num: float, size_qualifier: str) -> str:
    """
    Return size in human readable format.

    Parameters
    ----------
    num : int
        Size in bytes.
    size_qualifier : str
        Either empty, or '+' (if lower bound).

    Returns
    -------
    str
        Size in human readable format.

    Examples
    --------
    >>> _sizeof_fmt(23028, "")
    '22.5 KB'

    >>> _sizeof_fmt(23028, "+")
    '22.5+ KB'
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f}{size_qualifier} {x}"
        num /= 1024.0
    return f"{num:3.1f}{size_qualifier} PB"
```



---

### Функция ID: 202

**Исходный код:**
```python
def atleast_nd(x: Array, /, *, ndim: int, xp: ModuleType | None = None) -> Array:
    """
    Recursively expand the dimension of an array to at least `ndim`.

    Parameters
    ----------
    x : array
        Input array.
    ndim : int
        The minimum number of dimensions for the result.
    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer.

    Returns
    -------
    array
        An array with ``res.ndim`` >= `ndim`.
        If ``x.ndim`` >= `ndim`, `x` is returned.
        If ``x.ndim`` < `ndim`, `x` is expanded by prepending new axes
        until ``res.ndim`` equals `ndim`.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> x = xp.asarray([1])
    >>> xpx.atleast_nd(x, ndim=3, xp=xp)
    Array([[[1]]], dtype=array_api_strict.int64)

    >>> x = xp.asarray([[[1, 2],
    ...                  [3, 4]]])
    >>> xpx.atleast_nd(x, ndim=1, xp=xp) is x
    True
    """
    if xp is None:
        xp = array_namespace(x)

    if x.ndim < ndim:
        x = xp.expand_dims(x, axis=0)
        x = atleast_nd(x, ndim=ndim, xp=xp)
    return x
```



---

### Функция ID: 203

**Исходный код:**
```java
@SuppressWarnings("IdentifierName")
  public static void close(@Nullable Closeable closeable, boolean swallowIOException)
      throws IOException {
    if (closeable == null) {
      return;
    }
    try {
      closeable.close();
    } catch (IOException e) {
      if (swallowIOException) {
        logger.log(Level.WARNING, "IOException thrown while closing Closeable.", e);
      } else {
        throw e;
      }
    }
  }
```



---

### Функция ID: 204

**Исходный код:**
```java
protected synchronized boolean rejoinNeededOrPending() {
        // if there's a pending joinFuture, we should try to complete handling it.
        return rejoinNeeded || joinFuture != null;
    }
```



---

### Функция ID: 205

**Исходный код:**
```typescript
function getVisualListRange(node: Node, list: TextRange, sourceFile: SourceFile): TextRange {

        const children = node.getChildren(sourceFile);

        for (let i = 1; i < children.length - 1; i++) {

            if (children[i].pos === list.pos && children[i].end === list.end) {

                return { pos: children[i - 1].end, end: children[i + 1].getStart(sourceFile) };

            }

        }

        return list;

    }
```



---

### Функция ID: 206

**Исходный код:**
```python
def to_series(self, index=None, name: Hashable | None = None) -> Series:
        """
        Create a Series with both index and values equal to the index keys.

        Useful with map for returning an indexer based on an index.

        Parameters
        ----------
        index : Index, optional
            Index of resulting Series. If None, defaults to original index.
        name : str, optional
            Name of resulting Series. If None, defaults to name of original
            index.

        Returns
        -------
        Series
            The dtype will be based on the type of the Index values.

        See Also
        --------
        Index.to_frame : Convert an Index to a DataFrame.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = pd.Index(["Ant", "Bear", "Cow"], name="animal")

        By default, the original index and original name is reused.

        >>> idx.to_series()
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
        Name: animal, dtype: object

        To enforce a new index, specify new labels to ``index``:

        >>> idx.to_series(index=[0, 1, 2])
        0     Ant
        1    Bear
        2     Cow
        Name: animal, dtype: object

        To override the name of the resulting column, specify ``name``:

        >>> idx.to_series(name="zoo")
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
        Name: zoo, dtype: object
        """
        from pandas import Series

        if index is None:
            index = self._view()
        if name is None:
            name = self.name

        return Series(self._values.copy(), index=index, name=name)
```



---

### Функция ID: 207

**Исходный код:**
```python
def memory_usage(self, index: bool = True, deep: bool = False) -> int:
        """
        Return the memory usage of the Series.

        The memory usage can optionally include the contribution of
        the index and of elements of `object` dtype.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the Series index.
        deep : bool, default False
            If True, introspect the data deeply by interrogating
            `object` dtypes for system-level memory consumption, and include
            it in the returned value.

        Returns
        -------
        int
            Bytes of memory consumed.

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.
        DataFrame.memory_usage : Bytes consumed by a DataFrame.

        Examples
        --------
        >>> s = pd.Series(range(3))
        >>> s.memory_usage()
        152

        Not including the index gives the size of the rest of the data, which
        is necessarily smaller:

        >>> s.memory_usage(index=False)
        24

        The memory footprint of `object` values is ignored by default:

        >>> s = pd.Series(["a", "b"])
        >>> s.values
        array(['a', 'b'], dtype=object)
        >>> s.memory_usage()
        144
        >>> s.memory_usage(deep=True)
        244
        """
        v = self._memory_usage(deep=deep)
        if index:
            v += self.index.memory_usage(deep=deep)
        return v
```



---

### Функция ID: 208

**Исходный код:**
```python
def infer_objects(self, copy: bool = True) -> Index:
        """
        If we have an object dtype, try to infer a non-object dtype.

        Parameters
        ----------
        copy : bool, default True
            Whether to make a copy in cases where no inference occurs.

        Returns
        -------
        Index
            An Index with a new dtype if the dtype was inferred
            or a shallow copy if the dtype could not be inferred.

        See Also
        --------
        Index.inferred_type: Return a string of the type inferred from the values.

        Examples
        --------
        >>> pd.Index(["a", 1]).infer_objects()
        Index(['a', '1'], dtype='object')
        >>> pd.Index([1, 2], dtype="object").infer_objects()
        Index([1, 2], dtype='int64')
        """
        if self._is_multi:
            raise NotImplementedError(
                "infer_objects is not implemented for MultiIndex. "
                "Use index.to_frame().infer_objects() instead."
            )
        if self.dtype != object:
            return self.copy() if copy else self

        values = self._values
        values = cast("npt.NDArray[np.object_]", values)
        res_values = lib.maybe_convert_objects(
            values,
            convert_non_numeric=True,
        )
        if copy and res_values is values:
            return self.copy()
        result = Index(res_values, name=self.name)
        if not copy and res_values is values and self._references is not None:
            result._references = self._references
            result._references.add_index_reference(result)
        return result
```



---

### Функция ID: 209

**Исходный код:**
```python
def truncate(self, before=None, after=None) -> MultiIndex:
        """
        Slice index between two labels / tuples, return new MultiIndex.

        Parameters
        ----------
        before : label or tuple, can be partial. Default None
            None defaults to start.
        after : label or tuple, can be partial. Default None
            None defaults to end.

        Returns
        -------
        MultiIndex
            The truncated MultiIndex.

        See Also
        --------
        DataFrame.truncate : Truncate a DataFrame before and after some index values.
        Series.truncate : Truncate a Series before and after some index values.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a", "b", "c"], ["x", "y", "z"]])
        >>> mi
        MultiIndex([('a', 'x'), ('b', 'y'), ('c', 'z')],
                   )
        >>> mi.truncate(before="a", after="b")
        MultiIndex([('a', 'x'), ('b', 'y')],
                   )
        """
        if after and before and after < before:
            raise ValueError("after < before")

        i, j = self.levels[0].slice_locs(before, after)
        left, right = self.slice_locs(before, after)

        new_levels = list(self.levels)
        new_levels[0] = new_levels[0][i:j]

        new_codes = [level_codes[left:right] for level_codes in self.codes]
        new_codes[0] = new_codes[0] - i

        return MultiIndex(
            levels=new_levels,
            codes=new_codes,
            names=self._names,
            verify_integrity=False,
        )
```



---

### Функция ID: 210

**Исходный код:**
```python
def is_file_like(obj: object) -> bool:
    """
    Check if the object is a file-like object.

    For objects to be considered file-like, they must
    be an iterator AND have either a `read` and/or `write`
    method as an attribute.

    Note: file-like objects must be iterable, but
    iterable objects need not be file-like.

    Parameters
    ----------
    obj : object
        The object to check for file-like properties.
        This can be any Python object, and the function will
        check if it has attributes typically associated with
        file-like objects (e.g., `read`, `write`, `__iter__`).

    Returns
    -------
    bool
        Whether `obj` has file-like properties.

    See Also
    --------
    api.types.is_dict_like : Check if the object is dict-like.
    api.types.is_hashable : Return True if hash(obj) will succeed, False otherwise.
    api.types.is_named_tuple : Check if the object is a named tuple.
    api.types.is_iterator : Check if the object is an iterator.

    Examples
    --------
    >>> import io
    >>> from pandas.api.types import is_file_like
    >>> buffer = io.StringIO("data")
    >>> is_file_like(buffer)
    True
    >>> is_file_like([1, 2, 3])
    False
    """
    if not (hasattr(obj, "read") or hasattr(obj, "write")):
        return False

    return bool(hasattr(obj, "__iter__"))
```



---

### Функция ID: 211

**Исходный код:**
```python
def get_data_home(data_home=None) -> str:
    """Return the path of the scikit-learn data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data directory is set to a folder named 'scikit_learn_data' in the
    user home folder.

    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The path to scikit-learn data directory. If `None`, the default path
        is `~/scikit_learn_data`.

    Returns
    -------
    data_home: str
        The path to scikit-learn data directory.

    Examples
    --------
    >>> import os
    >>> from sklearn.datasets import get_data_home
    >>> data_home_path = get_data_home()
    >>> os.path.exists(data_home_path)
    True
    """
    if data_home is None:
        data_home = environ.get("SCIKIT_LEARN_DATA", join("~", "scikit_learn_data"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home
```



---

### Функция ID: 212

**Исходный код:**
```python
def reset_option(pat: str) -> None:
    """
    Reset one or more options to their default value.

    This method resets the specified pandas option(s) back to their default
    values. It allows partial string matching for convenience, but users should
    exercise caution to avoid unintended resets due to changes in option names
    in future versions.

    Parameters
    ----------
    pat : str/regex
        If specified only options matching ``pat*`` will be reset.
        Pass ``"all"`` as argument to reset all options.

        .. warning::

            Partial matches are supported for convenience, but unless you
            use the full option name (e.g. x.y.z.option_name), your code may break
            in future versions if new options with similar names are introduced.

    Returns
    -------
    None
        No return value.

    See Also
    --------
    get_option : Retrieve the value of the specified option.
    set_option : Set the value of the specified option or options.
    describe_option : Print the description for one or more registered options.

    Notes
    -----
    For all available options, please view the
    :ref:`User Guide <options.available>`.

    Examples
    --------
    >>> pd.reset_option("display.max_columns")  # doctest: +SKIP
    """
    keys = _select_options(pat)

    if len(keys) == 0:
        raise OptionError(f"No such keys(s) for {pat=}")

    if len(keys) > 1 and len(pat) < 4 and pat != "all":
        raise ValueError(
            "You must specify at least 4 characters when "
            "resetting multiple keys, use the special keyword "
            '"all" to reset all the options to their default value'
        )

    for k in keys:
        set_option(k, _registered_options[k].defval)
```



---

### Функция ID: 213

**Исходный код:**
```python
def from_product(
        cls,
        iterables: Sequence[Iterable[Hashable]],
        sortorder: int | None = None,
        names: Sequence[Hashable] | Hashable | lib.NoDefault = lib.no_default,
    ) -> MultiIndex:
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.
            If not explicitly provided, names will be inferred from the
            elements of iterables if an element has a name attribute.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ["green", "purple"]
        >>> pd.MultiIndex.from_product([numbers, colors], names=["number", "color"])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """

        if not is_list_like(iterables):
            raise TypeError("Input must be a list / sequence of iterables.")
        if is_iterator(iterables):
            iterables = list(iterables)

        codes, levels = factorize_from_iterables(iterables)
        if names is lib.no_default:
            names = [getattr(it, "name", None) for it in iterables]

        # codes are all ndarrays, so cartesian_product is lossless
        codes = cartesian_product(codes)
        return cls(levels, codes, sortorder=sortorder, names=names)
```



---

### Функция ID: 214

**Исходный код:**
```python
def hasnans(self) -> bool:
        """
        Return True if there are any NaNs.

        Enables various performance speedups.

        Returns
        -------
        bool

        See Also
        --------
        Index.isna : Detect missing values.
        Index.dropna : Return Index without NA/NaN values.
        Index.fillna : Fill NA/NaN values with the specified value.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3], index=["a", "b", None])
        >>> s
        a    1
        b    2
        None 3
        dtype: int64
        >>> s.index.hasnans
        True
        """
        if self._can_hold_na:
            return bool(self._isnan.any())
        else:
            return False
```



---

### Функция ID: 215

**Исходный код:**
```python
def is_re_compilable(obj: object) -> bool:
    """
    Check if the object can be compiled into a regex pattern instance.

    Parameters
    ----------
    obj : The object to check
        The object to check if the object can be compiled into a regex pattern instance.

    Returns
    -------
    bool
        Whether `obj` can be compiled as a regex pattern.

    See Also
    --------
    api.types.is_re : Check if the object is a regex pattern instance.

    Examples
    --------
    >>> from pandas.api.types import is_re_compilable
    >>> is_re_compilable(".*")
    True
    >>> is_re_compilable(1)
    False
    """
    try:
        re.compile(obj)  # type: ignore[call-overload]
    except TypeError:
        return False
    else:
        return True
```



---

### Функция ID: 216

**Исходный код:**
```python
def has_wrong_whitespace(first_line: str, second_line: str) -> bool:
        """
        Checking if the two lines are mattching the unwanted pattern.

        Parameters
        ----------
        first_line : str
            First line to check.
        second_line : str
            Second line to check.

        Returns
        -------
        bool
            True if the two received string match, an unwanted pattern.

        Notes
        -----
        The unwanted pattern that we are trying to catch is if the spaces in
        a string that is concatenated over multiple lines are placed at the
        end of each string, unless this string is ending with a
        newline character (\n).

        For example, this is bad:

        >>> rule = "We want the space at the end of the line, not at the beginning"

        And what we want is:

        >>> rule = "We want the space at the end of the line, not at the beginning"

        And if the string is ending with a new line character (\n) we
        do not want any trailing whitespaces after it.

        For example, this is bad:

        >>> rule = (
        ...     "We want the space at the begging of "
        ...     "the line if the previous line is ending with a \n "
        ...     "not at the end, like always"
        ... )

        And what we do want is:

        >>> rule = (
        ...     "We want the space at the begging of "
        ...     "the line if the previous line is ending with a \n"
        ...     " not at the end, like always"
        ... )
        """
        if first_line.endswith(r"\n"):
            return False
        elif first_line.startswith("  ") or second_line.startswith("  "):
            return False
        elif first_line.endswith("  ") or second_line.endswith("  "):
            return False
        elif (not first_line.endswith(" ")) and second_line.startswith(" "):
            return True
        return False
```



---

### Функция ID: 217

**Исходный код:**
```python
def get_slice_bound(
        self,
        label: Hashable | Sequence[Hashable],
        side: Literal["left", "right"],
    ) -> int:
        """
        For an ordered MultiIndex, compute slice bound
        that corresponds to given label.

        Returns leftmost (one-past-the-rightmost if `side=='right') position
        of given label.

        Parameters
        ----------
        label : object or tuple of objects
        side : {'left', 'right'}

        Returns
        -------
        int
            Index of label.

        Notes
        -----
        This method only works if level 0 index of the MultiIndex is lexsorted.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list("abbc"), list("gefd")])

        Get the locations from the leftmost 'b' in the first level
        until the end of the multiindex:

        >>> mi.get_slice_bound("b", side="left")
        1

        Like above, but if you get the locations from the rightmost
        'b' in the first level and 'f' in the second level:

        >>> mi.get_slice_bound(("b", "f"), side="right")
        3

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        if not isinstance(label, tuple):
            label = (label,)
        return self._partial_tup_index(label, side=side)
```



---

### Функция ID: 218

**Исходный код:**
```python
def items(self) -> Iterable[tuple[Hashable, Series]]:
        r"""
        Iterate over (column name, Series) pairs.

        Iterates over the DataFrame columns, returning a tuple with
        the column name and the content as a Series.

        Yields
        ------
        label : object
            The column names for the DataFrame being iterated over.
        content : Series
            The column entries belonging to each label, as a Series.

        See Also
        --------
        DataFrame.iterrows : Iterate over DataFrame rows as
            (index, Series) pairs.
        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples
            of the values.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "species": ["bear", "bear", "marsupial"],
        ...         "population": [1864, 22000, 80000],
        ...     },
        ...     index=["panda", "polar", "koala"],
        ... )
        >>> df
                species   population
        panda   bear      1864
        polar   bear      22000
        koala   marsupial 80000
        >>> for label, content in df.items():
        ...     print(f"label: {label}")
        ...     print(f"content: {content}", sep="\n")
        label: species
        content:
        panda         bear
        polar         bear
        koala    marsupial
        Name: species, dtype: object
        label: population
        content:
        panda     1864
        polar    22000
        koala    80000
        Name: population, dtype: int64
        """
        for i, k in enumerate(self.columns):
            yield k, self._ixs(i, axis=1)
```



---

### Функция ID: 219

**Исходный код:**
```python
def get_series_repr_params() -> dict[str, Any]:
    """Get the parameters used to repr(Series) calls using Series.to_string.

    Supplying these parameters to Series.to_string is equivalent to calling
    ``repr(series)``. This is useful if you want to adjust the series repr output.

    Example
    -------
    >>> import pandas as pd
    >>>
    >>> ser = pd.Series([1, 2, 3, 4])
    >>> repr_params = pd.io.formats.format.get_series_repr_params()
    >>> repr(ser) == ser.to_string(**repr_params)
    True
    """
    width, height = get_terminal_size()
    max_rows_opt = get_option("display.max_rows")
    max_rows = height if max_rows_opt == 0 else max_rows_opt
    min_rows = height if max_rows_opt == 0 else get_option("display.min_rows")

    return {
        "name": True,
        "dtype": True,
        "min_rows": min_rows,
        "max_rows": max_rows,
        "length": get_option("display.show_dimensions"),
    }
```



---

### Функция ID: 220

**Исходный код:**
```python
def keys(self, include: str = "pandas") -> list[str]:
        """
        Return a list of keys corresponding to objects stored in HDFStore.

        Parameters
        ----------

        include : str, default 'pandas'
                When kind equals 'pandas' return pandas objects.
                When kind equals 'native' return native HDF5 Table objects.

        Returns
        -------
        list
            List of ABSOLUTE path-names (e.g. have the leading '/').

        Raises
        ------
        raises ValueError if kind has an illegal value

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.
        HDFStore.get_node : Returns the node with the key.
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        ['/data1', '/data2']
        >>> store.close()  # doctest: +SKIP
        """
        if include == "pandas":
            return [n._v_pathname for n in self.groups()]

        elif include == "native":
            assert self._handle is not None  # mypy
            return [
                n._v_pathname for n in self._handle.walk_nodes("/", classname="Table")
            ]
        raise ValueError(
            f"`include` should be either 'pandas' or 'native' but is '{include}'"
        )
```



---

### Функция ID: 221

**Исходный код:**
```python
def value_labels(self) -> dict[str, dict[int, str]]:
        """
        Return a nested dict associating each variable name to its value and label.

        This method retrieves the value labels from a Stata file. Value labels are
        mappings between the coded values and their corresponding descriptive labels
        in a Stata dataset.

        Returns
        -------
        dict
            A python dictionary.

        See Also
        --------
        read_stata : Read Stata file into DataFrame.
        DataFrame.to_stata : Export DataFrame object to Stata dta format.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> path = "/My_path/filename.dta"
        >>> value_labels = {"col_1": {3: "x"}}
        >>> df.to_stata(
        ...     path,
        ...     time_stamp=time_stamp,  # doctest: +SKIP
        ...     value_labels=value_labels,
        ...     version=None,
        ... )  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.value_labels())  # doctest: +SKIP
        {'col_1': {3: 'x'}}
        >>> pd.read_stata(path)  # doctest: +SKIP
            index col_1 col_2
        0       0    1    2
        1       1    x    4
        """
        if not self._value_labels_read:
            self._read_value_labels()

        return self._value_label_dict
```



---

### Функция ID: 222

**Исходный код:**
```python
def python_type(self) -> type:
        """
        Abstract method to be implemented by subclasses of VariableTracker.

        This method should return the type represented by the instance of the subclass.
        The purpose is to provide a standardized way to retrieve the Python type information
        of the variable being tracked.

        Returns:
            type: The Python type (such as int, str, list, etc.) of the variable tracked by
                the subclass. If the type cannot be determined or is not relevant,
                leaving it undefined or invoking super() is always sound.

        Note:
            This is an abstract method and may be overridden in subclasses.

        Example:
            class SetVariable(VariableTracker):
                def python_type(self):
                    return set

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        try:
            return type(self.as_python_constant())
        except NotImplementedError:
            raise NotImplementedError(f"{self} has no type") from None
```



---

### Функция ID: 223

**Исходный код:**
```java
private long sendEligibleCalls(long now) {
            long pollTimeout = Long.MAX_VALUE;
            for (Iterator<Map.Entry<Node, List<Call>>> iter = callsToSend.entrySet().iterator(); iter.hasNext(); ) {
                Map.Entry<Node, List<Call>> entry = iter.next();
                List<Call> calls = entry.getValue();
                if (calls.isEmpty()) {
                    iter.remove();
                    continue;
                }
                Node node = entry.getKey();
                if (callsInFlight.containsKey(node.idString())) {
                    log.trace("Still waiting for other calls to finish on node {}.", node);
                    nodeReadyDeadlines.remove(node);
                    continue;
                }
                if (!client.ready(node, now)) {
                    Long deadline = nodeReadyDeadlines.get(node);
                    if (deadline != null) {
                        if (now >= deadline) {
                            log.info("Disconnecting from {} and revoking {} node assignment(s) " +
                                    "because the node is taking too long to become ready.",
                                node.idString(), calls.size());
                            transitionToPendingAndClearList(calls);
                            client.disconnect(node.idString());
                            nodeReadyDeadlines.remove(node);
                            iter.remove();
                            continue;
                        }
                        pollTimeout = Math.min(pollTimeout, deadline - now);
                    } else {
                        nodeReadyDeadlines.put(node, now + requestTimeoutMs);
                    }
                    long nodeTimeout = client.pollDelayMs(node, now);
                    pollTimeout = Math.min(pollTimeout, nodeTimeout);
                    log.trace("Client is not ready to send to {}. Must delay {} ms", node, nodeTimeout);
                    continue;
                }
                // Subtract the time we spent waiting for the node to become ready from
                // the total request time.
                int remainingRequestTime;
                Long deadlineMs = nodeReadyDeadlines.remove(node);
                if (deadlineMs == null) {
                    remainingRequestTime = requestTimeoutMs;
                } else {
                    remainingRequestTime = calcTimeoutMsRemainingAsInt(now, deadlineMs);
                }
                while (!calls.isEmpty()) {
                    Call call = calls.remove(0);
                    int timeoutMs = Math.min(remainingRequestTime,
                        calcTimeoutMsRemainingAsInt(now, call.deadlineMs));
                    AbstractRequest.Builder<?> requestBuilder;
                    try {
                        requestBuilder = call.createRequest(timeoutMs);
                    } catch (Throwable t) {
                        call.fail(now, new KafkaException(String.format(
                            "Internal error sending %s to %s.", call.callName, node), t));
                        continue;
                    }
                    ClientRequest clientRequest = client.newClientRequest(node.idString(),
                        requestBuilder, now, true, timeoutMs, null);
                    log.debug("Sending {} to {}. correlationId={}, timeoutMs={}",
                        requestBuilder, node, clientRequest.correlationId(), timeoutMs);
                    client.send(clientRequest, now);
                    callsInFlight.put(node.idString(), call);
                    correlationIdToCalls.put(clientRequest.correlationId(), call);
                    break;
                }
            }
            return pollTimeout;
        }
```



---

### Функция ID: 224

**Исходный код:**
```java
static double fastAsin(double x) {
        if (x < 0) {
            return -fastAsin(-x);
        } else if (x > 1) {
            return Double.NaN;
        } else {
            // Cutoffs for models. Note that the ranges overlap. In the
            // overlap we do linear interpolation to guarantee the overall
            // result is "nice"
            double c0High = 0.1;
            double c1High = 0.55;
            double c2Low = 0.5;
            double c2High = 0.8;
            double c3Low = 0.75;
            double c3High = 0.9;
            double c4Low = 0.87;
            if (x > c3High) {
                return Math.asin(x);
            } else {
                // the models
                double[] m0 = { 0.2955302411, 1.2221903614, 0.1488583743, 0.2422015816, -0.3688700895, 0.0733398445 };
                double[] m1 = { -0.0430991920, 0.9594035750, -0.0362312299, 0.1204623351, 0.0457029620, -0.0026025285 };
                double[] m2 = { -0.034873933724, 1.054796752703, -0.194127063385, 0.283963735636, 0.023800124916, -0.000872727381 };
                double[] m3 = { -0.37588391875, 2.61991859025, -2.48835406886, 1.48605387425, 0.00857627492, -0.00015802871 };

                // the parameters for all of the models
                double[] vars = { 1, x, x * x, x * x * x, 1 / (1 - x), 1 / (1 - x) / (1 - x) };

                // raw grist for interpolation coefficients
                double x0 = bound((c0High - x) / c0High);
                double x1 = bound((c1High - x) / (c1High - c2Low));
                double x2 = bound((c2High - x) / (c2High - c3Low));
                double x3 = bound((c3High - x) / (c3High - c4Low));

                // interpolation coefficients
                // noinspection UnnecessaryLocalVariable
                double mix0 = x0;
                double mix1 = (1 - x0) * x1;
                double mix2 = (1 - x1) * x2;
                double mix3 = (1 - x2) * x3;
                double mix4 = 1 - x3;

                // now mix all the results together, avoiding extra evaluations
                double r = 0;
                if (mix0 > 0) {
                    r += mix0 * eval(m0, vars);
                }
                if (mix1 > 0) {
                    r += mix1 * eval(m1, vars);
                }
                if (mix2 > 0) {
                    r += mix2 * eval(m2, vars);
                }
                if (mix3 > 0) {
                    r += mix3 * eval(m3, vars);
                }
                if (mix4 > 0) {
                    // model 4 is just the real deal
                    r += mix4 * Math.asin(x);
                }
                return r;
            }
        }
    }
```



---

### Функция ID: 225

**Исходный код:**
```python
def simplify_index_in_vec_range(index: sympy.Expr, var: sympy.Expr, vec_length: int):
    """
    Simplifies the index expression within the range of a vectorized loop.
    Given a vectorized loop variable `var` in the range of a loop with `vec_length`,
    this function transforms the `index` into an equivalent form. It handles
    simplifications for cases where `var` can be expressed as `vec_length * a + b`,
    where `b` ranges from 0 to `vec_length - 1`. The function reduces occurrences
    of `FloorDiv` and `ModularIndexing` in the `index` with best-effort optimizations.

    NOTE:
    The simplified index expression is intended for analysis purposes only, not
    for code generation. It replaces `FloorDiv` and `ModularIndexing` with free variables
    which are not dependent on the loop variable `var` in the vectorized range. Check
    https://github.com/pytorch/pytorch/pull/117221#discussion_r1449746217 for more details.

    Examples:
    1. If `var` is `x3` and `vec_length` is 16, and `x3 = 16*a + b`, then
       `FloorDiv(x3, div)` or `ModularIndexing(x3, div, mod)` becomes a free variable
       when `div` is divisible by 16.
    2. `ModularIndexing(x3, 1, mod)` can be simplified to `x3 + c` where `c` is a free
       variable when `mod` is divisible by 16.
    """

    div_freevar_id = 0
    mod_freevar_id = 0

    def visit_indexing_div(divisor):
        nonlocal div_freevar_id
        result = FloorDiv(var, divisor)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_div_c{div_freevar_id}")
            div_freevar_id += 1
        return result

    def visit_modular_indexing(divisor, modulus):
        nonlocal mod_freevar_id
        result = ModularIndexing(var, divisor, modulus)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        elif divisor == 1 and sympy.gcd(modulus, vec_length) == vec_length:
            result = var + sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        return result

    original_index = index

    div = sympy.Wild("divisor", integer=True)
    if index.has(FloorDiv):
        index = index.replace(FloorDiv(var, div), visit_indexing_div)

    mod = sympy.Wild("modulus", integer=True)
    if index.has(ModularIndexing):
        index = index.replace(ModularIndexing(var, div, mod), visit_modular_indexing)

    index = sympy.simplify(index)
    if index != original_index:
        return simplify_index_in_vec_range(index, var, vec_length)

    return index
```



---

### Функция ID: 226

**Исходный код:**
```typescript
function getListByRange(start: number, end: number, node: Node, sourceFile: SourceFile): NodeArray<Node> | undefined {

        switch (node.kind) {

            case SyntaxKind.TypeReference:

                return getList((node as TypeReferenceNode).typeArguments);

            case SyntaxKind.ObjectLiteralExpression:

                return getList((node as ObjectLiteralExpression).properties);

            case SyntaxKind.ArrayLiteralExpression:

                return getList((node as ArrayLiteralExpression).elements);

            case SyntaxKind.TypeLiteral:

                return getList((node as TypeLiteralNode).members);

            case SyntaxKind.FunctionDeclaration:

            case SyntaxKind.FunctionExpression:

            case SyntaxKind.ArrowFunction:

            case SyntaxKind.MethodDeclaration:

            case SyntaxKind.MethodSignature:

            case SyntaxKind.CallSignature:

            case SyntaxKind.Constructor:

            case SyntaxKind.ConstructorType:

            case SyntaxKind.ConstructSignature:

                return getList((node as SignatureDeclaration).typeParameters) || getList((node as SignatureDeclaration).parameters);

            case SyntaxKind.GetAccessor:

                return getList((node as GetAccessorDeclaration).parameters);

            case SyntaxKind.ClassDeclaration:

            case SyntaxKind.ClassExpression:

            case SyntaxKind.InterfaceDeclaration:

            case SyntaxKind.TypeAliasDeclaration:

            case SyntaxKind.JSDocTemplateTag:

                return getList((node as ClassDeclaration | ClassExpression | InterfaceDeclaration | TypeAliasDeclaration | JSDocTemplateTag).typeParameters);

            case SyntaxKind.NewExpression:

            case SyntaxKind.CallExpression:

                return getList((node as CallExpression).typeArguments) || getList((node as CallExpression).arguments);

            case SyntaxKind.VariableDeclarationList:

                return getList((node as VariableDeclarationList).declarations);

            case SyntaxKind.NamedImports:

            case SyntaxKind.NamedExports:

                return getList((node as NamedImportsOrExports).elements);

            case SyntaxKind.ObjectBindingPattern:

            case SyntaxKind.ArrayBindingPattern:

                return getList((node as ObjectBindingPattern | ArrayBindingPattern).elements);

        }



        function getList(list: NodeArray<Node> | undefined): NodeArray<Node> | undefined {

            return list && rangeContainsStartEnd(getVisualListRange(node, list, sourceFile), start, end) ? list : undefined;

        }

    }
```



---

### Функция ID: 227

**Исходный код:**
```java
@Override
    public double quantile(double q) {
        if (q < 0 || q > 1) {
            throw new IllegalArgumentException("q should be in [0,1], got " + q);
        }

        AVLGroupTree values = summary;
        if (values.isEmpty()) {
            // no centroids means no data, no way to get a quantile
            return Double.NaN;
        } else if (values.size() == 1) {
            // with one data point, all quantiles lead to Rome
            return values.iterator().next().mean();
        }

        // if values were stored in a sorted array, index would be the offset we are interested in
        final double index = q * count;

        // deal with min and max as a special case singletons
        if (index <= 0) {
            return min;
        }

        if (index >= count) {
            return max;
        }

        int currentNode = values.first();
        long currentWeight = values.count(currentNode);

        // Total mass to the left of the center of the current node.
        double weightSoFar = currentWeight / 2.0;

        if (index <= weightSoFar && weightSoFar > 1) {
            // Interpolate between min and first mean, if there's no singleton on the left boundary.
            return weightedAverage(min, weightSoFar - index, values.mean(currentNode), index);
        }

        for (int i = 0; i < values.size() - 1; i++) {
            int nextNode = values.next(currentNode);
            long nextWeight = values.count(nextNode);
            // this is the mass between current center and next center
            double dw = (currentWeight + nextWeight) / 2.0;

            if (index < weightSoFar + dw) {
                // index is bracketed between centroids i and i+1
                assert dw >= 1;

                double w1 = index - weightSoFar;
                double w2 = weightSoFar + dw - index;
                return weightedAverage(values.mean(currentNode), w2, values.mean(nextNode), w1);
            }
            weightSoFar += dw;
            currentNode = nextNode;
            currentWeight = nextWeight;
        }

        // Index is close or after the last centroid.
        assert currentWeight >= 1;
        assert index - weightSoFar < count - currentWeight / 2.0;
        assert count - weightSoFar >= 0.5;

        // Interpolate between the last mean and the max.
        double w1 = index - weightSoFar;
        double w2 = currentWeight / 2.0 - w1;
        return weightedAverage(values.mean(currentNode), w2, max, w1);
    }
```



---

### Функция ID: 228

**Исходный код:**
```java
private @Nullable Collection<CacheOperation> computeCacheOperations(Method method, @Nullable Class<?> targetClass) {
		// Don't allow non-public methods, as configured.
		if (allowPublicMethodsOnly() && !Modifier.isPublic(method.getModifiers())) {
			return null;
		}
		// Skip setBeanFactory method on BeanFactoryAware.
		if (method.getDeclaringClass() == BeanFactoryAware.class) {
			return null;
		}

		// The method may be on an interface, but we need metadata from the target class.
		// If the target class is null, the method will be unchanged.
		Method specificMethod = AopUtils.getMostSpecificMethod(method, targetClass);

		// First try is the method in the target class.
		Collection<CacheOperation> opDef = findCacheOperations(specificMethod);
		if (opDef != null) {
			return opDef;
		}

		// Second try is the caching operation on the target class.
		opDef = findCacheOperations(specificMethod.getDeclaringClass());
		if (opDef != null && ClassUtils.isUserLevelMethod(method)) {
			return opDef;
		}

		if (specificMethod != method) {
			// Fallback is to look at the original method.
			opDef = findCacheOperations(method);
			if (opDef != null) {
				return opDef;
			}
			// Last fallback is the class of the original method.
			opDef = findCacheOperations(method.getDeclaringClass());
			if (opDef != null && ClassUtils.isUserLevelMethod(method)) {
				return opDef;
			}
		}

		return null;
	}
```



---

### Функция ID: 229

**Исходный код:**
```java
public boolean isParentOf(ConfigurationPropertyName name) {
		Assert.notNull(name, "'name' must not be null");
		if (getNumberOfElements() != name.getNumberOfElements() - 1) {
			return false;
		}
		return isAncestorOf(name);
	}
```



---

### Функция ID: 230

**Исходный код:**
```java
private void completeFutureAndFireCallbacks(
        long baseOffset,
        long logAppendTime,
        Function<Integer, RuntimeException> recordExceptions
    ) {
        // Set the future before invoking the callbacks as we rely on its state for the `onCompletion` call
        produceFuture.set(baseOffset, logAppendTime, recordExceptions);

        // execute callbacks
        for (int i = 0; i < thunks.size(); i++) {
            try {
                Thunk thunk = thunks.get(i);
                if (thunk.callback != null) {
                    if (recordExceptions == null) {
                        RecordMetadata metadata = thunk.future.value();
                        thunk.callback.onCompletion(metadata, null);
                    } else {
                        RuntimeException exception = recordExceptions.apply(i);
                        thunk.callback.onCompletion(null, exception);
                    }
                }
            } catch (Exception e) {
                log.error("Error executing user-provided callback on message for topic-partition '{}'", topicPartition, e);
            }
        }

        produceFuture.done();
    }
```



---

### Функция ID: 231

**Исходный код:**
```java
public static <E extends Throwable> short getAsShort(final FailableShortSupplier<E> supplier) {
        try {
            return supplier.getAsShort();
        } catch (final Throwable t) {
            throw rethrow(t);
        }
    }
```



---

### Функция ID: 232

**Исходный код:**
```java
private Format getFormat(final String desc) {
        if (registry != null) {
            String name = desc;
            String args = null;
            final int i = desc.indexOf(START_FMT);
            if (i > 0) {
                name = desc.substring(0, i).trim();
                args = desc.substring(i + 1).trim();
            }
            final FormatFactory factory = registry.get(name);
            if (factory != null) {
                return factory.getFormat(name, args, getLocale());
            }
        }
        return null;
    }
```



---

### Функция ID: 233

**Исходный код:**
```java
private static String read(Path path) {
        try {
            return Files.readString(path);
        } catch (IOException e) {
            log.error("Could not read file {} for property {}", path, path.getFileName(), e);
            throw new ConfigException("Could not read file " + path + " for property " + path.getFileName());
        }
    }
```



---

### Функция ID: 234

**Исходный код:**
```typescript
function getDocumentationComment(declarations: readonly Declaration[] | undefined, checker: TypeChecker | undefined): SymbolDisplayPart[] {

    if (!declarations) return emptyArray;



    let doc = JsDoc.getJsDocCommentsFromDeclarations(declarations, checker);

    if (checker && (doc.length === 0 || declarations.some(hasJSDocInheritDocTag))) {

        const seenSymbols = new Set<Symbol>();

        for (const declaration of declarations) {

            const inheritedDocs = findBaseOfDeclaration(checker, declaration, symbol => {

                if (!seenSymbols.has(symbol)) {

                    seenSymbols.add(symbol);

                    if (declaration.kind === SyntaxKind.GetAccessor || declaration.kind === SyntaxKind.SetAccessor) {

                        return symbol.getContextualDocumentationComment(declaration, checker);

                    }

                    return symbol.getDocumentationComment(checker);

                }

            });

            // TODO: GH#16312 Return a ReadonlyArray, avoid copying inheritedDocs

            if (inheritedDocs) doc = doc.length === 0 ? inheritedDocs.slice() : inheritedDocs.concat(lineBreakPart(), doc);

        }

    }

    return doc;

}
```



---

### Функция ID: 235

**Исходный код:**
```java
public static String convertToString(Object parsedValue, Type type) {
        if (parsedValue == null) {
            return null;
        }

        if (type == null) {
            return parsedValue.toString();
        }

        switch (type) {
            case BOOLEAN:
            case SHORT:
            case INT:
            case LONG:
            case DOUBLE:
            case STRING:
            case PASSWORD:
                return parsedValue.toString();
            case LIST:
                List<?> valueList = (List<?>) parsedValue;
                return valueList.stream().map(Object::toString).collect(Collectors.joining(","));
            case CLASS:
                Class<?> clazz = (Class<?>) parsedValue;
                return clazz.getName();
            default:
                throw new IllegalStateException("Unknown type.");
        }
    }
```



---

### Функция ID: 236

**Исходный код:**
```java
public StrBuilder deleteAll(final String str) {
        final int len = StringUtils.length(str);
        if (len > 0) {
            int index = indexOf(str, 0);
            while (index >= 0) {
                deleteImpl(index, index + len, len);
                index = indexOf(str, index);
            }
        }
        return this;
    }
```



---

### Функция ID: 237

**Исходный код:**
```python
def copy(  # type: ignore[override]
        self,
        names=None,
        deep: bool = False,
        name=None,
    ) -> Self:
        """
        Make a copy of this object. Names, dtype, levels and codes can be passed and \
        will be set on new copy.

        The `copy` method provides a mechanism to create a duplicate of an
        existing MultiIndex object. This is particularly useful in scenarios where
        modifications are required on an index, but the original MultiIndex should
        remain unchanged. By specifying the `deep` parameter, users can control
        whether the copy should be a deep or shallow copy, providing flexibility
        depending on the size and complexity of the MultiIndex.

        Parameters
        ----------
        names : sequence, optional
            Names to set on the new MultiIndex object.
        deep : bool, default False
            If False, the new object will be a shallow copy. If True, a deep copy
            will be attempted. Deep copying can be potentially expensive for large
            MultiIndex objects.
        name : Label
            Kept for compatibility with 1-dimensional Index. Should not be used.

        Returns
        -------
        MultiIndex
            A new MultiIndex object with the specified modifications.

        See Also
        --------
        MultiIndex.from_arrays : Convert arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_frame : Convert DataFrame to MultiIndex.

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.
        This could be potentially expensive on large MultiIndex objects.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a"], ["b"], ["c"]])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.copy()
        MultiIndex([('a', 'b', 'c')],
                   )
        """
        names = self._validate_names(name=name, names=names, deep=deep)
        keep_id = not deep
        levels, codes = None, None

        if deep:
            from copy import deepcopy

            levels = deepcopy(self.levels)
            codes = deepcopy(self.codes)

        levels = levels if levels is not None else self.levels
        codes = codes if codes is not None else self.codes

        new_index = type(self)(
            levels=levels,
            codes=codes,
            sortorder=self.sortorder,
            names=names,
            verify_integrity=False,
        )
        new_index._cache = self._cache.copy()
        new_index._reset_cache("levels")  # GH32669
        if keep_id:
            new_index._id = self._id
        return new_index
```



---

### Функция ID: 238

**Исходный код:**
```java
@Override
	public String toString() {
		StringBuilder result = new StringBuilder();
		result.append(getResourceDescription(this.resource));
		if (this.location != null) {
			result.append(" - ").append(this.location);
		}
		return result.toString();
	}
```



---

### Функция ID: 239

**Исходный код:**
```java
public static Integer createInteger(final String str) {
        if (str == null) {
            return null;
        }
        // decode() handles 0xAABD and 0777 (hex and octal) as well.
        return Integer.decode(str);
    }
```



---

### Функция ID: 240

**Исходный код:**
```java
private static void readCertificates(String text, CertificateFactory factory, Consumer<X509Certificate> consumer) {
		try {
			Matcher matcher = PATTERN.matcher(text);
			while (matcher.find()) {
				String encodedText = matcher.group(1);
				byte[] decodedBytes = decodeBase64(encodedText);
				ByteArrayInputStream inputStream = new ByteArrayInputStream(decodedBytes);
				while (inputStream.available() > 0) {
					consumer.accept((X509Certificate) factory.generateCertificate(inputStream));
				}
			}
		}
		catch (CertificateException ex) {
			throw new IllegalStateException("Error reading certificate: " + ex.getMessage(), ex);
		}
	}
```



---

### Функция ID: 241

**Исходный код:**
```java
private static double bound(double v) {
        if (v <= 0) {
            return 0;
        } else if (v >= 1) {
            return 1;
        } else {
            return v;
        }
    }
```



---

### Функция ID: 242

**Исходный код:**
```java
@J2ktIncompatible
  @CanIgnoreReturnValue
  public static long copy(ReadableByteChannel from, WritableByteChannel to) throws IOException {
    checkNotNull(from);
    checkNotNull(to);
    if (from instanceof FileChannel) {
      FileChannel sourceChannel = (FileChannel) from;
      long oldPosition = sourceChannel.position();
      long position = oldPosition;
      long copied;
      do {
        copied = sourceChannel.transferTo(position, ZERO_COPY_CHUNK_SIZE, to);
        position += copied;
        sourceChannel.position(position);
      } while (copied > 0 || position < sourceChannel.size());
      return position - oldPosition;
    }

    ByteBuffer buf = ByteBuffer.wrap(createBuffer());
    long total = 0;
    while (from.read(buf) != -1) {
      Java8Compatibility.flip(buf);
      while (buf.hasRemaining()) {
        total += to.write(buf);
      }
      Java8Compatibility.clear(buf);
    }
    return total;
  }
```



---

### Функция ID: 243

**Исходный код:**
```python
def render_dag_dependencies(deps: dict[str, list[DagDependency]]) -> graphviz.Digraph:
    """
    Render the DAG dependency to the DOT object.

    :param deps: List of DAG dependencies
    :return: Graphviz object
    """
    if not graphviz:
        raise AirflowException(
            "Could not import graphviz. Install the graphviz python package to fix this error."
        )
    dot = graphviz.Digraph(graph_attr={"rankdir": "LR"})

    for dag, dependencies in deps.items():
        for dep in dependencies:
            with dot.subgraph(
                name=dag,
                graph_attr={
                    "rankdir": "LR",
                    "labelloc": "t",
                    "label": dag,
                },
            ) as dep_subgraph:
                leaf_nodes = ("asset", "asset-name-ref", "asset-uri-ref", "asset-alias")
                if dep.source not in leaf_nodes:
                    dep_subgraph.edge(dep.source, dep.dependency_id)

                if dep.target not in leaf_nodes:
                    dep_subgraph.edge(dep.dependency_id, dep.target)

    return dot
```



---

### Функция ID: 244

**Исходный код:**
```java
public static double sinh(double value) {
        // sinh(x) = (exp(x)-exp(-x))/2
        double h;
        if (value < 0.0) {
            value = -value;
            h = -0.5;
        } else {
            h = 0.5;
        }
        if (value < 22.0) {
            if (value < TWO_POW_N28) {
                return (h < 0.0) ? -value : value;
            } else {
                double t = Math.expm1(value);
                // Might be more accurate, if value < 1: return h*((t+t)-t*t/(t+1.0)).
                return h * (t + t / (t + 1.0));
            }
        } else if (value < LOG_DOUBLE_MAX_VALUE) {
            return h * Math.exp(value);
        } else {
            double t = Math.exp(value * 0.5);
            return (h * t) * t;
        }
    }
```



---

### Функция ID: 245

**Исходный код:**
```java
@Override
    public boolean equals(final Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof Map.Entry<?, ?>) {
            final Map.Entry<?, ?> other = (Map.Entry<?, ?>) obj;
            return Objects.equals(getKey(), other.getKey())
                    && Objects.equals(getValue(), other.getValue());
        }
        return false;
    }
```



---

### Функция ID: 246

**Исходный код:**
```python
def drop_duplicates(
        self,
        *,
        keep: DropKeep = "first",
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> Series | None:
        """
        Return Series with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        inplace : bool, default ``False``
            If ``True``, performs operation inplace and returns None.

        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        Series or None
            Series with duplicates dropped or None if ``inplace=True``.

        See Also
        --------
        Index.drop_duplicates : Equivalent method on Index.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.
        Series.duplicated : Related method on Series, indicating duplicate
            Series values.
        Series.unique : Return unique values as an array.

        Examples
        --------
        Generate a Series with duplicated entries.

        >>> s = pd.Series(
        ...     ["llama", "cow", "llama", "beetle", "llama", "hippo"], name="animal"
        ... )
        >>> s
        0     llama
        1       cow
        2     llama
        3    beetle
        4     llama
        5     hippo
        Name: animal, dtype: object

        With the 'keep' parameter, the selection behavior of duplicated values
        can be changed. The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> s.drop_duplicates()
        0     llama
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object

        The value 'last' for parameter 'keep' keeps the last occurrence for
        each set of duplicated entries.

        >>> s.drop_duplicates(keep="last")
        1       cow
        3    beetle
        4     llama
        5     hippo
        Name: animal, dtype: object

        The value ``False`` for parameter 'keep' discards all sets of
        duplicated entries.

        >>> s.drop_duplicates(keep=False)
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        result = super().drop_duplicates(keep=keep)

        if ignore_index:
            result.index = default_index(len(result))

        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result
```



---

### Функция ID: 247

**Исходный код:**
```python
def insert(self, key: str, value: bytes) -> bool:
        """Insert a key-value pair into the on-disk cache.

        Args:
            key: The key to insert (must be str).
            value: The value to associate with the key (must be bytes).

        Returns:
            True if successfully inserted, False if the key already exists
            with a valid version.
        """
        fpath: Path = self._fpath_from_key(key)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        r_fp, w_fp, inserted = None, None, False
        try:
            w_fp = open(fpath, "xb")  # noqa: SIM115
        except FileExistsError:
            is_stale: bool = False
            with open(fpath, "rb") as r_fp:
                is_stale = not self._version_header_matches(r_fp)

            if is_stale:
                # same story as above, in this case the version header doesn't
                # match so we choose to remove the old entry so that the new
                # k/v pair can be cached
                fpath.unlink()
                w_fp = open(fpath, "xb")  # noqa: SIM115
            else:
                w_fp = None
        finally:
            if w_fp:
                try:
                    self._write_version_header(w_fp)
                    w_fp.write(value)
                    inserted = True
                finally:
                    w_fp.close()

        return inserted
```



---

### Функция ID: 248

**Исходный код:**
```java
@Override
    public boolean equals(final Object obj) {
        if (this == obj) {
            return true;
        }
        if (!super.equals(obj)) {
            return false;
        }
        if (!(obj instanceof ExtendedMessageFormat)) {
            return false;
        }
        final ExtendedMessageFormat other = (ExtendedMessageFormat) obj;
        return Objects.equals(registry, other.registry) && Objects.equals(toPattern, other.toPattern);
    }
```



---

### Функция ID: 249

**Исходный код:**
```java
private boolean isRegistered(@Nullable Throwable ex) {
		if (ex == null) {
			return false;
		}
		if (this.loggedExceptions.contains(ex)) {
			return true;
		}
		if (ex instanceof InvocationTargetException) {
			return isRegistered(ex.getCause());
		}
		return false;
	}
```



---

### Функция ID: 250

**Исходный код:**
```python
def get_import_status(self, import_arn: str) -> tuple[str, str | None, str | None]:
        """
        Get import status from Dynamodb.

        :param import_arn: The Amazon Resource Name (ARN) for the import.
        :return: Import status, Error code and Error message
        """
        self.log.info("Poking for Dynamodb import %s", import_arn)

        try:
            describe_import = self.client.describe_import(ImportArn=import_arn)
            status = describe_import["ImportTableDescription"]["ImportStatus"]
            error_code = describe_import["ImportTableDescription"].get("FailureCode")
            error_msg = describe_import["ImportTableDescription"].get("FailureMessage")
            return status, error_code, error_msg
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "ImportNotFoundException":
                raise AirflowException("S3 import into Dynamodb job not found.")
            raise e
```



---

### Функция ID: 251

**Исходный код:**
```java
public <T> CompletableFuture<T> setActiveTask(final CompletableFuture<T> currentTask) {
        Objects.requireNonNull(currentTask, "currentTask cannot be null");
        pendingTask.getAndUpdate(task -> {
            if (task == null) {
                return new ActiveFuture(currentTask);
            } else if (task instanceof WakeupFuture) {
                currentTask.completeExceptionally(new WakeupException());
                return null;
            } else if (task instanceof DisabledWakeups) {
                return task;
            }
            // last active state is still active
            throw new KafkaException("Last active task is still active");
        });
        return currentTask;
    }
```



---

### Функция ID: 252

**Исходный код:**
```java
private TypeMirror getCollectionElementType(TypeMirror type) {
		if (((TypeElement) this.types.asElement(type)).getQualifiedName().contentEquals(Collection.class.getName())) {
			DeclaredType declaredType = (DeclaredType) type;
			// raw type, just "Collection"
			if (declaredType.getTypeArguments().isEmpty()) {
				return this.types.getDeclaredType(this.env.getElementUtils().getTypeElement(Object.class.getName()));
			}
			// return type argument to Collection<...>
			return declaredType.getTypeArguments().get(0);
		}

		// recursively walk the supertypes, looking for Collection<...>
		for (TypeMirror superType : this.env.getTypeUtils().directSupertypes(type)) {
			if (this.types.isAssignable(superType, this.collectionType)) {
				return getCollectionElementType(superType);
			}
		}
		return null;
	}
```



---

### Функция ID: 253

**Исходный код:**
```java
public static boolean containsTypeVariables(final Type type) {
        if (type instanceof TypeVariable<?>) {
            return true;
        }
        if (type instanceof Class<?>) {
            return ((Class<?>) type).getTypeParameters().length > 0;
        }
        if (type instanceof ParameterizedType) {
            for (final Type arg : ((ParameterizedType) type).getActualTypeArguments()) {
                if (containsTypeVariables(arg)) {
                    return true;
                }
            }
            return false;
        }
        if (type instanceof WildcardType) {
            final WildcardType wild = (WildcardType) type;
            return containsTypeVariables(getImplicitLowerBounds(wild)[0]) || containsTypeVariables(getImplicitUpperBounds(wild)[0]);
        }
        if (type instanceof GenericArrayType) {
            return containsTypeVariables(((GenericArrayType) type).getGenericComponentType());
        }
        return false;
    }
```



---

### Функция ID: 254

**Исходный код:**
```java
private boolean containsElements(final Collection<?> coll) {
        if (coll == null || coll.isEmpty()) {
            return false;
        }
        return coll.stream().anyMatch(Objects::nonNull);
    }
```



---

### Функция ID: 255

**Исходный код:**
```java
@SuppressWarnings("NullAway") // Dataflow analysis limitation
	protected @Nullable Method doFindMatchingMethod(@Nullable Object[] arguments) {
		TypeConverter converter = getTypeConverter();
		if (converter != null) {
			String targetMethod = getTargetMethod();
			Method matchingMethod = null;
			int argCount = arguments.length;
			Class<?> targetClass = getTargetClass();
			Assert.state(targetClass != null, "No target class set");
			Method[] candidates = ReflectionUtils.getAllDeclaredMethods(targetClass);
			int minTypeDiffWeight = Integer.MAX_VALUE;
			@Nullable Object[] argumentsToUse = null;
			for (Method candidate : candidates) {
				if (candidate.getName().equals(targetMethod)) {
					// Check if the inspected method has the correct number of parameters.
					int parameterCount = candidate.getParameterCount();
					if (parameterCount == argCount) {
						Class<?>[] paramTypes = candidate.getParameterTypes();
						@Nullable Object[] convertedArguments = new Object[argCount];
						boolean match = true;
						for (int j = 0; j < argCount && match; j++) {
							// Verify that the supplied argument is assignable to the method parameter.
							try {
								convertedArguments[j] = converter.convertIfNecessary(arguments[j], paramTypes[j]);
							}
							catch (TypeMismatchException ex) {
								// Ignore -> simply doesn't match.
								match = false;
							}
						}
						if (match) {
							int typeDiffWeight = getTypeDifferenceWeight(paramTypes, convertedArguments);
							if (typeDiffWeight < minTypeDiffWeight) {
								minTypeDiffWeight = typeDiffWeight;
								matchingMethod = candidate;
								argumentsToUse = convertedArguments;
							}
						}
					}
				}
			}
			if (matchingMethod != null) {
				setArguments(argumentsToUse);
				return matchingMethod;
			}
		}
		return null;
	}
```



---

### Функция ID: 256

**Исходный код:**
```java
public Throwable @Nullable [] getRelatedCauses() {
		if (this.relatedCauses == null) {
			return null;
		}
		return this.relatedCauses.toArray(new Throwable[0]);
	}
```



---

### Функция ID: 257

**Исходный код:**
```java
@Override
		public boolean equals(Object obj) {
			if (this == obj) {
				return true;
			}
			if (obj == null || getClass() != obj.getClass()) {
				return false;
			}
			Options other = (Options) obj;
			return this.options.equals(other.options);
		}
```



---

### Функция ID: 258

**Исходный код:**
```java
public Object get(int index) throws JSONException {
		try {
			Object value = this.values.get(index);
			if (value == null) {
				throw new JSONException("Value at " + index + " is null.");
			}
			return value;
		}
		catch (IndexOutOfBoundsException e) {
			throw new JSONException("Index " + index + " out of range [0.." + this.values.size() + ")");
		}
	}
```



---

### Функция ID: 259

**Исходный код:**
```java
private void logNonMatchingType(C callback, ClassCastException ex) {
			if (this.logger.isDebugEnabled()) {
				Class<?> expectedType = ResolvableType.forClass(this.callbackType).resolveGeneric();
				String expectedTypeName = (expectedType != null) ? ClassUtils.getShortName(expectedType) + " type"
						: "type";
				String message = "Non-matching " + expectedTypeName + " for callback "
						+ ClassUtils.getShortName(this.callbackType) + ": " + callback;
				this.logger.debug(message, ex);
			}
		}
```



---

### Функция ID: 260

**Исходный код:**
```java
public StrBuilder appendWithSeparators(final Object[] array, final String separator) {
        if (array != null && array.length > 0) {
            final String sep = Objects.toString(separator, "");
            append(array[0]);
            for (int i = 1; i < array.length; i++) {
                append(sep);
                append(array[i]);
            }
        }
        return this;
    }
```



---

### Функция ID: 261

**Исходный код:**
```python
def find_airflow_root_path_to_operate_on() -> Path:
    """
    Find the root of airflow sources we operate on. Handle the case when Breeze is installed via
    `pipx` or `uv tool` from a different source tree, so it searches upwards of the current directory
    to find the right root of airflow directory we are actually in. This **might** be different
    than the sources of Airflow Breeze was installed from.

    If not found, we operate on Airflow sources that we were installed it. This handles the case when
    we run Breeze from a "random" directory.

    This method also handles the following errors and warnings:

       * It fails (and exits hard) if Breeze is installed in non-editable mode (in which case it will
         not find the Airflow sources when walking upwards the directory where it is installed)
       * It warns (with 2 seconds timeout) if you are using Breeze from a different airflow sources than
         the one you operate on.
       * If we are running in the same source tree as where Breeze was installed from (so no warning above),
         it warns (with 2 seconds timeout) if there is a change in setup.* files of Breeze since installation
         time. In such case usesr is encouraged to re-install Breeze to update dependencies.

    :return: Path for the found sources.

    """
    sources_root_from_env = os.getenv("AIRFLOW_ROOT_PATH", None)
    if sources_root_from_env:
        return Path(sources_root_from_env)
    installation_airflow_sources = get_installation_airflow_sources()
    if installation_airflow_sources is None and not skip_breeze_self_upgrade_check():
        get_console().print(
            "\n[error]Breeze should only be installed with --editable flag[/]\n\n"
            "[warning]Please go to Airflow sources and run[/]\n\n"
            f"     {NAME} setup self-upgrade --use-current-airflow-sources\n"
            '[warning]If during installation you see warning starting "Ignoring --editable install",[/]\n'
            '[warning]make sure you first downgrade "packaging" package to <23.2, for example by:[/]\n\n'
            f'     pip install "packaging<23.2"\n\n'
        )
        sys.exit(1)
    airflow_sources = get_used_airflow_sources()
    if not skip_breeze_self_upgrade_check():
        # only print warning and sleep if not producing complete results
        reinstall_if_different_sources(airflow_sources)
        reinstall_if_setup_changed()
    os.chdir(airflow_sources.as_posix())
    airflow_home_dir = Path(os.environ.get("AIRFLOW_HOME", (Path.home() / "airflow").resolve().as_posix()))
    if airflow_sources.resolve() == airflow_home_dir.resolve():
        get_console().print(
            f"\n[error]Your Airflow sources are checked out in {airflow_home_dir} which "
            f"is your also your AIRFLOW_HOME where airflow writes logs and database. \n"
            f"This is a bad idea because Airflow might override and cleanup your checked out "
            f"sources and .git repository.[/]\n"
        )
        get_console().print("\nPlease check out your Airflow code elsewhere.\n")
        sys.exit(1)
    return airflow_sources
```



---

### Функция ID: 262

**Исходный код:**
```java
@Nullable String getErrorReport() {
		Map<String, List<PropertyMigration>> content = getContent(LegacyProperties::getUnsupported);
		if (content.isEmpty()) {
			return null;
		}
		StringBuilder report = new StringBuilder();
		report.append(String
			.format("%nThe use of configuration keys that are no longer supported was found in the environment:%n%n"));
		append(report, content);
		report.append(String.format("%n"));
		report.append("Please refer to the release notes or reference guide for potential alternatives.");
		report.append(String.format("%n"));
		return report.toString();
	}
```



---

### Функция ID: 263

**Исходный код:**
```java
public boolean awaitPendingRequests(Node node, Timer timer) {
        while (hasPendingRequests(node) && timer.notExpired()) {
            poll(timer);
        }
        return !hasPendingRequests(node);
    }
```



---

### Функция ID: 264

**Исходный код:**
```java
private boolean acquirePermit() {
        if (getLimit() <= NO_LIMIT || acquireCount < getLimit()) {
            acquireCount++;
            return true;
        }
        return false;
    }
```



---

### Функция ID: 265

**Исходный код:**
```java
public static short min(short a, final short b, final short c) {
        if (b < a) {
            a = b;
        }
        if (c < a) {
            a = c;
        }
        return a;
    }
```



---

### Функция ID: 266

**Исходный код:**
```java
public static StrMatcher charSetMatcher(final char... chars) {
        if (ArrayUtils.isEmpty(chars)) {
            return NONE_MATCHER;
        }
        if (chars.length == 1) {
            return new CharMatcher(chars[0]);
        }
        return new CharSetMatcher(chars);
    }
```



---

### Функция ID: 267

**Исходный код:**
```python
def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to enable memoization.

            Args:
                fn: The function to wrap.

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, return the original function unchanged
            if not config.IS_CACHING_MODULE_ENABLED():
                return fn

            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Call the original function and cache the result.

                Args:
                    *args: Positional arguments to pass to the function.
                    **kwargs: Keyword arguments to pass to the function.

                Returns:
                    The result of calling the original function.
                """
                # Call the function to compute the result
                result = fn(*args, **kwargs)

                # Generate cache key from parameters
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)

                # Encode params for human-readable dump
                if custom_params_encoder is not None:
                    encoded_params = custom_params_encoder(*args, **kwargs)
                else:
                    encoded_params = {
                        "args": args,
                        "kwargs": kwargs,
                    }

                # Encode the result if encoder is provided
                if custom_result_encoder is not None:
                    # Get the encoder function by calling the factory with params
                    encoder_fn = custom_result_encoder(*args, **kwargs)
                    encoded_result = encoder_fn(result)
                else:
                    encoded_result = result

                # Store CacheEntry in cache
                cache_entry = CacheEntry(
                    encoded_params=encoded_params,
                    encoded_result=encoded_result,
                )
                self._cache.insert(cache_key, cache_entry)

                # Return the original result (not the encoded version)
                return result

            return inner
```



---

### Функция ID: 268

**Исходный код:**
```java
public MutablePropertyValues addPropertyValues(@Nullable Map<?, ?> other) {
		if (other != null) {
			other.forEach((attrName, attrValue) -> addPropertyValue(
					new PropertyValue(attrName.toString(), attrValue)));
		}
		return this;
	}
```



---

### Функция ID: 269

**Исходный код:**
```java
public static Integer[] toObject(final int[] array) {
        if (array == null) {
            return null;
        }
        if (array.length == 0) {
            return EMPTY_INTEGER_OBJECT_ARRAY;
        }
        return setAll(new Integer[array.length], i -> Integer.valueOf(array[i]));
    }
```



---

### Функция ID: 270

**Исходный код:**
```java
public boolean hasIndexedElement() {
		for (int i = 0; i < getNumberOfElements(); i++) {
			if (isIndexed(i)) {
				return true;
			}
		}
		return false;
	}
```



---

### Функция ID: 271

**Исходный код:**
```java
@Override
	public boolean removeAdvisor(Advisor advisor) {
		int index = indexOf(advisor);
		if (index == -1) {
			return false;
		}
		else {
			removeAdvisor(index);
			return true;
		}
	}
```



---

### Функция ID: 272

**Исходный код:**
```java
protected boolean shouldInject(@Nullable PropertyValues pvs) {
			if (this.isField) {
				return true;
			}
			return !checkPropertySkipping(pvs);
		}
```



---

### Функция ID: 273

**Исходный код:**
```cpp
Ptr<RHO_HEST> rhoInit(void){
    /* Select an optimized implementation of RHO here. */

#if 1
    /**
     * For now, only the generic C implementation is available. In the future,
     * SSE2/AVX/AVX2/FMA/NEON versions may be added, and they will be selected
     * depending on cv::checkHardwareSupport()'s return values.
     */

    Ptr<RHO_HEST> p = Ptr<RHO_HEST>(new RHO_HEST_REFC);
#endif

    /* Initialize it. */
    if(p){
        if(!p->initialize()){
            p.release();
        }
    }

    /* Return it. */
    return p;
}
```



---

### Функция ID: 274

**Исходный код:**
```java
Map<Uuid, SortedSet<Integer>> topicPartitionsAwaitingReconciliation() {
        if (currentTargetAssignment == LocalAssignment.NONE) {
            return Collections.emptyMap();
        }
        if (currentAssignment == LocalAssignment.NONE) {
            return currentTargetAssignment.partitions;
        }
        final Map<Uuid, SortedSet<Integer>> topicPartitionMap = new HashMap<>();
        currentTargetAssignment.partitions.forEach((topicId, targetPartitions) -> {
            final SortedSet<Integer> reconciledPartitions = currentAssignment.partitions.get(topicId);
            if (!targetPartitions.equals(reconciledPartitions)) {
                final TreeSet<Integer> missingPartitions = new TreeSet<>(targetPartitions);
                if (reconciledPartitions != null) {
                    missingPartitions.removeAll(reconciledPartitions);
                }
                topicPartitionMap.put(topicId, missingPartitions);
            }
        });
        return Collections.unmodifiableMap(topicPartitionMap);
    }
```



---

### Функция ID: 275

**Исходный код:**
```python
def _refine_color(color: str):
    """
    Convert color in #RGB (12 bits) format to #RRGGBB (32 bits), if it possible.

    Otherwise, it returns the original value. Graphviz does not support colors in #RGB format.

    :param color: Text representation of color
    :return: Refined representation of color
    """
    if len(color) == 4 and color[0] == "#":
        color_r = color[1]
        color_g = color[2]
        color_b = color[3]
        return "#" + color_r + color_r + color_g + color_g + color_b + color_b
    return color
```



---

### Функция ID: 276

**Исходный код:**
```java
@Override
    public boolean equals(final Object obj) {
        if (!(obj instanceof FastDateParser)) {
            return false;
        }
        final FastDateParser other = (FastDateParser) obj;
        return pattern.equals(other.pattern) && timeZone.equals(other.timeZone) && locale.equals(other.locale);
    }
```



---

### Функция ID: 277

**Исходный код:**
```typescript
function modelsLayer(client: Client): CompositeProxyLayer {
  const dmmfModelKeys = Object.keys(client._runtimeDataModel.models)
  const jsModelKeys = dmmfModelKeys.map(dmmfToJSModelName)
  const allKeys = [...new Set(dmmfModelKeys.concat(jsModelKeys))]

  return cacheProperties({
    getKeys() {
      return allKeys
    },

    getPropertyValue(prop) {
      const dmmfModelName = jsToDMMFModelName(prop)
      // creates a new model proxy on the fly and caches it
      if (client._runtimeDataModel.models[dmmfModelName] !== undefined) {
        return applyModel(client, dmmfModelName)
      }

      // above silently failed if model name is lower cased
      if (client._runtimeDataModel.models[prop] !== undefined) {
        return applyModel(client, prop)
      }

      return undefined
    },

    getPropertyDescriptor(key) {
      if (!jsModelKeys.includes(key)) {
        return { enumerable: false }
      }

      return undefined
    },
  })
}
```



---

### Функция ID: 278

**Исходный код:**
```python
def get_job_state(self, job_name: str, run_id: str) -> str:
        """
        Get state of the Glue job; the job state can be running, finished, failed, stopped or timeout.

        .. seealso::
            - :external+boto3:py:meth:`Glue.Client.get_job_run`

        :param job_name: unique job name per AWS account
        :param run_id: The job-run ID of the predecessor job run
        :return: State of the Glue job
        """
        for attempt in Retrying(**self.retry_config):
            with attempt:
                try:
                    job_run = self.conn.get_job_run(JobName=job_name, RunId=run_id, PredecessorsIncluded=True)
                    return job_run["JobRun"]["JobRunState"]
                except ClientError as e:
                    self.log.error("Failed to get job state for job %s run %s: %s", job_name, run_id, e)
                    raise
                except Exception as e:
                    self.log.error(
                        "Unexpected error getting job state for job %s run %s: %s", job_name, run_id, e
                    )
                    raise
        # This should never be reached due to reraise=True, but mypy needs it
        raise RuntimeError("Unexpected end of retry loop")
```



---

### Функция ID: 279

**Исходный код:**
```java
public static TimeValue parseTimeValue(@Nullable String sValue, TimeValue defaultValue, String settingName) {
        settingName = Objects.requireNonNull(settingName);
        if (sValue == null) {
            return defaultValue;
        }
        final String normalized = sValue.toLowerCase(Locale.ROOT).trim();
        if (normalized.endsWith("nanos")) {
            return TimeValue.timeValueNanos(parse(sValue, normalized, "nanos", settingName));
        } else if (normalized.endsWith("micros")) {
            return new TimeValue(parse(sValue, normalized, "micros", settingName), TimeUnit.MICROSECONDS);
        } else if (normalized.endsWith("ms")) {
            return TimeValue.timeValueMillis(parse(sValue, normalized, "ms", settingName));
        } else if (normalized.endsWith("s")) {
            return TimeValue.timeValueSeconds(parse(sValue, normalized, "s", settingName));
        } else if (sValue.endsWith("m")) {
            // parsing minutes should be case-sensitive as 'M' means "months", not "minutes"; this is the only special case.
            return TimeValue.timeValueMinutes(parse(sValue, normalized, "m", settingName));
        } else if (normalized.endsWith("h")) {
            return TimeValue.timeValueHours(parse(sValue, normalized, "h", settingName));
        } else if (normalized.endsWith("d")) {
            return new TimeValue(parse(sValue, normalized, "d", settingName), TimeUnit.DAYS);
        } else if (normalized.matches("-0*1")) {
            return TimeValue.MINUS_ONE;
        } else if (normalized.matches("0+")) {
            return TimeValue.ZERO;
        } else {
            // Missing units:
            throw new IllegalArgumentException(
                "failed to parse setting [" + settingName + "] with value [" + sValue + "] as a time value: unit is missing or unrecognized"
            );
        }
    }
```



---

### Функция ID: 280

**Исходный код:**
```java
private static float getObjectTransformationCost(Class<?> srcClass, final Class<?> destClass) {
        if (destClass.isPrimitive()) {
            return getPrimitivePromotionCost(srcClass, destClass);
        }
        float cost = 0.0f;
        while (srcClass != null && !destClass.equals(srcClass)) {
            if (destClass.isInterface() && ClassUtils.isAssignable(srcClass, destClass)) {
                // slight penalty for interface match.
                // we still want an exact match to override an interface match,
                // but
                // an interface match should override anything where we have to
                // get a superclass.
                cost += 0.25f;
                break;
            }
            cost++;
            srcClass = srcClass.getSuperclass();
        }
        /*
         * If the destination class is null, we've traveled all the way up to an Object match. We'll penalize this by adding 1.5 to the cost.
         */
        if (srcClass == null) {
            cost += 1.5f;
        }
        return cost;
    }
```



---

### Функция ID: 281

**Исходный код:**
```java
public int getInt(int index) throws JSONException {
		Object object = get(index);
		Integer result = JSON.toInteger(object);
		if (result == null) {
			throw JSON.typeMismatch(index, object, "int");
		}
		return result;
	}
```



---

### Функция ID: 282

**Исходный код:**
```python
def strip_leading_zeros_from_version(version: str) -> str:
    """
    Strips leading zeros from version number.

    This converts 1974.04.03 to 1974.4.3 as the format with leading month and day zeros is not accepted
    by PIP versioning.

    :param version: version number in CALVER format (potentially with leading 0s in date and month)
    :return: string with leading 0s after dot replaced.
    """
    return ".".join(i.lstrip("0") or "0" for i in version.split("."))
```



---

### Функция ID: 283

**Исходный код:**
```java
public KafkaFuture<Map<String, UserScramCredentialsDescription>> all() {
        final KafkaFutureImpl<Map<String, UserScramCredentialsDescription>> retval = new KafkaFutureImpl<>();
        dataFuture.whenComplete((data, throwable) -> {
            if (throwable != null) {
                retval.completeExceptionally(throwable);
            } else {
                /* Check to make sure every individual described user succeeded.  Note that a successfully described user
                 * is one that appears with *either* a NONE error code or a RESOURCE_NOT_FOUND error code. The
                 * RESOURCE_NOT_FOUND means the client explicitly requested a describe of that particular user but it could
                 * not be described because it does not exist; such a user will not appear as a key in the returned map.
                 */
                Optional<DescribeUserScramCredentialsResponseData.DescribeUserScramCredentialsResult> optionalFirstFailedDescribe =
                        data.results().stream().filter(result ->
                                result.errorCode() != Errors.NONE.code() && result.errorCode() != Errors.RESOURCE_NOT_FOUND.code()).findFirst();
                if (optionalFirstFailedDescribe.isPresent()) {
                    retval.completeExceptionally(Errors.forCode(optionalFirstFailedDescribe.get().errorCode()).exception(optionalFirstFailedDescribe.get().errorMessage()));
                } else {
                    Map<String, UserScramCredentialsDescription> retvalMap = new HashMap<>();
                    data.results().forEach(userResult ->
                            retvalMap.put(userResult.user(), new UserScramCredentialsDescription(userResult.user(),
                                    getScramCredentialInfosFor(userResult))));
                    retval.complete(retvalMap);
                }
            }
        });
        return retval;
    }
```



---

### Функция ID: 284

**Исходный код:**
```python
def check_md5checksum_in_cache_modified(file_hash: str, cache_path: Path, update: bool) -> bool:
    """
    Check if the file hash is present in cache and its content has been modified. Optionally updates
    the hash.

    :param file_hash: hash of the current version of the file
    :param cache_path: path where the hash is stored
    :param update: whether to update hash if it is found different
    :return: True if the hash file was missing or hash has changed.
    """
    if cache_path.exists():
        old_md5_checksum_content = Path(cache_path).read_text()
        if old_md5_checksum_content.strip() != file_hash.strip():
            if update:
                save_md5_file(cache_path, file_hash)
            return True
    else:
        if update:
            save_md5_file(cache_path, file_hash)
        return True
    return False
```



---

### Функция ID: 285

**Исходный код:**
```java
protected Object wrapIfNecessary(Object bean, String beanName, Object cacheKey) {
		if (StringUtils.hasLength(beanName) && this.targetSourcedBeans.contains(beanName)) {
			return bean;
		}
		if (Boolean.FALSE.equals(this.advisedBeans.get(cacheKey))) {
			return bean;
		}
		if (isInfrastructureClass(bean.getClass()) || shouldSkip(bean.getClass(), beanName)) {
			this.advisedBeans.put(cacheKey, Boolean.FALSE);
			return bean;
		}

		// Create proxy if we have advice.
		Object[] specificInterceptors = getAdvicesAndAdvisorsForBean(bean.getClass(), beanName, null);
		if (specificInterceptors != DO_NOT_PROXY) {
			this.advisedBeans.put(cacheKey, Boolean.TRUE);
			Object proxy = createProxy(
					bean.getClass(), beanName, specificInterceptors, new SingletonTargetSource(bean));
			this.proxyTypes.put(cacheKey, proxy.getClass());
			return proxy;
		}

		this.advisedBeans.put(cacheKey, Boolean.FALSE);
		return bean;
	}
```



---

### Функция ID: 286

**Исходный код:**
```java
int handleTimeouts(Collection<Call> calls, String msg) {
            int numTimedOut = 0;
            for (Iterator<Call> iter = calls.iterator(); iter.hasNext(); ) {
                Call call = iter.next();
                int remainingMs = calcTimeoutMsRemainingAsInt(now, call.deadlineMs);
                if (remainingMs < 0) {
                    call.fail(now, new TimeoutException(msg + " Call: " + call.callName));
                    iter.remove();
                    numTimedOut++;
                } else {
                    nextTimeoutMs = Math.min(nextTimeoutMs, remainingMs);
                }
            }
            return numTimedOut;
        }
```



---

### Функция ID: 287

**Исходный код:**
```java
private int nextCleanInternal() throws JSONException {
		while (this.pos < this.in.length()) {
			int c = this.in.charAt(this.pos++);
			switch (c) {
				case '\t', ' ', '\n', '\r':
					continue;

				case '/':
					if (this.pos == this.in.length()) {
						return c;
					}

					char peek = this.in.charAt(this.pos);
					switch (peek) {
						case '*':
							// skip a /* c-style comment */
							this.pos++;
							int commentEnd = this.in.indexOf("*/", this.pos);
							if (commentEnd == -1) {
								throw syntaxError("Unterminated comment");
							}
							this.pos = commentEnd + 2;
							continue;

						case '/':
							// skip a // end-of-line comment
							this.pos++;
							skipToEndOfLine();
							continue;

						default:
							return c;
					}

				case '#':
					/*
					 * Skip a # hash end-of-line comment. The JSON RFC doesn't specify
					 * this behavior, but it's required to parse existing documents. See
					 * https://b/2571423.
					 */
					skipToEndOfLine();
					continue;

				default:
					return c;
			}
		}

		return -1;
	}
```



---

### Функция ID: 288

**Исходный код:**
```java
public static @Nullable ColorConverter newInstance(@Nullable Configuration config, @Nullable String[] options) {
		if (options.length < 1) {
			LOGGER.error("Incorrect number of options on style. Expected at least 1, received {}", options.length);
			return null;
		}
		if (options[0] == null) {
			LOGGER.error("No pattern supplied on style");
			return null;
		}
		PatternParser parser = PatternLayout.createPatternParser(config);
		List<PatternFormatter> formatters = parser.parse(options[0]);
		AnsiElement element = (options.length != 1) ? ELEMENTS.get(options[1]) : null;
		return new ColorConverter(formatters, element);
	}
```



---

### Функция ID: 289

**Исходный код:**
```java
public static long min(long a, final long b, final long c) {
        if (b < a) {
            a = b;
        }
        if (c < a) {
            a = c;
        }
        return a;
    }
```



---

### Функция ID: 290

**Исходный код:**
```java
static int indexOf(final CharSequence cs, final int searchChar, int start) {
        if (cs instanceof String) {
            return ((String) cs).indexOf(searchChar, start);
        }
        final int sz = cs.length();
        if (start < 0) {
            start = 0;
        }
        if (searchChar < Character.MIN_SUPPLEMENTARY_CODE_POINT) {
            for (int i = start; i < sz; i++) {
                if (cs.charAt(i) == searchChar) {
                    return i;
                }
            }
            return NOT_FOUND;
        }
        //supplementary characters (LANG1300)
        if (searchChar <= Character.MAX_CODE_POINT) {
            final char[] chars = Character.toChars(searchChar);
            for (int i = start; i < sz - 1; i++) {
                final char high = cs.charAt(i);
                final char low = cs.charAt(i + 1);
                if (high == chars[0] && low == chars[1]) {
                    return i;
                }
            }
        }
        return NOT_FOUND;
    }
```



---

### Функция ID: 291

**Исходный код:**
```java
public static float[] sort(final float[] array) {
        if (array != null) {
            Arrays.sort(array);
        }
        return array;
    }
```



---

### Функция ID: 292

**Исходный код:**
```python
def download_patch(pr_number: int, repo_url: str, download_dir: str) -> str:
    """
    Downloads the patch file for a given PR from the specified GitHub repository.

    Args:
        pr_number (int): The pull request number.
        repo_url (str): The URL of the repository where the PR is hosted.
        download_dir (str): The directory to store the downloaded patch.

    Returns:
        str: The path to the downloaded patch file.

    Exits:
        If the download fails, the script will exit.
    """
    patch_url = f"{repo_url}/pull/{pr_number}.diff"
    patch_file = os.path.join(download_dir, f"pr-{pr_number}.patch")
    print(f"Downloading PR #{pr_number} patch from {patch_url}...")
    try:
        with (
            urllib.request.urlopen(patch_url) as response,
            open(patch_file, "wb") as out_file,
        ):
            # pyrefly: ignore [bad-specialization]
            shutil.copyfileobj(response, out_file)
        if not os.path.isfile(patch_file):
            print(f"Failed to download patch for PR #{pr_number}")
            sys.exit(1)
        print(f"Patch downloaded to {patch_file}")
        return patch_file
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} when downloading patch for PR #{pr_number}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while downloading the patch: {e}")
        sys.exit(1)
```



---

### Функция ID: 293

**Исходный код:**
```java
protected Log getApplicationLog() {
		if (this.mainApplicationClass == null) {
			return logger;
		}
		return LogFactory.getLog(this.mainApplicationClass);
	}
```



---

### Функция ID: 294

**Исходный код:**
```java
boolean callHasExpired(Call call) {
            int remainingMs = calcTimeoutMsRemainingAsInt(now, call.deadlineMs);
            if (remainingMs < 0)
                return true;
            nextTimeoutMs = Math.min(nextTimeoutMs, remainingMs);
            return false;
        }
```



---

### Функция ID: 295

**Исходный код:**
```java
public static Writer asWriter(Appendable target) {
    if (target instanceof Writer) {
      return (Writer) target;
    }
    return new AppendableWriter(target);
  }
```



---

### Функция ID: 296

**Исходный код:**
```java
public String replace(final char[] source) {
        if (source == null) {
            return null;
        }
        final StrBuilder buf = new StrBuilder(source.length).append(source);
        substitute(buf, 0, source.length);
        return buf.toString();
    }
```



---

### Функция ID: 297

**Исходный код:**
```python
async def check_for_prefix_async(
        self,
        client: AioBaseClient,
        prefix: str,
        delimiter: str,
        bucket_name: str | None = None,
    ) -> bool:
        """
        Check that a prefix exists in a bucket.

        :param bucket_name: the name of the bucket
        :param prefix: a key prefix
        :param delimiter: the delimiter marks key hierarchy.
        :return: False if the prefix does not exist in the bucket and True if it does.
        """
        if not prefix.endswith(delimiter):
            prefix += delimiter
        prefix_split = re.split(rf"(\w+[{delimiter}])$", prefix, 1)
        previous_level = prefix_split[0]
        plist = await self.list_prefixes_async(client, bucket_name, previous_level, delimiter)
        return prefix in plist
```



---

### Функция ID: 298

**Исходный код:**
```java
public @Nullable MethodOverride getOverride(Method method) {
		MethodOverride match = null;
		for (MethodOverride candidate : this.overrides) {
			if (candidate.matches(method)) {
				match = candidate;
			}
		}
		return match;
	}
```



---

### Функция ID: 299

**Исходный код:**
```java
@CanIgnoreReturnValue // to skip a line
  public @Nullable String readLine() throws IOException {
    while (lines.peek() == null) {
      Java8Compatibility.clear(cbuf);
      // The default implementation of Reader#read(CharBuffer) allocates a
      // temporary char[], so we call Reader#read(char[], int, int) instead.
      int read = (reader != null) ? reader.read(buf, 0, buf.length) : readable.read(cbuf);
      if (read == -1) {
        lineBuf.finish();
        break;
      }
      lineBuf.add(buf, 0, read);
    }
    return lines.poll();
  }
```



---

### Функция ID: 300

**Исходный код:**
```java
public ApiException exception(String message) {
        if (message == null) {
            // If no error message was specified, return an exception with the default error message.
            return exception;
        }
        // Return an exception with the given error message.
        return builder.apply(message);
    }
```



---

### Функция ID: 301

**Исходный код:**
```java
private Map<String, Object> innerCaptures(
        String text,
        Function<GrokCaptureConfig, Function<Consumer<Object>, GrokCaptureExtracter>> getExtracter
    ) {
        byte[] utf8Bytes = text.getBytes(StandardCharsets.UTF_8);
        GrokCaptureExtracter.MapExtracter extracter = new GrokCaptureExtracter.MapExtracter(captureConfig, getExtracter);
        if (match(utf8Bytes, 0, utf8Bytes.length, extracter)) {
            return extracter.result();
        }
        return null;
    }
```



---

### Функция ID: 302

**Исходный код:**
```java
protected ExitStatus run(String... args) throws Exception {
		if (args.length == 0) {
			throw new NoArgumentsException();
		}
		String commandName = args[0];
		String[] commandArguments = Arrays.copyOfRange(args, 1, args.length);
		Command command = findCommand(commandName);
		if (command == null) {
			throw new NoSuchCommandException(commandName);
		}
		beforeRun(command);
		try {
			return command.run(commandArguments);
		}
		finally {
			afterRun(command);
		}
	}
```



---

### Функция ID: 303

**Исходный код:**
```java
public static <E extends Throwable> long getAsLong(final FailableLongSupplier<E> supplier) {
        try {
            return supplier.getAsLong();
        } catch (final Throwable t) {
            throw rethrow(t);
        }
    }
```



---

### Функция ID: 304

**Исходный код:**
```python
def option_context(*args) -> Generator[None]:
    """
    Context manager to temporarily set options in a ``with`` statement.

    This method allows users to set one or more pandas options temporarily
    within a controlled block. The previous options' values are restored
    once the block is exited. This is useful when making temporary adjustments
    to pandas' behavior without affecting the global state.

    Parameters
    ----------
    *args : str | object | dict
        An even amount of arguments provided in pairs which will be
        interpreted as (pattern, value) pairs. Alternatively, a single
        dictionary of {pattern: value} may be provided.

    Returns
    -------
    None
        No return value.

    Yields
    ------
    None
        No yield value.

    See Also
    --------
    get_option : Retrieve the value of the specified option.
    set_option : Set the value of the specified option.
    reset_option : Reset one or more options to their default value.
    describe_option : Print the description for one or more registered options.

    Notes
    -----
    For all available options, please view the :ref:`User Guide <options.available>`
    or use ``pandas.describe_option()``.

    Examples
    --------
    >>> from pandas import option_context
    >>> with option_context("display.max_rows", 10, "display.max_columns", 5):
    ...     pass
    >>> with option_context({"display.max_rows": 10, "display.max_columns": 5}):
    ...     pass
    """
    if len(args) == 1 and isinstance(args[0], dict):
        args = tuple(kv for item in args[0].items() for kv in item)

    if len(args) % 2 != 0 or len(args) < 2:
        raise ValueError(
            "Provide an even amount of arguments as "
            "option_context(pat, val, pat, val...)."
        )

    ops = tuple(zip(args[::2], args[1::2], strict=True))
    undo: tuple[tuple[Any, Any], ...] = ()
    try:
        undo = tuple((pat, get_option(pat)) for pat, val in ops)
        for pat, val in ops:
            set_option(pat, val)
        yield
    finally:
        for pat, val in undo:
            set_option(pat, val)
```



---

### Функция ID: 305

**Исходный код:**
```java
@Override
    public boolean equals(final Object obj) {
        if (!(obj instanceof FastDatePrinter)) {
            return false;
        }
        final FastDatePrinter other = (FastDatePrinter) obj;
        return pattern.equals(other.pattern)
            && timeZone.equals(other.timeZone)
            && locale.equals(other.locale);
    }
```



---

### Функция ID: 306

**Исходный код:**
```java
@Override
    public String toString() {
        if (tokens == null) {
            return "StrTokenizer[not tokenized yet]";
        }
        return "StrTokenizer" + getTokenList();
    }
```



---

### Функция ID: 307

**Исходный код:**
```java
public static int max(int a, final int b, final int c) {
        if (b > a) {
            a = b;
        }
        if (c > a) {
            a = c;
        }
        return a;
    }
```



---

### Функция ID: 308

**Исходный код:**
```python
def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to enable memoization with replay and record.

            Args:
                fn: The function to wrap.

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, return the original function unchanged
            if not config.IS_CACHING_MODULE_ENABLED():
                return fn

            # Create decorated versions using record and replay
            replay_fn = self.replay(
                custom_params_encoder,
                custom_result_decoder,
            )(fn)
            record_fn = self.record(
                custom_params_encoder,
                custom_result_encoder,
            )(fn)

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Attempt to replay from cache, or record on cache miss.

                Args:
                    *args: Positional arguments to pass to the function.
                    **kwargs: Keyword arguments to pass to the function.

                Returns:
                    The result from cache (if hit) or from executing the function (if miss).
                """
                # Try to replay first
                try:
                    return replay_fn(*args, **kwargs)
                except KeyError:
                    # Cache miss - record the result
                    return record_fn(*args, **kwargs)

            return inner
```



---

### Функция ID: 309

**Исходный код:**
```java
private void newline() {
		if (this.indent == null) {
			return;
		}

		this.out.append("\n");
		this.out.append(this.indent.repeat(this.stack.size()));
	}
```



---

### Функция ID: 310

**Исходный код:**
```java
public boolean isAfter(final T element) {
        if (element == null) {
            return false;
        }
        return comparator.compare(element, minimum) < 0;
    }
```



---

### Функция ID: 311

**Исходный код:**
```java
static @Nullable Command find(Collection<? extends Command> commands, String name) {
		for (Command command : commands) {
			if (command.getName().equals(name)) {
				return command;
			}
		}
		return null;
	}
```



---

### Функция ID: 312

**Исходный код:**
```java
default String printStackTraceToString(Throwable throwable) {
		try {
			StringBuilder out = new StringBuilder(4096);
			printStackTrace(throwable, out);
			return out.toString();
		}
		catch (IOException ex) {
			throw new UncheckedIOException(ex);
		}
	}
```



---

### Функция ID: 313

**Исходный код:**
```java
public long getLong(int index) throws JSONException {
		Object object = get(index);
		Long result = JSON.toLong(object);
		if (result == null) {
			throw JSON.typeMismatch(index, object, "long");
		}
		return result;
	}
```



---

### Функция ID: 314

**Исходный код:**
```python
def get_table_primary_key(self, table: str, schema: str | None = "public") -> list[str] | None:
        """
        Get the table's primary key.

        :param table: Name of the target table
        :param schema: Name of the target schema, public by default
        :return: Primary key columns list
        """
        sql = """
            select kcu.column_name
            from information_schema.table_constraints tco
                    join information_schema.key_column_usage kcu
                        on kcu.constraint_name = tco.constraint_name
                            and kcu.constraint_schema = tco.constraint_schema
                            and kcu.constraint_name = tco.constraint_name
            where tco.constraint_type = 'PRIMARY KEY'
            and kcu.table_schema = %s
            and kcu.table_name = %s
        """
        pk_columns = [row[0] for row in self.get_records(sql, (schema, table))]
        return pk_columns or None
```



---

### Функция ID: 315

**Исходный код:**
```java
public static Byte[] toObject(final byte[] array) {
        if (array == null) {
            return null;
        }
        if (array.length == 0) {
            return EMPTY_BYTE_OBJECT_ARRAY;
        }
        return setAll(new Byte[array.length], i -> Byte.valueOf(array[i]));
    }
```



---

### Функция ID: 316

**Исходный код:**
```java
public static byte min(final byte... array) {
        // Validates input
        validateArray(array);
        // Finds and returns min
        byte min = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] < min) {
                min = array[i];
            }
        }
        return min;
    }
```



---

### Функция ID: 317

**Исходный код:**
```python
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = None,
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: CompressionOptions = "infer",
) -> DataFrame | SASReader:
    """
    Read SAS files stored as either XPORT or SAS7BDAT format files.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function. The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.sas7bdat``.
    format : str {{'xport', 'sas7bdat'}} or None
        If None, file format is inferred from file extension. If 'xport' or
        'sas7bdat', uses the corresponding format.
    index : identifier of index column, defaults to None
        Identifier of column that should be used as index of the DataFrame.
    encoding : str, default is None
        Encoding for text data.  If None, text data are stored as raw bytes.
    chunksize : int
        Read file `chunksize` lines at a time, returns iterator.
    iterator : bool, defaults to False
        If True, returns an iterator for reading the file incrementally.
    {decompression_options}

    Returns
    -------
    DataFrame, SAS7BDATReader, or XportReader
        DataFrame if iterator=False and chunksize=None, else SAS7BDATReader
        or XportReader, file format is inferred from file extension.

    See Also
    --------
    read_csv : Read a comma-separated values (csv) file into a pandas DataFrame.
    read_excel : Read an Excel file into a pandas DataFrame.
    read_spss : Read an SPSS file into a pandas DataFrame.
    read_orc : Load an ORC object into a pandas DataFrame.
    read_feather : Load a feather-format object into a pandas DataFrame.

    Examples
    --------
    >>> df = pd.read_sas("sas_data.sas7bdat")  # doctest: +SKIP
    """
    if format is None:
        buffer_error_msg = (
            "If this is a buffer object rather "
            "than a string name, you must specify a format string"
        )
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        if not isinstance(filepath_or_buffer, str):
            raise ValueError(buffer_error_msg)
        fname = filepath_or_buffer.lower()
        if ".xpt" in fname:
            format = "xport"
        elif ".sas7bdat" in fname:
            format = "sas7bdat"
        else:
            raise ValueError(
                f"unable to infer format of SAS file from filename: {fname!r}"
            )

    reader: SASReader
    if format.lower() == "xport":
        from pandas.io.sas.sas_xport import XportReader

        reader = XportReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    elif format.lower() == "sas7bdat":
        from pandas.io.sas.sas7bdat import SAS7BDATReader

        reader = SAS7BDATReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    else:
        raise ValueError("unknown SAS format")

    if iterator or chunksize:
        return reader

    with reader:
        return reader.read()
```



---

### Функция ID: 318

**Исходный код:**
```python
def deactivate_deleted_dags(
        cls,
        bundle_name: str,
        rel_filelocs: list[str],
        session: Session = NEW_SESSION,
    ) -> bool:
        """
        Set ``is_active=False`` on the DAGs for which the DAG files have been removed.

        :param bundle_name: bundle for filelocs
        :param rel_filelocs: relative filelocs for bundle
        :param session: ORM Session
        :return: True if any DAGs were marked as stale, False otherwise
        """
        log.debug("Deactivating DAGs (for which DAG files are deleted) from %s table ", cls.__tablename__)
        dag_models = session.scalars(
            select(cls)
            .where(
                cls.bundle_name == bundle_name,
            )
            .options(
                load_only(
                    cls.relative_fileloc,
                    cls.is_stale,
                ),
            )
        )

        any_deactivated = False
        for dm in dag_models:
            if dm.relative_fileloc not in rel_filelocs:
                dm.is_stale = True
                any_deactivated = True

        return any_deactivated
```



---

### Функция ID: 319

**Исходный код:**
```typescript
function topStep<T>(array: ReadonlyArray<T>, compare: (a: T, b: T) => number, result: T[], i: number, m: number): void {
	for (const n = result.length; i < m; i++) {
		const element = array[i];
		if (compare(element, result[n - 1]) < 0) {
			result.pop();
			const j = findFirstIdxMonotonousOrArrLen(result, e => compare(element, e) < 0);
			result.splice(j, 0, element);
		}
	}
}
```



---

### Функция ID: 320

**Исходный код:**
```typescript
function getLastDefinedValue<T>(data: T[], index: number): T {
  for (let i = index; i > -1; i--) {
    if (typeof data[i] !== 'undefined') {
      return data[i];
    }
  }
  throw new RuntimeError(
    RuntimeErrorCode.LOCALE_DATA_UNDEFINED,
    ngDevMode && 'Locale data API: locale data undefined',
  );
}
```



---

### Функция ID: 321

**Исходный код:**
```java
public Object remove(int index) {
		if (index < 0 || index >= this.values.size()) {
			return null;
		}
		return this.values.remove(index);
	}
```



---

### Функция ID: 322

**Исходный код:**
```java
public boolean equalsIgnoreCase(final StrBuilder other) {
        if (this == other) {
            return true;
        }
        if (this.size != other.size) {
            return false;
        }
        final char[] thisBuf = this.buffer;
        final char[] otherBuf = other.buffer;
        for (int i = size - 1; i >= 0; i--) {
            final char c1 = thisBuf[i];
            final char c2 = otherBuf[i];
            if (c1 != c2 && Character.toUpperCase(c1) != Character.toUpperCase(c2)) {
                return false;
            }
        }
        return true;
    }
```



---

### Функция ID: 323

**Исходный код:**
```java
@SuppressWarnings("unchecked")
	private void extract(String name, Map<String, Object> result, Object value) {
		if (value instanceof Map<?, ?> map) {
			if (CollectionUtils.isEmpty(map)) {
				result.put(name, value);
				return;
			}
			flatten(name, result, (Map<String, Object>) value);
		}
		else if (value instanceof Collection<?> collection) {
			if (CollectionUtils.isEmpty(collection)) {
				result.put(name, value);
				return;
			}
			int index = 0;
			for (Object object : collection) {
				extract(name + "[" + index + "]", result, object);
				index++;
			}
		}
		else {
			result.put(name, value);
		}
	}
```



---

### Функция ID: 324

**Исходный код:**
```java
private static long parse(final String initialInput, final String normalized, final String suffix, String settingName) {
        final String s = normalized.substring(0, normalized.length() - suffix.length()).trim();
        try {
            final long value = Long.parseLong(s);
            if (value < -1) {
                // -1 is magic, but reject any other negative values
                throw new IllegalArgumentException(
                    "failed to parse setting ["
                        + settingName
                        + "] with value ["
                        + initialInput
                        + "] as a time value: negative durations are not supported"
                );
            }
            return value;
        } catch (final NumberFormatException e) {
            try {
                @SuppressWarnings("unused")
                final double ignored = Double.parseDouble(s);
                throw new IllegalArgumentException("failed to parse [" + initialInput + "], fractional time values are not supported", e);
            } catch (final NumberFormatException ignored) {
                throw new IllegalArgumentException("failed to parse [" + initialInput + "]", e);
            }
        }
    }
```



---

### Функция ID: 325

**Исходный код:**
```java
public boolean equals(final StrBuilder other) {
        if (this == other) {
            return true;
        }
        if (other == null) {
            return false;
        }
        if (this.size != other.size) {
            return false;
        }
        final char[] thisBuf = this.buffer;
        final char[] otherBuf = other.buffer;
        for (int i = size - 1; i >= 0; i--) {
            if (thisBuf[i] != otherBuf[i]) {
                return false;
            }
        }
        return true;
    }
```



---

### Функция ID: 326

**Исходный код:**
```java
private static int hashMember(final String name, final Object value) {
        final int part1 = name.hashCode() * 127;
        if (ObjectUtils.isArray(value)) {
            return part1 ^ arrayMemberHash(value.getClass().getComponentType(), value);
        }
        if (value instanceof Annotation) {
            return part1 ^ hashCode((Annotation) value);
        }
        return part1 ^ value.hashCode();
    }
```



---

### Функция ID: 327

**Исходный код:**
```python
def _called_with_wrong_args(f: t.Callable[..., Flask]) -> bool:
    """Check whether calling a function raised a ``TypeError`` because
    the call failed or because something in the factory raised the
    error.

    :param f: The function that was called.
    :return: ``True`` if the call failed.
    """
    tb = sys.exc_info()[2]

    try:
        while tb is not None:
            if tb.tb_frame.f_code is f.__code__:
                # In the function, it was called successfully.
                return False

            tb = tb.tb_next

        # Didn't reach the function.
        return True
    finally:
        # Delete tb to break a circular reference.
        # https://docs.python.org/2/library/sys.html#sys.exc_info
        del tb
```



---

### Функция ID: 328

**Исходный код:**
```java
String toString(ToStringFormat format, boolean upperCase) {
		String string = this.string[format.ordinal()];
		if (string == null) {
			string = buildToString(format);
			this.string[format.ordinal()] = string;
		}
		return (!upperCase) ? string : string.toUpperCase(Locale.ENGLISH);
	}
```



---

### Функция ID: 329

**Исходный код:**
```java
private boolean isNotInGroup() {
        return state == MemberState.UNSUBSCRIBED ||
            state == MemberState.FENCED ||
            state == MemberState.FATAL ||
            state == MemberState.STALE;
    }
```



---

### Функция ID: 330

**Исходный код:**
```python
def sem(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series | Any:
        """
        Return unbiased standard error of the mean over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Parameters
        ----------
        axis : {index (0), columns (1)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.sem with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.
        **kwargs :
            Additional keywords passed.

        Returns
        -------
        Series or DataFrame (if level specified)
            Unbiased standard error of the mean over requested axis.

        See Also
        --------
        DataFrame.var : Return unbiased variance over requested axis.
        DataFrame.std : Returns sample standard deviation over requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.sem().round(6)
        0.57735

        With a DataFrame

        >>> df = pd.DataFrame({"a": [1, 2], "b": [2, 3]}, index=["tiger", "zebra"])
        >>> df
               a   b
        tiger  1   2
        zebra  2   3
        >>> df.sem()
        a   0.5
        b   0.5
        dtype: float64

        Using axis=1

        >>> df.sem(axis=1)
        tiger   0.5
        zebra   0.5
        dtype: float64

        In this case, `numeric_only` should be set to `True`
        to avoid getting an error.

        >>> df = pd.DataFrame({"a": [1, 2], "b": ["T", "Z"]}, index=["tiger", "zebra"])
        >>> df.sem(numeric_only=True)
        a   0.5
        dtype: float64
        """
        result = super().sem(
            axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs
        )
        if isinstance(result, Series):
            result = result.__finalize__(self, method="sem")
        return result
```



---

### Функция ID: 331

**Исходный код:**
```python
def get_s3_bucket_key(
        bucket: str | None, key: str, bucket_param_name: str, key_param_name: str
    ) -> tuple[str, str]:
        """
        Get the S3 bucket name and key.

        From either:
        - bucket name and key. Return the info as it is after checking `key` is a relative path.
        - key. Must be a full s3:// url.

        :param bucket: The S3 bucket name
        :param key: The S3 key
        :param bucket_param_name: The parameter name containing the bucket name
        :param key_param_name: The parameter name containing the key name
        :return: the parsed bucket name and key
        """
        if bucket is None:
            return S3Hook.parse_s3_url(key)

        parsed_url = urlsplit(key)
        if parsed_url.scheme != "" or parsed_url.netloc != "":
            raise TypeError(
                f"If `{bucket_param_name}` is provided, {key_param_name} should be a relative path "
                "from root level, rather than a full s3:// url"
            )
        return bucket, key
```



---

### Функция ID: 332

**Исходный код:**
```java
public static <E extends Throwable> boolean getAsBoolean(final FailableBooleanSupplier<E> supplier) {
        try {
            return supplier.getAsBoolean();
        } catch (final Throwable t) {
            throw rethrow(t);
        }
    }
```



---

### Функция ID: 333

**Исходный код:**
```python
def set_flags(
        self,
        *,
        copy: bool | lib.NoDefault = lib.no_default,
        allows_duplicate_labels: bool | None = None,
    ) -> Self:
        """
        Return a new object with updated flags.

        This method creates a shallow copy of the original object, preserving its
        underlying data while modifying its global flags. In particular, it allows
        you to update properties such as whether duplicate labels are permitted. This
        behavior is especially useful in method chains, where one wishes to
        adjust DataFrame or Series characteristics without altering the original object.

        Parameters
        ----------
        copy : bool, default False
            This keyword is now ignored; changing its value will have no
            impact on the method.

            .. deprecated:: 3.0.0

                This keyword is ignored and will be removed in pandas 4.0. Since
                pandas 3.0, this method always returns a new object using a lazy
                copy mechanism that defers copies until necessary
                (Copy-on-Write). See the `user guide on Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                for more details.

        allows_duplicate_labels : bool, optional
            Whether the returned object allows duplicate labels.

        Returns
        -------
        Series or DataFrame
            The same type as the caller.

        See Also
        --------
        DataFrame.attrs : Global metadata applying to this dataset.
        DataFrame.flags : Global flags applying to this object.

        Notes
        -----
        This method returns a new object that's a view on the same data
        as the input. Mutating the input or the output values will be reflected
        in the other.

        This method is intended to be used in method chains.

        "Flags" differ from "metadata". Flags reflect properties of the
        pandas object (the Series or DataFrame). Metadata refer to properties
        of the dataset, and should be stored in :attr:`DataFrame.attrs`.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> df.flags.allows_duplicate_labels
        True
        >>> df2 = df.set_flags(allows_duplicate_labels=False)
        >>> df2.flags.allows_duplicate_labels
        False
        """
        self._check_copy_deprecation(copy)
        df = self.copy(deep=False)
        if allows_duplicate_labels is not None:
            df.flags["allows_duplicate_labels"] = allows_duplicate_labels
        return df
```



---

### Функция ID: 334

**Исходный код:**
```java
public static <E extends Throwable> int getAsInt(final FailableIntSupplier<E> supplier) {
        try {
            return supplier.getAsInt();
        } catch (final Throwable t) {
            throw rethrow(t);
        }
    }
```



---

### Функция ID: 335

**Исходный код:**
```java
boolean hasUserSuppliedInterfaces() {
		for (Class<?> ifc : this.interfaces) {
			if (!SpringProxy.class.isAssignableFrom(ifc) && !isAdvisorIntroducedInterface(ifc)) {
				return true;
			}
		}
		return false;
	}
```



---

### Функция ID: 336

**Исходный код:**
```java
public MemoryRecords build() {
        if (aborted) {
            throw new IllegalStateException("Attempting to build an aborted record batch");
        }
        close();
        return builtRecords;
    }
```



---

### Функция ID: 337

**Исходный код:**
```java
private static String readFile(final String envVarFile, final String key) {
        try {
            final byte[] bytes = Files.readAllBytes(Paths.get(envVarFile));
            final String content = new String(bytes, Charset.defaultCharset());
            // Split by null byte character
            final String[] lines = content.split(String.valueOf(CharUtils.NUL));
            final String prefix = key + "=";
            // @formatter:off
            return Arrays.stream(lines)
                    .filter(line -> line.startsWith(prefix))
                    .map(line -> line.split("=", 2))
                    .map(keyValue -> keyValue[1])
                    .findFirst()
                    .orElse(null);
            // @formatter:on
        } catch (final IOException e) {
            return null;
        }
    }
```



---

### Функция ID: 338

**Исходный код:**
```java
@SafeVarargs
    public static <T> T mode(final T... items) {
        if (ArrayUtils.isNotEmpty(items)) {
            final HashMap<T, MutableInt> occurrences = new HashMap<>(items.length);
            for (final T t : items) {
                ArrayUtils.increment(occurrences, t);
            }
            T result = null;
            int max = 0;
            for (final Map.Entry<T, MutableInt> e : occurrences.entrySet()) {
                final int cmp = e.getValue().intValue();
                if (cmp == max) {
                    result = null;
                } else if (cmp > max) {
                    max = cmp;
                    result = e.getKey();
                }
            }
            return result;
        }
        return null;
    }
```



---

### Функция ID: 339

**Исходный код:**
```java
@Override
	public String toString() {
		StringBuilder builder = new StringBuilder(getOrDeduceName(this));
		if (this.servletNames.isEmpty() && this.urlPatterns.isEmpty()) {
			builder.append(" urls=").append(Arrays.toString(DEFAULT_URL_MAPPINGS));
		}
		else {
			if (!this.servletNames.isEmpty()) {
				builder.append(" servlets=").append(this.servletNames);
			}
			if (!this.urlPatterns.isEmpty()) {
				builder.append(" urls=").append(this.urlPatterns);
			}
		}
		builder.append(" order=").append(getOrder());
		return builder.toString();
	}
```



---

### Функция ID: 340

**Исходный код:**
```python
def shift(
        self,
        periods: int | Sequence[int] = 1,
        freq=None,
        fill_value=lib.no_default,
        suffix: str | None = None,
    ):
        """
        Shift each group by periods observations.

        If freq is passed, the index will be increased using the periods and the freq.

        Parameters
        ----------
        periods : int | Sequence[int], default 1
            Number of periods to shift. If a list of values, shift each group by
            each period.
        freq : str, optional
            Frequency string.
        fill_value : optional
            The scalar value to use for newly introduced missing values.

            .. versionchanged:: 2.1.0
                Will raise a ``ValueError`` if ``freq`` is provided too.

        suffix : str, optional
            A string to add to each shifted column if there are multiple periods.
            Ignored otherwise.

        Returns
        -------
        Series or DataFrame
            Object shifted within each group.

        See Also
        --------
        Index.shift : Shift values of Index.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b", "b"]
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).shift(1)
        a    NaN
        a    1.0
        b    NaN
        b    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(
        ...     data,
        ...     columns=["a", "b", "c"],
        ...     index=["tuna", "salmon", "catfish", "goldfish"],
        ... )
        >>> df
                   a  b  c
            tuna   1  2  3
          salmon   1  5  6
         catfish   2  5  8
        goldfish   2  6  9
        >>> df.groupby("a").shift(1)
                      b    c
            tuna    NaN  NaN
          salmon    2.0  3.0
         catfish    NaN  NaN
        goldfish    5.0  8.0
        """
        if is_list_like(periods):
            periods = cast(Sequence, periods)
            if len(periods) == 0:
                raise ValueError("If `periods` is an iterable, it cannot be empty.")
            from pandas.core.reshape.concat import concat

            add_suffix = True
        else:
            if not is_integer(periods):
                raise TypeError(
                    f"Periods must be integer, but {periods} is {type(periods)}."
                )
            if suffix:
                raise ValueError("Cannot specify `suffix` if `periods` is an int.")
            periods = [cast(int, periods)]
            add_suffix = False

        shifted_dataframes = []
        for period in periods:
            if not is_integer(period):
                raise TypeError(
                    f"Periods must be integer, but {period} is {type(period)}."
                )
            period = cast(int, period)
            if freq is not None:
                f = lambda x: x.shift(
                    period,
                    freq,
                    0,  # axis
                    fill_value,
                )
                shifted = self._python_apply_general(
                    f, self._selected_obj, is_transform=True
                )
            else:
                if fill_value is lib.no_default:
                    fill_value = None
                ids = self._grouper.ids
                ngroups = self._grouper.ngroups
                res_indexer = np.zeros(len(ids), dtype=np.int64)

                libgroupby.group_shift_indexer(res_indexer, ids, ngroups, period)

                obj = self._obj_with_exclusions

                shifted = obj._reindex_with_indexers(
                    {0: (obj.index, res_indexer)},
                    fill_value=fill_value,
                    allow_dups=True,
                )

            if add_suffix:
                if isinstance(shifted, Series):
                    shifted = cast(NDFrameT, shifted.to_frame())
                shifted = shifted.add_suffix(
                    f"{suffix}_{period}" if suffix else f"_{period}"
                )
            shifted_dataframes.append(cast(Union[Series, DataFrame], shifted))

        return (
            shifted_dataframes[0]
            if len(shifted_dataframes) == 1
            else concat(shifted_dataframes, axis=1, sort=False)
        )
```



---

### Функция ID: 341

**Исходный код:**
```java
public double getDouble(String name) throws JSONException {
		Object object = get(name);
		Double result = JSON.toDouble(object);
		if (result == null) {
			throw JSON.typeMismatch(name, object, "double");
		}
		return result;
	}
```



---

### Функция ID: 342

**Исходный код:**
```java
@Override
    public Iterator<Record> iterator() {
        if (count() == 0)
            return Collections.emptyIterator();

        if (!isCompressed())
            return uncompressedIterator();

        // for a normal iterator, we cannot ensure that the underlying compression stream is closed,
        // so we decompress the full record set here. Use cases which call for a lower memory footprint
        // can use `streamingIterator` at the cost of additional complexity
        try (CloseableIterator<Record> iterator = compressedIterator(BufferSupplier.NO_CACHING, false)) {
            List<Record> records = new ArrayList<>(count());
            while (iterator.hasNext())
                records.add(iterator.next());
            return records.iterator();
        }
    }
```



---

### Функция ID: 343

**Исходный код:**
```java
private Options copy(Consumer<EnumSet<Option>> processor) {
			EnumSet<Option> options = (!this.options.isEmpty()) ? EnumSet.copyOf(this.options)
					: EnumSet.noneOf(Option.class);
			processor.accept(options);
			return new Options(options);
		}
```



---

### Функция ID: 344

**Исходный код:**
```java
public static List<Throwable> getThrowableList(Throwable throwable) {
        final List<Throwable> list = new ArrayList<>();
        while (throwable != null && !list.contains(throwable)) {
            list.add(throwable);
            throwable = throwable.getCause();
        }
        return list;
    }
```



---

### Функция ID: 345

**Исходный код:**
```cpp
static INLINE OPJ_INT32 opj_int_fix_mul(OPJ_INT32 a, OPJ_INT32 b)
{
#if defined(_MSC_VER) && (_MSC_VER >= 1400) && !defined(__INTEL_COMPILER) && defined(_M_IX86)
    OPJ_INT64 temp = __emul(a, b);
#else
    OPJ_INT64 temp = (OPJ_INT64) a * (OPJ_INT64) b ;
#endif
    temp += 4096;
    assert((temp >> 13) <= (OPJ_INT64)0x7FFFFFFF);
    assert((temp >> 13) >= (-(OPJ_INT64)0x7FFFFFFF - (OPJ_INT64)1));
    return (OPJ_INT32)(temp >> 13);
}
```



---

### Функция ID: 346

**Исходный код:**
```java
public static Object toPrimitive(final Object array) {
        if (array == null) {
            return null;
        }
        final Class<?> ct = array.getClass().getComponentType();
        final Class<?> pt = ClassUtils.wrapperToPrimitive(ct);
        if (Boolean.TYPE.equals(pt)) {
            return toPrimitive((Boolean[]) array);
        }
        if (Character.TYPE.equals(pt)) {
            return toPrimitive((Character[]) array);
        }
        if (Byte.TYPE.equals(pt)) {
            return toPrimitive((Byte[]) array);
        }
        if (Integer.TYPE.equals(pt)) {
            return toPrimitive((Integer[]) array);
        }
        if (Long.TYPE.equals(pt)) {
            return toPrimitive((Long[]) array);
        }
        if (Short.TYPE.equals(pt)) {
            return toPrimitive((Short[]) array);
        }
        if (Double.TYPE.equals(pt)) {
            return toPrimitive((Double[]) array);
        }
        if (Float.TYPE.equals(pt)) {
            return toPrimitive((Float[]) array);
        }
        return array;
    }
```



---

### Функция ID: 347

**Исходный код:**
```java
public synchronized void requestOffsetResetIfPartitionAssigned(TopicPartition partition) {
        final TopicPartitionState state = assignedStateOrNull(partition);
        if (state != null) {
            state.reset(defaultResetStrategy);
        }
    }
```



---

### Функция ID: 348

**Исходный код:**
```java
public StrBuilder appendWithSeparators(final Iterable<?> iterable, final String separator) {
        if (iterable != null) {
            final String sep = Objects.toString(separator, "");
            final Iterator<?> it = iterable.iterator();
            while (it.hasNext()) {
                append(it.next());
                if (it.hasNext()) {
                    append(sep);
                }
            }
        }
        return this;
    }
```



---

### Функция ID: 349

**Исходный код:**
```java
public static <T extends Throwable> T throwUnchecked(final T throwable) {
        if (isUnchecked(throwable)) {
            throw asRuntimeException(throwable);
        }
        return throwable;
    }
```



---

### Функция ID: 350

**Исходный код:**
```java
protected final boolean isMethodOnIntroducedInterface(MethodInvocation mi) {
		Boolean rememberedResult = this.rememberedMethods.get(mi.getMethod());
		if (rememberedResult != null) {
			return rememberedResult;
		}
		else {
			// Work it out and cache it.
			boolean result = implementsInterface(mi.getMethod().getDeclaringClass());
			this.rememberedMethods.put(mi.getMethod(), result);
			return result;
		}
	}
```



---

### Функция ID: 351

**Исходный код:**
```java
@Override
    public boolean equals(final Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj == null || obj.getClass() != getClass()) {
            return false;
        }
        @SuppressWarnings("unchecked") // OK because we checked the class above
        final
        Range<T> range = (Range<T>) obj;
        return minimum.equals(range.minimum) &&
               maximum.equals(range.maximum);
    }
```



---

### Функция ID: 352

**Исходный код:**
```java
public static String repeat(final String repeat, final String separator, final int count) {
        if (repeat == null || separator == null) {
            return repeat(repeat, count);
        }
        // given that repeat(String, int) is quite optimized, better to rely on it than try and splice this into it
        final String result = repeat(repeat + separator, count);
        return Strings.CS.removeEnd(result, separator);
    }
```



---

### Функция ID: 353

**Исходный код:**
```java
public static List<Class<?>> getAllSuperclasses(final Class<?> cls) {
        if (cls == null) {
            return null;
        }
        final List<Class<?>> classes = new ArrayList<>();
        Class<?> superclass = cls.getSuperclass();
        while (superclass != null) {
            classes.add(superclass);
            superclass = superclass.getSuperclass();
        }
        return classes;
    }
```



---

### Функция ID: 354

**Исходный код:**
```python
def take(
        self,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> Self:
        """
        Return a new Index of the values selected by the indices.
        For internal compatibility with numpy arrays.

        Parameters
        ----------
        indices : array-like
            Indices to be taken.
        axis : {0 or 'index'}, optional
            The axis over which to select values, always 0 or 'index'.
        allow_fill : bool, default True
            How to handle negative values in `indices`.
            * False: negative values in `indices` indicate positional indices
                from the right (the default). This is similar to
                :func:`numpy.take`.
            * True: negative values in `indices` indicate
                missing values. These values are set to `fill_value`. Any other
                other negative values raise a ``ValueError``.
        fill_value : scalar, default None
            If allow_fill=True and fill_value is not None, indices specified by
            -1 are regarded as NA. If Index doesn't hold NA, raise ValueError.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        Index
            An index formed of elements at the given indices. Will be the same
            type as self, except for RangeIndex.

        See Also
        --------
        numpy.ndarray.take: Return an array formed from the
            elements of a at the given indices.

        Examples
        --------
        >>> idx = pd.Index(["a", "b", "c"])
        >>> idx.take([2, 2, 1, 2])
        Index(['c', 'c', 'b', 'c'], dtype='str')
        """
        nv.validate_take((), kwargs)
        indices = np.asarray(indices, dtype=np.intp)

        result = NDArrayBackedExtensionIndex.take(
            self, indices, axis, allow_fill, fill_value, **kwargs
        )

        maybe_slice = lib.maybe_indices_to_slice(indices, len(self))
        if isinstance(maybe_slice, slice):
            freq = self._data._get_getitem_freq(maybe_slice)
            result._data._freq = freq
        return result
```



---

### Функция ID: 355

**Исходный код:**
```java
@Contract("!null -> !null")
	static @Nullable List<X509Certificate> parse(@Nullable String text) {
		if (text == null) {
			return null;
		}
		CertificateFactory factory = getCertificateFactory();
		List<X509Certificate> certs = new ArrayList<>();
		readCertificates(text, factory, certs::add);
		Assert.state(!CollectionUtils.isEmpty(certs), "Missing certificates or unrecognized format");
		return List.copyOf(certs);
	}
```



---

### Функция ID: 356

**Исходный код:**
```python
def calculate_number_of_dag_runs(performance_dag_conf: dict[str, str]) -> int:
    """
    Calculate how many Dag Runs will be created with given performance DAG configuration.

    :param performance_dag_conf: dict with environment variables as keys and their values as values

    :return: total number of Dag Runs
    :rtype: int
    """
    max_runs = get_performance_dag_environment_variable(performance_dag_conf, "PERF_MAX_RUNS")

    total_dags_count = get_dags_count(performance_dag_conf)

    # if PERF_MAX_RUNS is missing from the configuration,
    # it means that PERF_SCHEDULE_INTERVAL must be '@once'
    if max_runs is None:
        return total_dags_count

    return int(max_runs) * total_dags_count
```



---

### Функция ID: 357

**Исходный код:**
```java
private Object @Nullable [] getArgs(Class<?>[] parameterTypes) {
		Object[] args = new Object[parameterTypes.length];
		for (int i = 0; i < parameterTypes.length; i++) {
			Function<Class<?>, Object> parameter = getAvailableParameter(parameterTypes[i]);
			if (parameter == null) {
				return null;
			}
			args[i] = parameter.apply(this.type);
		}
		return args;
	}
```



---

### Функция ID: 358

**Исходный код:**
```python
def _stata_elapsed_date_to_datetime_vec(dates: Series, fmt: str) -> Series:
    """
    Convert from SIF to datetime. https://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    dates : Series
        The Stata Internal Format date to convert to datetime according to fmt
    fmt : str
        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty
        Returns

    Returns
    -------
    converted : Series
        The converted dates

    Examples
    --------
    >>> dates = pd.Series([52])
    >>> _stata_elapsed_date_to_datetime_vec(dates, "%tw")
    0   1961-01-01
    dtype: datetime64[s]

    Notes
    -----
    datetime/c - tc
        milliseconds since 01jan1960 00:00:00.000, assuming 86,400 s/day
    datetime/C - tC - NOT IMPLEMENTED
        milliseconds since 01jan1960 00:00:00.000, adjusted for leap seconds
    date - td
        days since 01jan1960 (01jan1960 = 0)
    weekly date - tw
        weeks since 1960w1
        This assumes 52 weeks in a year, then adds 7 * remainder of the weeks.
        The datetime value is the start of the week in terms of days in the
        year, not ISO calendar weeks.
    monthly date - tm
        months since 1960m1
    quarterly date - tq
        quarters since 1960q1
    half-yearly date - th
        half-years since 1960h1 yearly
    date - ty
        years since 0000
    """

    if fmt.startswith(("%tc", "tc")):
        # Delta ms relative to base
        td = np.timedelta64(stata_epoch - unix_epoch, "ms")
        res = np.array(dates._values, dtype="M8[ms]") + td
        return Series(res, index=dates.index)

    elif fmt.startswith(("%td", "td", "%d", "d")):
        # Delta days relative to base
        td = np.timedelta64(stata_epoch - unix_epoch, "D")
        res = np.array(dates._values, dtype="M8[D]") + td
        return Series(res, index=dates.index)

    elif fmt.startswith(("%tm", "tm")):
        # Delta months relative to base
        ordinals = dates + (stata_epoch.year - unix_epoch.year) * 12
        res = np.array(ordinals, dtype="M8[M]").astype("M8[s]")
        return Series(res, index=dates.index)

    elif fmt.startswith(("%tq", "tq")):
        # Delta quarters relative to base
        ordinals = dates + (stata_epoch.year - unix_epoch.year) * 4
        res = np.array(ordinals, dtype="M8[3M]").astype("M8[s]")
        return Series(res, index=dates.index)

    elif fmt.startswith(("%th", "th")):
        # Delta half-years relative to base
        ordinals = dates + (stata_epoch.year - unix_epoch.year) * 2
        res = np.array(ordinals, dtype="M8[6M]").astype("M8[s]")
        return Series(res, index=dates.index)

    elif fmt.startswith(("%ty", "ty")):
        # Years -- not delta
        ordinals = dates - 1970
        res = np.array(ordinals, dtype="M8[Y]").astype("M8[s]")
        return Series(res, index=dates.index)

    bad_locs = np.isnan(dates)
    has_bad_values = False
    if bad_locs.any():
        has_bad_values = True
        dates._values[bad_locs] = 1.0  # Replace with NaT
    dates = dates.astype(np.int64)

    if fmt.startswith(("%tC", "tC")):
        warnings.warn(
            "Encountered %tC format. Leaving in Stata Internal Format.",
            stacklevel=find_stack_level(),
        )
        conv_dates = Series(dates, dtype=object)
        if has_bad_values:
            conv_dates[bad_locs] = NaT
        return conv_dates
    # does not count leap days - 7 days is a week.
    # 52nd week may have more than 7 days
    elif fmt.startswith(("%tw", "tw")):
        year = stata_epoch.year + dates // 52
        days = (dates % 52) * 7
        per_y = (year - 1970).array.view("Period[Y]")
        per_d = per_y.asfreq("D", how="S")
        per_d_shifted = per_d + days._values
        per_s = per_d_shifted.asfreq("s", how="S")
        conv_dates_arr = per_s.view("M8[s]")
        conv_dates = Series(conv_dates_arr, index=dates.index)

    else:
        raise ValueError(f"Date fmt {fmt} not understood")

    if has_bad_values:  # Restore NaT for bad values
        conv_dates[bad_locs] = NaT

    return conv_dates
```



---

### Функция ID: 359

**Исходный код:**
```java
private boolean remainderIsNotAlphanumeric(Elements elements, int element, int index) {
		if (elements.getType(element).isIndexed()) {
			return false;
		}
		int length = elements.getLength(element);
		do {
			char c = Character.toLowerCase(elements.charAt(element, index++));
			if (ElementsParser.isAlphaNumeric(c)) {
				return false;
			}
		}
		while (index < length);
		return true;
	}
```



---

### Функция ID: 360

**Исходный код:**
```java
public StrBuilder trim() {
        if (size == 0) {
            return this;
        }
        int len = size;
        final char[] buf = buffer;
        int pos = 0;
        while (pos < len && buf[pos] <= ' ') {
            pos++;
        }
        while (pos < len && buf[len - 1] <= ' ') {
            len--;
        }
        if (len < size) {
            delete(len, size);
        }
        if (pos > 0) {
            delete(0, pos);
        }
        return this;
    }
```



---

### Функция ID: 361

**Исходный код:**
```java
public int optInt(int index, int fallback) {
		Object object = opt(index);
		Integer result = JSON.toInteger(object);
		return result != null ? result : fallback;
	}
```



---

### Функция ID: 362

**Исходный код:**
```java
@Override
	public boolean equals(Object o) {
		if (o == null || getClass() != o.getClass()) {
			return false;
		}
		ItemIgnore that = (ItemIgnore) o;
		return this.type == that.type && Objects.equals(this.name, that.name);
	}
```



---

### Функция ID: 363

**Исходный код:**
```java
private static ValueAndPreviousValue getElementAtRank(ExponentialHistogram histo, long rank) {
        long negativeValuesCount = histo.negativeBuckets().valueCount();
        long zeroCount = histo.zeroBucket().count();
        if (rank < negativeValuesCount) {
            if (rank == 0) {
                return new ValueAndPreviousValue(Double.NaN, -getLastBucketMidpoint(histo.negativeBuckets()));
            } else {
                return getBucketMidpointForRank(histo.negativeBuckets().iterator(), negativeValuesCount - rank).negateAndSwap();
            }
        } else if (rank < (negativeValuesCount + zeroCount)) {
            if (rank == negativeValuesCount) {
                // the element at the previous rank falls into the negative bucket range
                return new ValueAndPreviousValue(-getFirstBucketMidpoint(histo.negativeBuckets()), 0.0);
            } else {
                return new ValueAndPreviousValue(0.0, 0.0);
            }
        } else {
            ValueAndPreviousValue result = getBucketMidpointForRank(
                histo.positiveBuckets().iterator(),
                rank - negativeValuesCount - zeroCount
            );
            if ((rank - 1) < negativeValuesCount) {
                // previous value falls into the negative bucket range or has rank -1 and therefore doesn't exist
                return new ValueAndPreviousValue(-getFirstBucketMidpoint(histo.negativeBuckets()), result.valueAtRank);
            } else if ((rank - 1) < (negativeValuesCount + zeroCount)) {
                // previous value falls into the zero bucket
                return new ValueAndPreviousValue(0.0, result.valueAtRank);
            } else {
                return result;
            }
        }
    }
```



---

### Функция ID: 364

**Исходный код:**
```java
protected Cache resolveCache(CacheOperationInvocationContext<O> context) {
		Collection<? extends Cache> caches = context.getOperation().getCacheResolver().resolveCaches(context);
		Cache cache = extractFrom(caches);
		if (cache == null) {
			throw new IllegalStateException("Cache could not have been resolved for " + context.getOperation());
		}
		return cache;
	}
```



---

### Функция ID: 365

**Исходный код:**
```java
Object cloneReset() throws CloneNotSupportedException {
        // this method exists to enable 100% test coverage
        final StrTokenizer cloned = (StrTokenizer) super.clone();
        if (cloned.chars != null) {
            cloned.chars = cloned.chars.clone();
        }
        cloned.reset();
        return cloned;
    }
```



---

### Функция ID: 366

**Исходный код:**
```java
boolean hasDashedElement() {
		Boolean hasDashedElement = this.hasDashedElement;
		if (hasDashedElement != null) {
			return hasDashedElement;
		}
		for (int i = 0; i < getNumberOfElements(); i++) {
			if (getElement(i, Form.DASHED).indexOf('-') != -1) {
				this.hasDashedElement = true;
				return true;
			}
		}
		this.hasDashedElement = false;
		return false;
	}
```



---

### Функция ID: 367

**Исходный код:**
```java
public JSONArray toJSONArray(JSONArray names) {
		JSONArray result = new JSONArray();
		if (names == null) {
			return null;
		}
		int length = names.length();
		if (length == 0) {
			return null;
		}
		for (int i = 0; i < length; i++) {
			String name = JSON.toString(names.opt(i));
			result.put(opt(name));
		}
		return result;
	}
```



---

### Функция ID: 368

**Исходный код:**
```java
static Long parseAsn(final String asn) {
        if (asn == null || Strings.hasText(asn) == false) {
            return null;
        } else {
            String stripped = asn.toUpperCase(Locale.ROOT).replaceAll("AS", "").trim();
            try {
                return Long.parseLong(stripped);
            } catch (NumberFormatException e) {
                logger.trace("Unable to parse non-compliant ASN string [{}]", asn);
                return null;
            }
        }
    }
```



---

### Функция ID: 369

**Исходный код:**
```java
public static <O, T extends Throwable> O get(final FailableSupplier<O, T> supplier) {
        try {
            return supplier.get();
        } catch (final Throwable t) {
            throw rethrow(t);
        }
    }
```



---

### Функция ID: 370

**Исходный код:**
```java
public JSONStringer value(boolean value) throws JSONException {
		if (this.stack.isEmpty()) {
			throw new JSONException("Nesting problem");
		}
		beforeValue();
		this.out.append(value);
		return this;
	}
```



---

### Функция ID: 371

**Исходный код:**
```java
public static List<Class<?>> getAllInterfaces(final Class<?> cls) {
        if (cls == null) {
            return null;
        }
        final LinkedHashSet<Class<?>> interfacesFound = new LinkedHashSet<>();
        getAllInterfaces(cls, interfacesFound);
        return new ArrayList<>(interfacesFound);
    }
```



---

### Функция ID: 372

**Исходный код:**
```java
public JSONStringer value(long value) throws JSONException {
		if (this.stack.isEmpty()) {
			throw new JSONException("Nesting problem");
		}
		beforeValue();
		this.out.append(value);
		return this;
	}
```



---

### Функция ID: 373

**Исходный код:**
```java
@Override
    public ConfigData get(String path, Set<String> keys) {
        return get(path, pathname ->
                Files.isRegularFile(pathname)
                        && keys.contains(pathname.getFileName().toString()));
    }
```



---

### Функция ID: 374

**Исходный код:**
```java
public Fraction abs() {
        if (numerator >= 0) {
            return this;
        }
        return negate();
    }
```



---

### Функция ID: 375

**Исходный код:**
```python
def reorder_levels(self, order: Sequence[int | str], axis: Axis = 0) -> DataFrame:
        """
        Rearrange index or column levels using input ``order``.

        May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int or list of str
            List representing new level order. Reference level by number
            (position) or by key (label).
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Where to reorder levels.

        Returns
        -------
        DataFrame
            DataFrame with indices or columns with reordered levels.

        See Also
        --------
            DataFrame.swaplevel : Swap levels i and j in a MultiIndex.

        Examples
        --------
        >>> data = {
        ...     "class": ["Mammals", "Mammals", "Reptiles"],
        ...     "diet": ["Omnivore", "Carnivore", "Carnivore"],
        ...     "species": ["Humans", "Dogs", "Snakes"],
        ... }
        >>> df = pd.DataFrame(data, columns=["class", "diet", "species"])
        >>> df = df.set_index(["class", "diet"])
        >>> df
                                          species
        class      diet
        Mammals    Omnivore                Humans
                   Carnivore                 Dogs
        Reptiles   Carnivore               Snakes

        Let's reorder the levels of the index:

        >>> df.reorder_levels(["diet", "class"])
                                          species
        diet      class
        Omnivore  Mammals                  Humans
        Carnivore Mammals                    Dogs
                  Reptiles                 Snakes
        """
        axis = self._get_axis_number(axis)
        if not isinstance(self._get_axis(axis), MultiIndex):  # pragma: no cover
            raise TypeError("Can only reorder levels on a hierarchical axis.")

        result = self.copy(deep=False)

        if axis == 0:
            assert isinstance(result.index, MultiIndex)
            result.index = result.index.reorder_levels(order)
        else:
            assert isinstance(result.columns, MultiIndex)
            result.columns = result.columns.reorder_levels(order)
        return result
```



---

### Функция ID: 376

**Исходный код:**
```java
private static boolean equals(final Type[] type1, final Type[] type2) {
        if (type1.length == type2.length) {
            for (int i = 0; i < type1.length; i++) {
                if (!equals(type1[i], type2[i])) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }
```



---

### Функция ID: 377

**Исходный код:**
```java
public void setShareFetchAction(final ShareFetchBuffer fetchBuffer) {
        final AtomicBoolean throwWakeupException = new AtomicBoolean(false);
        pendingTask.getAndUpdate(task -> {
            if (task == null) {
                return new ShareFetchAction(fetchBuffer);
            } else if (task instanceof WakeupFuture) {
                throwWakeupException.set(true);
                return null;
            } else if (task instanceof DisabledWakeups) {
                return task;
            }
            // last active state is still active
            throw new IllegalStateException("Last active task is still active");
        });
        if (throwWakeupException.get()) {
            throw new WakeupException();
        }
    }
```



---

### Функция ID: 378

**Исходный код:**
```java
public TimestampAndOffset searchForTimestamp(long targetTimestamp, int startingPosition, long startingOffset) {
        for (RecordBatch batch : batchesFrom(startingPosition)) {
            if (batch.maxTimestamp() >= targetTimestamp) {
                // We found a message
                for (Record record : batch) {
                    long timestamp = record.timestamp();
                    if (timestamp >= targetTimestamp && record.offset() >= startingOffset)
                        return new TimestampAndOffset(timestamp, record.offset(),
                                maybeLeaderEpoch(batch.partitionLeaderEpoch()));
                }
            }
        }
        return null;
    }
```



---

### Функция ID: 379

**Исходный код:**
```java
public String replace(final CharSequence source, final int offset, final int length) {
        if (source == null) {
            return null;
        }
        final StrBuilder buf = new StrBuilder(length).append(source, offset, length);
        substitute(buf, 0, length);
        return buf.toString();
    }
```



---

### Функция ID: 380

**Исходный код:**
```java
private ClassicHttpResponse execute(HttpUriRequest request, URI url, String description) {
		try {
			HttpHost host = HttpHost.create(url);
			request.addHeader("User-Agent", "SpringBootCli/" + getClass().getPackage().getImplementationVersion());
			return getHttp().executeOpen(host, request, null);
		}
		catch (IOException ex) {
			throw new ReportableException(
					"Failed to " + description + " from service at '" + url + "' (" + ex.getMessage() + ")");
		}
	}
```



---

### Функция ID: 381

**Исходный код:**
```java
private String findPropertySource(MutablePropertySources sources) {
		if (ClassUtils.isPresent(SERVLET_ENVIRONMENT_CLASS, null)) {
			PropertySource<?> servletPropertySource = sources.stream()
				.filter((source) -> SERVLET_ENVIRONMENT_PROPERTY_SOURCES.contains(source.getName()))
				.findFirst()
				.orElse(null);
			if (servletPropertySource != null) {
				return servletPropertySource.getName();
			}
		}
		return StandardEnvironment.SYSTEM_PROPERTIES_PROPERTY_SOURCE_NAME;
	}
```



---

### Функция ID: 382

**Исходный код:**
```java
public Set<String> outputKeys() {
        Set<String> result = new LinkedHashSet<>(matchPairs.size());
        for (DissectPair matchPair : matchPairs) {
            if (matchPair.key.getModifier() != DissectKey.Modifier.NAMED_SKIP) {
                result.add(matchPair.key.getName());
            }
        }
        return result;
    }
```



---

### Функция ID: 383

**Исходный код:**
```java
@Override
    public long read(ByteBuffer[] dsts, int offset, int length) throws IOException {
        if ((offset < 0) || (length < 0) || (offset > dsts.length - length))
            throw new IndexOutOfBoundsException();

        int totalRead = 0;
        int i = offset;
        while (i < offset + length) {
            if (dsts[i].hasRemaining()) {
                int read = read(dsts[i]);
                if (read > 0)
                    totalRead += read;
                else
                    break;
            }
            if (!dsts[i].hasRemaining()) {
                i++;
            }
        }
        return totalRead;
    }
```



---

### Функция ID: 384

**Исходный код:**
```typescript
function onResolve(fillers: Fillers, args: esbuild.OnResolveArgs, namespace: string): esbuild.OnResolveResult {
  // removes trailing slashes in imports paths
  const path = args.path.replace(/\/$/, '')
  const item = fillers[path]

  // if a path is provided, we just replace it
  if (item.imports !== undefined) {
    return { path: item.imports }
  }

  // if not, we defer action to the loaders cb
  return {
    namespace,
    path: path,
    pluginData: args.importer,
  }
}
```



---

### Функция ID: 385

**Исходный код:**
```java
@CanIgnoreReturnValue
  public final double getAndAdd(int i, double delta) {
    while (true) {
      long current = longs.get(i);
      double currentVal = longBitsToDouble(current);
      double nextVal = currentVal + delta;
      long next = doubleToRawLongBits(nextVal);
      if (longs.compareAndSet(i, current, next)) {
        return currentVal;
      }
    }
  }
```



---

### Функция ID: 386

**Исходный код:**
```java
public int optInt(String name, int fallback) {
		Object object = opt(name);
		Integer result = JSON.toInteger(object);
		return result != null ? result : fallback;
	}
```



---

### Функция ID: 387

**Исходный код:**
```java
private Deque<ProducerBatch> splitRecordsIntoBatches(RecordBatch recordBatch, int splitBatchSize) {
        Deque<ProducerBatch> batches = new ArrayDeque<>();
        Iterator<Thunk> thunkIter = thunks.iterator();
        // We always allocate batch size because we are already splitting a big batch.
        // And we also Retain the create time of the original batch.
        ProducerBatch batch = null;

        for (Record record : recordBatch) {
            assert thunkIter.hasNext();
            Thunk thunk = thunkIter.next();
            if (batch == null)
                batch = createBatchOffAccumulatorForRecord(record, splitBatchSize);

            // A newly created batch can always host the first message.
            if (!batch.tryAppendForSplit(record.timestamp(), record.key(), record.value(), record.headers(), thunk)) {
                batches.add(batch);
                batch.closeForRecordAppends();
                batch = createBatchOffAccumulatorForRecord(record, splitBatchSize);
                batch.tryAppendForSplit(record.timestamp(), record.key(), record.value(), record.headers(), thunk);
            }
        }

        // Close the last batch and add it to the batch list after split.
        if (batch != null) {
            batches.add(batch);
            batch.closeForRecordAppends();
        }

        return batches;
    }
```



---

### Функция ID: 388

**Исходный код:**
```java
public void setFetchAction(final FetchBuffer fetchBuffer) {
        final AtomicBoolean throwWakeupException = new AtomicBoolean(false);
        pendingTask.getAndUpdate(task -> {
            if (task == null) {
                return new FetchAction(fetchBuffer);
            } else if (task instanceof WakeupFuture) {
                throwWakeupException.set(true);
                return null;
            } else if (task instanceof DisabledWakeups) {
                return task;
            }
            // last active state is still active
            throw new IllegalStateException("Last active task is still active");
        });
        if (throwWakeupException.get()) {
            throw new WakeupException();
        }
    }
```



---

### Функция ID: 389

**Исходный код:**
```java
private static <T extends Throwable> boolean getAsBoolean(final FailableBooleanSupplier<T> supplier) {
        try {
            return supplier.getAsBoolean();
        } catch (final Throwable t) {
            throw rethrow(t);
        }
    }
```



---

### Функция ID: 390

**Исходный код:**
```java
void handleCoordinatorReady() {
        NodeApiVersions nodeApiVersions = transactionCoordinator != null ?
                apiVersions.get(transactionCoordinator.idString()) :
                null;
        ApiVersion initProducerIdVersion = nodeApiVersions != null ?
                nodeApiVersions.apiVersion(ApiKeys.INIT_PRODUCER_ID) :
                null;
        this.coordinatorSupportsBumpingEpoch = initProducerIdVersion != null &&
                initProducerIdVersion.maxVersion() >= 3;
    }
```



---

### Функция ID: 391

**Исходный код:**
```java
public StrBuilder appendAll(final Iterator<?> it) {
        if (it != null) {
            it.forEachRemaining(this::append);
        }
        return this;
    }
```



---

### Функция ID: 392

**Исходный код:**
```java
private String buildDefaultToString() {
		if (this.elements.canShortcutWithSource(ElementType.UNIFORM, ElementType.DASHED)) {
			return this.elements.getSource().toString();
		}
		int elements = getNumberOfElements();
		StringBuilder result = new StringBuilder(elements * 8);
		for (int i = 0; i < elements; i++) {
			boolean indexed = isIndexed(i);
			if (!result.isEmpty() && !indexed) {
				result.append('.');
			}
			if (indexed) {
				result.append('[');
				result.append(getElement(i, Form.ORIGINAL));
				result.append(']');
			}
			else {
				result.append(getElement(i, Form.DASHED));
			}
		}
		return result.toString();
	}
```



---

### Функция ID: 393

**Исходный код:**
```python
def from_codes(
        cls,
        codes,
        categories=None,
        ordered=None,
        dtype: Dtype | None = None,
        validate: bool = True,
    ) -> Self:
        """
        Make a Categorical type from codes and categories or dtype.

        This constructor is useful if you already have codes and
        categories/dtype and so do not need the (computation intensive)
        factorization step, which is usually done on the constructor.

        If your data does not follow this convention, please use the normal
        constructor.

        Parameters
        ----------
        codes : array-like of int
            An integer array, where each integer points to a category in
            categories or dtype.categories, or else is -1 for NaN.
        categories : index-like, optional
            The categories for the categorical. Items need to be unique.
            If the categories are not given here, then they must be provided
            in `dtype`.
        ordered : bool, optional
            Whether or not this categorical is treated as an ordered
            categorical. If not given here or in `dtype`, the resulting
            categorical will be unordered.
        dtype : CategoricalDtype or "category", optional
            If :class:`CategoricalDtype`, cannot be used together with
            `categories` or `ordered`.
        validate : bool, default True
            If True, validate that the codes are valid for the dtype.
            If False, don't validate that the codes are valid. Be careful about skipping
            validation, as invalid codes can lead to severe problems, such as segfaults.

            .. versionadded:: 2.1.0

        Returns
        -------
        Categorical

        See Also
        --------
        codes : The category codes of the categorical.
        CategoricalIndex : An Index with an underlying ``Categorical``.

        Examples
        --------
        >>> dtype = pd.CategoricalDtype(["a", "b"], ordered=True)
        >>> pd.Categorical.from_codes(codes=[0, 1, 0, 1], dtype=dtype)
        ['a', 'b', 'a', 'b']
        Categories (2, str): ['a' < 'b']
        """
        dtype = CategoricalDtype._from_values_or_dtype(
            categories=categories, ordered=ordered, dtype=dtype
        )
        if dtype.categories is None:
            msg = (
                "The categories must be provided in 'categories' or "
                "'dtype'. Both were None."
            )
            raise ValueError(msg)

        if validate:
            # beware: non-valid codes may segfault
            codes = cls._validate_codes_for_dtype(codes, dtype=dtype)

        return cls._simple_new(codes, dtype=dtype)
```



---

### Функция ID: 394

**Исходный код:**
```python
def _get_relative_fileloc(self, filepath: str) -> str:
        """
        Get the relative file location for a given filepath.

        :param filepath: Absolute path to the file
        :return: Relative path from bundle_path, or original filepath if no bundle_path
        """
        if self.bundle_path:
            return str(Path(filepath).relative_to(self.bundle_path))
        return filepath
```



---

### Функция ID: 395

**Исходный код:**
```java
private boolean endsWithElementsEqualTo(ConfigurationPropertyName name) {
		for (int i = this.elements.getSize() - 1; i >= 0; i--) {
			if (elementDiffers(this.elements, name.elements, i)) {
				return false;
			}
		}
		return true;
	}
```



---

### Функция ID: 396

**Исходный код:**
```java
public double getDouble(int index) throws JSONException {
		Object object = get(index);
		Double result = JSON.toDouble(object);
		if (result == null) {
			throw JSON.typeMismatch(index, object, "double");
		}
		return result;
	}
```



---

### Функция ID: 397

**Исходный код:**
```java
private void reallocateResultWithCapacity(int newCapacity, boolean copyBucketsFromPreviousResult) {
        FixedCapacityExponentialHistogram newResult = FixedCapacityExponentialHistogram.create(newCapacity, breaker);
        if (copyBucketsFromPreviousResult && result != null) {
            BucketIterator it = result.negativeBuckets().iterator();
            while (it.hasNext()) {
                boolean added = newResult.tryAddBucket(it.peekIndex(), it.peekCount(), false);
                assert added : "Output histogram should have enough capacity";
                it.advance();
            }
            it = result.positiveBuckets().iterator();
            while (it.hasNext()) {
                boolean added = newResult.tryAddBucket(it.peekIndex(), it.peekCount(), true);
                assert added : "Output histogram should have enough capacity";
                it.advance();
            }
        }
        if (result != null && resultAlreadyReturned == false) {
            Releasables.close(result);
        }
        resultAlreadyReturned = false;
        result = newResult;
    }
```



---

### Функция ID: 398

**Исходный код:**
```java
public static @Nullable EnclosedInSquareBracketsConverter newInstance(@Nullable Configuration config,
			String[] options) {
		if (options.length < 1) {
			LOGGER.error("Incorrect number of options on style. Expected at least 1, received {}", options.length);
			return null;
		}
		PatternParser parser = PatternLayout.createPatternParser(config);
		List<PatternFormatter> formatters = parser.parse(options[0]);
		return new EnclosedInSquareBracketsConverter(formatters);
	}
```



---

### Функция ID: 399

**Исходный код:**
```python
def get_instance(self, instance_id: str, filters: list | None = None):
        """
        Get EC2 instance by id and return it.

        :param instance_id: id of the AWS EC2 instance
        :param filters: List of filters to specify instances to get
        :return: Instance object
        """
        if self._api_type == "client_type":
            return self.get_instances(filters=filters, instance_ids=[instance_id])[0]

        return self.conn.Instance(id=instance_id)
```



---

### Функция ID: 400

**Исходный код:**
```java
public static boolean equals(final Type type1, final Type type2) {
        if (Objects.equals(type1, type2)) {
            return true;
        }
        if (type1 instanceof ParameterizedType) {
            return equals((ParameterizedType) type1, type2);
        }
        if (type1 instanceof GenericArrayType) {
            return equals((GenericArrayType) type1, type2);
        }
        if (type1 instanceof WildcardType) {
            return equals((WildcardType) type1, type2);
        }
        return false;
    }
```



---


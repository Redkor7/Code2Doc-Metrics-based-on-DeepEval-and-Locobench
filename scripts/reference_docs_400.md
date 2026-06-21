# Эталонная документация (Expected Output)

---

### Функция ID: 1

Tries to determine the type arguments of a class/interface based on a super parameterized type's type arguments. This method is the inverse of
{@link #getTypeArguments(Type, Class)} which gets a class/interface's type arguments based on a subtype. It is far more limited in determining the type
arguments for the subject class's type variables in that it can only determine those parameters that map from the subject {@link Class} object to the
supertype.
<p>
Example: {@link java.util.TreeSet TreeSet} sets its parameter as the parameter for {@link java.util.NavigableSet NavigableSet}, which in turn sets the
parameter of {@link java.util.SortedSet}, which in turn sets the parameter of {@link Set}, which in turn sets the parameter of
{@link java.util.Collection}, which in turn sets the parameter of {@link Iterable}. Since {@link TreeSet}'s parameter maps (indirectly) to
{@link Iterable}'s parameter, it will be able to determine that based on the super type {@code Iterable<? extends
Map<Integer, ? extends Collection<?>>>}, the parameter of {@link TreeSet} is {@code ? extends Map<Integer, ? extends
Collection<?>>}.
</p>
@param cls                    the class whose type parameters are to be determined, not {@code null}.
@param superParameterizedType the super type from which {@code cls}'s type arguments are to be determined, not {@code null}.
@return a {@link Map} of the type assignments that could be determined for the type variables in each type in the inheritance hierarchy from {@code type}
        to {@code toClass} inclusive.
@throws NullPointerException if either {@code cls} or {@code superParameterizedType} is {@code null}.

---

### Функция ID: 2

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

---

### Функция ID: 3

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

---

### Функция ID: 4

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

---

### Функция ID: 5

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

---

### Функция ID: 6

Converts a given name of class into canonical format. If name of class is not a name of array class it returns
unchanged name.
<p>
The method does not change the {@code $} separators in case the class is inner class.
</p>
<p>
Example:
<ul>
<li>{@code getCanonicalName("[I") = "int[]"}</li>
<li>{@code getCanonicalName("[Ljava.lang.String;") = "java.lang.String[]"}</li>
<li>{@code getCanonicalName("java.lang.String") = "java.lang.String"}</li>
</ul>
</p>
@param name the name of class.
@return canonical form of class name.
@throws IllegalArgumentException if the class name is invalid.

---

### Функция ID: 7

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

---

### Функция ID: 8

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

---

### Функция ID: 9

Repeat an {@link Accumulable} function.
@param f to be repeated until...
@param again return false to exit
@returns
@example
```ts
// concats `[2]` 10 times on `[1]`
repeat(concat, times(10))([1], [2])
```

---

### Функция ID: 10

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

---

### Функция ID: 11

Record a function call result with custom encoding to both caches.

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

---

### Функция ID: 12

Memoize a function with record and replay functionality.

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

---

### Функция ID: 13

Record a function call result with custom encoding.

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

---

### Функция ID: 14

Replay a cached function result without executing the function.

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

---

### Функция ID: 15

Convert split points into ranges for autotuning dispatch.

Example:
    split_points=[512, 2048]
    returns:
           [(1, 512), (513, 2048), (2049, float('inf'))]

---

### Функция ID: 16

Create shared task (decorator).

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

---

### Функция ID: 17

Gets the set of currently active Route configuration objects from the router state.
This function synchronously reads the current router state without waiting for navigation events.
@param router - The Angular Router instance
@returns A Set containing all Route configuration objects that are currently active
@example
```ts
const activeRoutes = getActiveRouteConfigs(router);
// activeRoutes is a Set<Route> containing all currently active route configurations
```

---

### Функция ID: 18

Returns a mutable ref object.
@example
```ts
const ref = useRef(0);
ref.current = 1;
```
@template T The type of the ref object.
@param {T} initialValue The initial value of the ref object.
@returns {{ current: T }} The mutable ref object.

---

### Функция ID: 19

Gets the set of currently active Route configuration objects from the router state.
This function synchronously reads the current router state without waiting for navigation events.
@param router - The Angular Router instance
@returns A Set containing all Route configuration objects that are currently active
@example
```ts
const activeRoutes = getActiveRouteConfigs(router);
// activeRoutes is a Set<Route> containing all currently active route configurations
```

---

### Функция ID: 20

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

---

### Функция ID: 21

Generate a list of ops organized in a specific format.
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

---

### Функция ID: 22

Reverts the `_` variable to its previous value and returns a reference to
the `lodash` function.
@static
@since 0.1.0
@memberOf _
@category Util
@returns {Function} Returns the `lodash` function.
@example
var lodash = _.noConflict();

---

### Функция ID: 23

This method is the wrapper version of `_.reverse`.
**Note:** This method mutates the wrapped array.
@name reverse
@memberOf _
@since 0.1.0
@category Seq
@returns {Object} Returns the new `lodash` wrapper instance.
@example
var array = [1, 2, 3];
_(array).reverse().value()
// => [3, 2, 1]
console.log(array);
// => [3, 2, 1]

---

### Функция ID: 24

Casts `value` as an array if it's not one.
@static
@memberOf _
@since 4.4.0
@category Lang
@param {*} value The value to inspect.
@returns {Array} Returns the cast array.
@example
_.castArray(1);
// => [1]
_.castArray({ 'a': 1 });
// => [{ 'a': 1 }]
_.castArray('abc');
// => ['abc']
_.castArray(null);
// => [null]
_.castArray(undefined);
// => [undefined]
_.castArray();
// => []
var array = [1, 2, 3];
console.log(_.castArray(array) === array);
// => true

---

### Функция ID: 25

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

---

### Функция ID: 26

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

---

### Функция ID: 27

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

---

### Функция ID: 28

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

---

### Функция ID: 29

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

---

### Функция ID: 30

{@code A.is(B)} is defined as {@code Foo<A>.isSubtypeOf(Foo<B>)}.
<p>Specifically, returns true if any of the following conditions is met:
<ol>
  <li>'this' and {@code formalType} are equal.
  <li>'this' and {@code formalType} have equal canonical form.
  <li>{@code formalType} is {@code <? extends Foo>} and 'this' is a subtype of {@code Foo}.
  <li>{@code formalType} is {@code <? super Foo>} and 'this' is a supertype of {@code Foo}.
</ol>
Note that condition 2 isn't technically accurate under the context of a recursively bounded
type variables. For example, {@code Enum<? extends Enum<E>>} canonicalizes to {@code Enum<?>}
where {@code E} is the type variable declared on the {@code Enum} class declaration. It's
technically <em>not</em> true that {@code Foo<Enum<? extends Enum<E>>>} is a subtype of {@code
Foo<Enum<?>>} according to JLS. See testRecursiveWildcardSubtypeBug() for a real example.
<p>It appears that properly handling recursive type bounds in the presence of implicit type
bounds is not easy. For now we punt, hoping that this defect should rarely cause issues in real
code.
@param formalType is {@code Foo<formalType>} a supertype of {@code Foo<T>}?
@param declaration The type variable in the context of a parameterized type. Used to infer type
    bound when {@code formalType} is a wildcard with implicit upper bound.

---

### Функция ID: 31

Remove repeating module names from string.

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

---

### Функция ID: 32

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

---

### Функция ID: 33

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

---

### Функция ID: 34

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

---

### Функция ID: 35

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

---

### Функция ID: 36

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

---

### Функция ID: 37

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

---

### Функция ID: 38

Chaining operator.

Example:
    >>> add.s(2, 2) | add.s(4) | add.s(8)

Returns:
    chain: Constructs a :class:`~celery.canvas.chain` of the given signatures.

---

### Функция ID: 39

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

---

### Функция ID: 40

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

---

### Функция ID: 41

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

---

### Функция ID: 42

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

---

### Функция ID: 43

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

---

### Функция ID: 44

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

---

### Функция ID: 45

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

---

### Функция ID: 46

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

---

### Функция ID: 47

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

---

### Функция ID: 48

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

---

### Функция ID: 49

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

---

### Функция ID: 50

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

---

### Функция ID: 51

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

---

### Функция ID: 52

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

---

### Функция ID: 53

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

---

### Функция ID: 54

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

---

### Функция ID: 55

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

---

### Функция ID: 56

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

---

### Функция ID: 57

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

---

### Функция ID: 58

Creates a new array concatenating `array` with any additional arrays
and/or values.
@static
@memberOf _
@since 4.0.0
@category Array
@param {Array} array The array to concatenate.
@param {...*} [values] The values to concatenate.
@returns {Array} Returns the new concatenated array.
@example
var array = [1];
var other = _.concat(array, 2, [3], [[4]]);
console.log(other);
// => [1, 2, 3, [4]]
console.log(array);
// => [1]

---

### Функция ID: 59

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

---

### Функция ID: 60

@param assumeNewLineBeforeCloseBrace

`false` when called on text from a real source file.

`true` when we need to assume `position` is on a newline.


This is useful for codefixes. Consider

```

function f() {

|}

```

with `position` at `|`.


When inserting some text after an open brace, we would like to get indentation as if a newline was already there.

By default indentation at `position` will be 0 so 'assumeNewLineBeforeCloseBrace' overrides this behavior.

---

### Функция ID: 61

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

---

### Функция ID: 62

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

---

### Функция ID: 63

@param assumeNewLineBeforeCloseBrace

`false` when called on text from a real source file.

`true` when we need to assume `position` is on a newline.


This is useful for codefixes. Consider

```

function f() {

|}

```

with `position` at `|`.


When inserting some text after an open brace, we would like to get indentation as if a newline was already there.

By default indentation at `position` will be 0 so 'assumeNewLineBeforeCloseBrace' overrides this behavior.

---

### Функция ID: 64

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

---

### Функция ID: 65

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

---

### Функция ID: 66

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

---

### Функция ID: 67

Swaps a series of elements in the given boolean array.
<p>This method does nothing for a {@code null} or empty input array or
for overflow indices. Negative indices are promoted to 0(zero). If any
of the sub-arrays to swap falls outside of the given array, then the
swap is stopped at the end of the array and as many as possible elements
are swapped.</p>
Examples:
<ul>
    <li>ArrayUtils.swap([true, false, true, false], 0, 2, 1) -&gt; [true, false, true, false]</li>
    <li>ArrayUtils.swap([true, false, true, false], 0, 0, 1) -&gt; [true, false, true, false]</li>
    <li>ArrayUtils.swap([true, false, true, false], 0, 2, 2) -&gt; [true, false, true, false]</li>
    <li>ArrayUtils.swap([true, false, true, false], -3, 2, 2) -&gt; [true, false, true, false]</li>
    <li>ArrayUtils.swap([true, false, true, false], 0, 3, 3) -&gt; [false, false, true, true]</li>
</ul>
@param array the array to swap, may be {@code null}.
@param offset1 the index of the first element in the series to swap.
@param offset2 the index of the second element in the series to swap.
@param len the number of elements to swap starting with the given indices.
@since 3.5

---

### Функция ID: 68

@param assumeNewLineBeforeCloseBrace

`false` when called on text from a real source file.

`true` when we need to assume `position` is on a newline.


This is useful for codefixes. Consider

```

function f() {

|}

```

with `position` at `|`.


When inserting some text after an open brace, we would like to get indentation as if a newline was already there.

By default indentation at `position` will be 0 so 'assumeNewLineBeforeCloseBrace' overrides this behavior.

---

### Функция ID: 69

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

---

### Функция ID: 70

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

---

### Функция ID: 71

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

---

### Функция ID: 72

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

---

### Функция ID: 73

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

---

### Функция ID: 74

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

---

### Функция ID: 75

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

---

### Функция ID: 76

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

---

### Функция ID: 77

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

---

### Функция ID: 78

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

---

### Функция ID: 79

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

---

### Функция ID: 80

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

---

### Функция ID: 81

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

---

### Функция ID: 82

Returns a mutable ref object.
@example
```ts
const ref = useRef(0);
ref.current = 1;
```
@template T The type of the ref object.
@param {T} initialValue The initial value of the ref object.
@returns {{ current: T }} The mutable ref object.

---

### Функция ID: 83

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

---

### Функция ID: 84

@param assumeNewLineBeforeCloseBrace

`false` when called on text from a real source file.

`true` when we need to assume `position` is on a newline.


This is useful for codefixes. Consider

```

function f() {

|}

```

with `position` at `|`.


When inserting some text after an open brace, we would like to get indentation as if a newline was already there.

By default indentation at `position` will be 0 so 'assumeNewLineBeforeCloseBrace' overrides this behavior.

---

### Функция ID: 85

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

---

### Функция ID: 86

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

---

### Функция ID: 87

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

---

### Функция ID: 88

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

---

### Функция ID: 89

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

---

### Функция ID: 90

@param assumeNewLineBeforeCloseBrace

`false` when called on text from a real source file.

`true` when we need to assume `position` is on a newline.


This is useful for codefixes. Consider

```

function f() {

|}

```

with `position` at `|`.


When inserting some text after an open brace, we would like to get indentation as if a newline was already there.

By default indentation at `position` will be 0 so 'assumeNewLineBeforeCloseBrace' overrides this behavior.

---

### Функция ID: 91

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

---

### Функция ID: 92

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

---

### Функция ID: 93

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

---

### Функция ID: 94

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

---

### Функция ID: 95

Sets the given bucket of the negative buckets. If the bucket already exists, it will be replaced.
Buckets may be set in arbitrary order. However, for best performance and minimal allocations,
buckets should be set in order of increasing index and all negative buckets should be set before positive buckets.
@param index the index of the bucket
@param count the count of the bucket, must be at least 1
@return the builder

---

### Функция ID: 96

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

---

### Функция ID: 97

Gets a {@link List} of stack frames, the message
is not included. Only the trace of the specified exception is
returned, any caused by trace is stripped.
<p>This works in most cases and will only fail if the exception
message contains a line that starts with: {@code "<whitespace>at"}.</p>
@param throwable is any throwable.
@return List of stack frames.

---

### Функция ID: 98

Return the profile or {@code null} if the resource is not profile specific.
@return the profile or {@code null}
@since 2.4.6

---

### Функция ID: 99

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

---

### Функция ID: 100

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

---

### Функция ID: 101

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

---

### Функция ID: 102

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

---

### Функция ID: 103

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

---

### Функция ID: 104

Check subscribed queue for messages and write them to xcom with the ``messages`` key.

:param context: the context object
:return: ``True`` if message is available or ``False``

---

### Функция ID: 105

Retrieves the meta-data of the service at the specified URL.
@param url the URL
@return the response

---

### Функция ID: 106

Retrieves the short path form of the specified path.
@param path the path
@return the short path name, or the original path name if unsupported or unavailable

---

### Функция ID: 107

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

---

### Функция ID: 108

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

---

### Функция ID: 109

Extract default values from JSON schema for any object type.

:param object_type: The object type to get defaults for (e.g., "operator", "dag")
:return: Dictionary of field name -> default value

---

### Функция ID: 110

Sets the given bucket of the negative buckets. If the bucket already exists, it will be replaced.
Buckets may be set in arbitrary order. However, for best performance and minimal allocations,
buckets should be set in order of increasing index and all negative buckets should be set before positive buckets.
@param index the index of the bucket
@param count the count of the bucket, must be at least 1
@return the builder

---

### Функция ID: 111

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

---

### Функция ID: 112

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

---

### Функция ID: 113

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

---

### Функция ID: 114

Determine the type of filesystem used - we might want to use different parameters if tmpfs is used.
:param filepath: path to check
:return: type of filesystem

---

### Функция ID: 115

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

---

### Функция ID: 116

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

---

### Функция ID: 117

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

---

### Функция ID: 118

Check if the transaction is in the prepared state.
@return true if the current state is PREPARED_TRANSACTION

---

### Функция ID: 119

Run the appropriate and handle and errors.
@param args the input arguments
@return a return status code (non boot is used to indicate an error)

---

### Функция ID: 120

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

---

### Функция ID: 121

Sets the given bucket of the negative buckets. If the bucket already exists, it will be replaced.
Buckets may be set in arbitrary order. However, for best performance and minimal allocations,
buckets should be set in order of increasing index and all negative buckets should be set before positive buckets.
@param index the index of the bucket
@param count the count of the bucket, must be at least 1
@return the builder

---

### Функция ID: 122

Check if the current platform is WSL2. This method will exit with error printing appropriate
message if WSL1 is detected as WSL1 is not supported.

:return: True if the current platform is WSL2, False otherwise (unless it's WSL1 then it exits).

---

### Функция ID: 123

Render the DAG object to the DOT object.

If an task instance list is passed, the nodes will be painted according to task statuses.

:param dag: DAG that will be rendered.
:param tis: List of task instances
:return: Graphviz object

---

### Функция ID: 124

Provide a human-readable explanation of why this future has not yet completed.
@return null if an explanation cannot be provided (e.g. because the future is done).
@since 23.0

---

### Функция ID: 125

Replaces matches for `pattern` in `string` with `replacement`.
**Note:** This method is based on
[`String#replace`](https://mdn.io/String/replace).
@static
@memberOf _
@since 4.0.0
@category String
@param {string} [string=''] The string to modify.
@param {RegExp|string} pattern The pattern to replace.
@param {Function|string} replacement The match replacement.
@returns {string} Returns the modified string.
@example
_.replace('Hi Fred', 'Fred', 'Barney');
// => 'Hi Barney'

---

### Функция ID: 126

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

---

### Функция ID: 127

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

---

### Функция ID: 128

Unset the preferred read replica. This causes the fetcher to go back to the leader for fetches.
@param tp The topic partition
@return the removed preferred read replica if set, Empty otherwise.

---

### Функция ID: 129

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

---

### Функция ID: 130

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

---

### Функция ID: 131

Gets an accessible method (that is, one that can be invoked via reflection) that implements the specified Method. If no such method can be found, return
{@code null}.
@param cls The implementing class, may be null.
@param method The method that we wish to call, may be null.
@return The accessible method or null.
@since 3.19.0

---

### Функция ID: 132

@param value Value in [-1,1].
@return Value arcsine, in radians, in [-PI/2,PI/2].

---

### Функция ID: 133

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

---

### Функция ID: 134

Writes a sequence of bytes to this channel from the subsequence of the given buffers.
@param srcs The buffers from which bytes are to be retrieved
@param offset The offset within the buffer array of the first buffer from which bytes are to be retrieved; must be non-negative and no larger than srcs.length.
@param length - The maximum number of buffers to be accessed; must be non-negative and no larger than srcs.length - offset.
@return returns no.of bytes written , possibly zero.
@throws IOException If some other I/O error occurs

---

### Функция ID: 135

Similar to load64, but allows offset + 8 > input.length, padding the result with zeroes. This
has to explicitly reverse the order of the bytes as it packs them into the result which makes
it slower than the native version.
@param input the input bytes
@param offset the offset into the array at which to start reading
@param length the number of bytes from the input to read
@return a long of a concatenated 8 bytes

---

### Функция ID: 136

Read all DAGs in serialized_dag table.

:param session: ORM Session
:returns: a dict of DAGs read from database

---

### Функция ID: 137

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

---

### Функция ID: 138

Cleans up the database_type String from an ipinfo database by splitting on punctuation, removing stop words, and then joining
with an underscore.
<p>
e.g. "ipinfo free_foo_sample.mmdb" -> "foo"
@param type the database_type from an ipinfo database
@return a cleaned up database_type string

---

### Функция ID: 139

Begin connecting to the given node, return true if we are already connected and ready to send to that node.
@param node The node to check
@param now The current timestamp
@return True if we are ready to send to the given node

---

### Функция ID: 140

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

---

### Функция ID: 141

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

---

### Функция ID: 142

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

---

### Функция ID: 143

Use a specific filter to determine when a callback should apply. If no explicit
filter is set filter will be attempted using the generic type on the callback
type.
@param filter the filter to use
@return this instance
@since 3.4.8

---

### Функция ID: 144

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

---

### Функция ID: 145

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

---

### Функция ID: 146

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

---

### Функция ID: 147

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

---

### Функция ID: 148

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

---

### Функция ID: 149

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

---

### Функция ID: 150

Returns list of PRs that are commented out in the changelog.
:return: list of PR numbers that appear only in comments in changelog.rst files in "providers" dir

---

### Функция ID: 151

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

---

### Функция ID: 152

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

---

### Функция ID: 153

Joins the elements of the provided {@link Iterator} into a single String containing the provided elements.
<p>
No delimiter is added before or after the list. Null objects or empty strings within the iteration are represented by empty strings.
</p>
<p>
See the examples here: {@link #join(Object[],char)}.
</p>
@param iterator  the {@link Iterator} of values to join together, may be null.
@param separator the separator character to use.
@return the joined String, {@code null} if null iterator input.
@since 2.0

---

### Функция ID: 154

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

---

### Функция ID: 155

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

---

### Функция ID: 156

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

---

### Функция ID: 157

Flush logs out of the heap, deduplicating them based on the last log.

:param heap: heap to flush logs from
:param flush_size: number of logs to flush
:param last_log_container: a container to store the last log, to avoid duplicate logs
:return: a generator that yields deduplicated logs

---

### Функция ID: 158

Parse and validate configs against this configuration definition. The input is a map of configs. It is expected
that the keys of the map are strings, but the values can either be strings or they may already be of the
appropriate type (int, string, etc). This will work equally well with either java.util.Properties instances or a
programmatically constructed map.
@param props The configs to parse and validate.
@return Parsed and validated configs. The key will be the config name and the value will be the value parsed into
the appropriate type (int, string, etc).

---

### Функция ID: 159

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

---

### Функция ID: 160

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

---

### Функция ID: 161

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

---

### Функция ID: 162

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

---

### Функция ID: 163

Compares two objects for equality.
@param obj the object to compare to.
@return {@code true} if equal.

---

### Функция ID: 164

Copies all characters between the {@link Readable} and {@link Appendable} objects. Does not
close or flush either object.
@param from the object to read from
@param to the object to write to
@return the number of characters copied
@throws IOException if an I/O error occurs

---

### Функция ID: 165

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

---

### Функция ID: 166

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

---

### Функция ID: 167

Appends each item in an array to the builder without any separators.
Appending a null array will have no effect.
Each object is appended using {@link #append(Object)}.
@param <T>  the element type
@param array  the array to append
@return {@code this} instance.
@since 2.3

---

### Функция ID: 168

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

---

### Функция ID: 169

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

---

### Функция ID: 170

Gets a fraction that is the negative (-fraction) of this one.
<p>
The returned fraction is not reduced.
</p>
@return a new fraction instance with the opposite signed numerator

---

### Функция ID: 171

Prepare code snippet with line numbers and  a specific line marked.

:param file_path: File name
:param line_no: Line number
:param context_lines_count: The number of lines that will be cut before and after.
:return: str

---

### Функция ID: 172

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

---

### Функция ID: 173

Turn a str log stream into a generator of parsed log lines.

:param log_stream: The stream to parse.
:return: A generator of parsed log lines.

---

### Функция ID: 174

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

---

### Функция ID: 175

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

---

### Функция ID: 176

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

---

### Функция ID: 177

Removes multiple array elements specified by indices.
@param array the input array, will not be modified, and may be {@code null}.
@param indices to remove.
@return new array of same type minus elements specified by the set bits in {@code indices}.

---

### Функция ID: 178

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

---

### Функция ID: 179

@param assumeNewLineBeforeCloseBrace

`false` when called on text from a real source file.

`true` when we need to assume `position` is on a newline.


This is useful for codefixes. Consider

```

function f() {

|}

```

with `position` at `|`.


When inserting some text after an open brace, we would like to get indentation as if a newline was already there.

By default indentation at `position` will be 0 so 'assumeNewLineBeforeCloseBrace' overrides this behavior.

---

### Функция ID: 180

Repeatedly call a provided boto3 API Callable and collates the responses into a List.

:param api_call: The api command to execute.
:param response_key: Which dict key to collect into the final list.
:param verbose: Provides additional logging if set to True.  Defaults to False.
:return: A List of the combined results of the provided API call.

---

### Функция ID: 181

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

---

### Функция ID: 182

Run the appropriate and handle and errors.
@param args the input arguments
@return a return status code (non boot is used to indicate an error)

---

### Функция ID: 183

Parse and validate configs against this configuration definition. The input is a map of configs. It is expected
that the keys of the map are strings, but the values can either be strings or they may already be of the
appropriate type (int, string, etc). This will work equally well with either java.util.Properties instances or a
programmatically constructed map.
@param props The configs to parse and validate.
@return Parsed and validated configs. The key will be the config name and the value will be the value parsed into
the appropriate type (int, string, etc).

---

### Функция ID: 184

Writes a sequence of bytes to this channel from the given buffer.
@param src The buffer from which bytes are to be retrieved
@return The number of bytes read from src, possibly zero, or -1 if the channel has reached end-of-stream
@throws IOException If some other I/O error occurs

---

### Функция ID: 185

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

---

### Функция ID: 186

(#49811)

Note that there are cases in which the symbol declaration is not present. For example, in the code below both

`MappedIndirect.ax` and `MappedIndirect.ay` have no declaration node attached (due to their mapped-type

parent):


```ts

type Base = { ax: number; ay: string };

type BaseKeys = keyof Base;

type MappedIndirect = { [K in BaseKeys]: boolean };

```


In such cases, we assume the declaration to be a `PropertySignature`.

---

### Функция ID: 187

Reads a sequence of key/value pairs and the trailing closing brace '}' of an
object. The opening brace '{' should have already been read.
@return an object
@throws JSONException if processing of json failed

---

### Функция ID: 188

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

---

### Функция ID: 189

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

---

### Функция ID: 190

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

---

### Функция ID: 191

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

---

### Функция ID: 192

Compares two objects for equality.
@param obj  the object to compare to.
@return {@code true} if equal.

---

### Функция ID: 193

Fetch the committed offsets for a set of partitions. This is a non-blocking call. The
returned future can be polled to get the actual offsets returned from the broker.
@param partitions The set of partitions to get offsets for.
@return A request future containing the committed offsets.

---

### Функция ID: 194

@param assumeNewLineBeforeCloseBrace

`false` when called on text from a real source file.

`true` when we need to assume `position` is on a newline.


This is useful for codefixes. Consider

```

function f() {

|}

```

with `position` at `|`.


When inserting some text after an open brace, we would like to get indentation as if a newline was already there.

By default indentation at `position` will be 0 so 'assumeNewLineBeforeCloseBrace' overrides this behavior.

---

### Функция ID: 195

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

---

### Функция ID: 196

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

---

### Функция ID: 197

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

---

### Функция ID: 198

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

---

### Функция ID: 199

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

---

### Функция ID: 200

Commit offsets for the specified list of topics and partitions. This is a non-blocking call
which returns a request future that can be polled in the case of a synchronous commit or ignored in the
asynchronous case.
NOTE: This is visible only for testing
@param offsets The list of offsets per partition that should be committed.
@return A request future whose value indicates whether the commit was successful or not

---

### Функция ID: 201

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

---

### Функция ID: 202

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

---

### Функция ID: 203

Closes a {@link Closeable}, with control over whether an {@code IOException} may be thrown.
This is primarily useful in a finally block, where a thrown exception needs to be logged but
not propagated (otherwise the original exception will be lost).
<p>If {@code swallowIOException} is true then we never throw {@code IOException} but merely log
it.
<p>Example:
{@snippet :
public void useStreamNicely() throws IOException {
  SomeStream stream = new SomeStream("foo");
  boolean threw = true;
  try {
    // ... code which does something with the stream ...
    threw = false;
  } finally {
    // If an exception occurs, rethrow it only if threw==false:
    Closeables.close(stream, threw);
  }
}
}
@param closeable the {@code Closeable} object to be closed, or null, in which case this method
    does nothing
@param swallowIOException if true, don't propagate IO exceptions thrown by the {@code close}
    methods
@throws IOException if {@code swallowIOException} is false and {@code close} throws an {@code
    IOException}.

---

### Функция ID: 204

Check whether the group should be rejoined (e.g. if metadata changes) or whether a
rejoin request is already in flight and needs to be completed.
@return true if it should, false otherwise

---

### Функция ID: 205

@param assumeNewLineBeforeCloseBrace

`false` when called on text from a real source file.

`true` when we need to assume `position` is on a newline.


This is useful for codefixes. Consider

```

function f() {

|}

```

with `position` at `|`.


When inserting some text after an open brace, we would like to get indentation as if a newline was already there.

By default indentation at `position` will be 0 so 'assumeNewLineBeforeCloseBrace' overrides this behavior.

---

### Функция ID: 206

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

---

### Функция ID: 207

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

---

### Функция ID: 208

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

---

### Функция ID: 209

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

---

### Функция ID: 210

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

---

### Функция ID: 211

Return the path of the scikit-learn data directory.

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

---

### Функция ID: 212

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

---

### Функция ID: 213

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

---

### Функция ID: 214

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

---

### Функция ID: 215

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

---

### Функция ID: 216

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

---

### Функция ID: 217

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

---

### Функция ID: 218

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

---

### Функция ID: 219

Get the parameters used to repr(Series) calls using Series.to_string.

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

---

### Функция ID: 220

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

---

### Функция ID: 221

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

---

### Функция ID: 222

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

---

### Функция ID: 223

Send the calls which are ready.
@param now The current time in milliseconds.
@return The minimum timeout we need for poll().

---

### Функция ID: 224

Approximates asin to within about 1e-6. This approximation works by breaking the range from 0 to 1 into 5 regions
for all but the region nearest 1, rational polynomial models get us a very good approximation of asin and by
interpolating as we move from region to region, we can guarantee continuity and we happen to get monotonicity as
well.  for the values near 1, we just use Math.asin as our region "approximation".
@param x sin(theta)
@return theta

---

### Функция ID: 225

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

---

### Функция ID: 226

@param assumeNewLineBeforeCloseBrace

`false` when called on text from a real source file.

`true` when we need to assume `position` is on a newline.


This is useful for codefixes. Consider

```

function f() {

|}

```

with `position` at `|`.


When inserting some text after an open brace, we would like to get indentation as if a newline was already there.

By default indentation at `position` will be 0 so 'assumeNewLineBeforeCloseBrace' overrides this behavior.

---

### Функция ID: 227

@param q The quantile desired.  Can be in the range [0,1].
@return The minimum value x such that we think that the proportion of samples is &le; x is q.

---

### Функция ID: 228

Determine a cache key for the given method and target class.
<p>Must not produce same key for overloaded methods.
Must produce same key for different instances of the same method.
@param method the method (never {@code null})
@param targetClass the target class (may be {@code null})
@return the cache key (never {@code null})

---

### Функция ID: 229

Returns {@code true} if this element is an immediate parent of the specified name.
@param name the name to check
@return {@code true} if this name is an ancestor

---

### Функция ID: 230

Finalize the state of a batch. Final state, once set, is immutable. This function may be called
once or twice on a batch. It may be called twice if
1. An inflight batch expires before a response from the broker is received. The batch's final
state is set to FAILED. But it could succeed on the broker and second time around batch.done() may
try to set SUCCEEDED final state.
2. If a transaction abortion happens or if the producer is closed forcefully, the final state is
ABORTED but again it could succeed if broker responds with a success.
Attempted transitions from [FAILED | ABORTED] --> SUCCEEDED are logged.
Attempted transitions from one failure state to the same or a different failed state are ignored.
Attempted transitions from SUCCEEDED to the same or a failed state throw an exception.
@param baseOffset The base offset of the messages assigned by the server
@param logAppendTime The log append time or -1 if CreateTime is being used
@param topLevelException The exception that occurred (or null if the request was successful)
@param recordExceptions Record exception function mapping batchIndex to the respective record exception
@return true if the batch was completed successfully and false if the batch was previously aborted

---

### Функция ID: 231

Invokes a short supplier, and returns the result.
@param supplier The short supplier to invoke.
@param <E> The type of checked exception, which the supplier can throw.
@return The short, which has been created by the supplier

---

### Функция ID: 232

Gets a custom format from a format description.
@param desc String
@return Format

---

### Функция ID: 233

Retrieves the data contained in the regular files named by {@code keys} in the directory given by {@code path}.
Non-regular files (such as directories) in the given directory are silently ignored.
@param path the directory where data files reside.
@param keys the keys whose values will be retrieved.
@return the configuration data.

---

### Функция ID: 234

Returns whether or not the given node has a JSDoc "inheritDoc" tag on it.

@param node the Node in question.

@returns `true` if `node` has a JSDoc "inheritDoc" tag on it, otherwise `false`.

---

### Функция ID: 235

Parse a value according to its expected type.
@param name  The config name
@param value The config value
@param type  The expected type
@return The parsed object

---

### Функция ID: 236

Deletes the string wherever it occurs in the builder.
@param str  the string to delete, null causes no action
@return {@code this} instance.

---

### Функция ID: 237

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

---

### Функция ID: 238

Return the location of the property within the source (if known).
@return the location or {@code null}

---

### Функция ID: 239

Creates an {@link Integer} from a {@link String}.
Handles hexadecimal (0xhhhh) and octal (0dddd) notations. A leading zero means octal; spaces are not trimmed.
<p>
Returns {@code null} if the string is {@code null}.
</p>
@param str a {@link String} to convert, may be null.
@return converted {@link Integer} (or null if the input is null).
@throws NumberFormatException if the value cannot be converted.

---

### Функция ID: 240

Parse certificates from the specified string.
@param text the text to parse
@return the parsed certificates

---

### Функция ID: 241

Approximates asin to within about 1e-6. This approximation works by breaking the range from 0 to 1 into 5 regions
for all but the region nearest 1, rational polynomial models get us a very good approximation of asin and by
interpolating as we move from region to region, we can guarantee continuity and we happen to get monotonicity as
well.  for the values near 1, we just use Math.asin as our region "approximation".
@param x sin(theta)
@return theta

---

### Функция ID: 242

Copies all bytes from the readable channel to the writable channel. Does not close or flush
either channel.
@param from the readable channel to read from
@param to the writable channel to write to
@return the number of bytes copied
@throws IOException if an I/O error occurs

---

### Функция ID: 243

Render the DAG dependency to the DOT object.

:param deps: List of DAG dependencies
:return: Graphviz object

---

### Функция ID: 244

A faster and less accurate {@link Math#sinh}
@param value A double value.
@return Value hyperbolic sine.

---

### Функция ID: 245

Compares this pair to another based on the two elements.
@param obj  the object to compare to, null returns false.
@return true if the elements of the pair are equal.

---

### Функция ID: 246

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

---

### Функция ID: 247

Insert a key-value pair into the on-disk cache.

Args:
    key: The key to insert (must be str).
    value: The value to associate with the key (must be bytes).

Returns:
    True if successfully inserted, False if the key already exists
    with a valid version.

---

### Функция ID: 248

Learn whether the specified Collection contains non-null elements.
@param coll to check
@return {@code true} if some Object was found, {@code false} otherwise.

---

### Функция ID: 249

Check if the exception is a log configuration message, i.e. the log call might not
have actually output anything.
@param ex the source exception
@return {@code true} if the exception contains a log configuration message

---

### Функция ID: 250

Get import status from Dynamodb.

:param import_arn: The Amazon Resource Name (ARN) for the import.
:return: Import status, Error code and Error message

---

### Функция ID: 251

If there is no pending task, set the pending task active.
 If wakeup was called before setting an active task, the current task will complete exceptionally with
 WakeupException right away.
 If there is an active task, throw exception.
@param currentTask
@param <T>
@return

---

### Функция ID: 252

Extract the target element type from the specified container type or {@code null}
if no element type was found.
@param type a type, potentially wrapping an element type
@return the element type or {@code null} if no specific type was found

---

### Функция ID: 253

Tests, recursively, whether any of the type parameters associated with {@code type} are bound to variables.
@param type The type to check for type variables.
@return Whether any of the type parameters associated with {@code type} are bound to variables.
@since 3.2

---

### Функция ID: 254

Learn whether the specified Collection contains non-null elements.
@param coll to check
@return {@code true} if some Object was found, {@code false} otherwise.

---

### Функция ID: 255

Actually find a method with matching parameter type, i.e. where each
argument value is assignable to the corresponding parameter type.
@param arguments the argument values to match against method parameters
@return a matching method, or {@code null} if none

---

### Функция ID: 256

Return the related causes, if any.
@return the array of related causes, or {@code null} if none

---

### Функция ID: 257

Returns if the given option is contained in this set.
@param option the option to check
@return {@code true} of the option is present

---

### Функция ID: 258

Returns the value at {@code index}.
@param index the index to get the value from
@return the value at {@code index}.
@throws JSONException if this array has no value at {@code index}, or if that value
is the {@code null} reference. This method returns normally if the value is
{@code JSONObject#NULL}.

---

### Функция ID: 259

Use a specific filter to determine when a callback should apply. If no explicit
filter is set filter will be attempted using the generic type on the callback
type.
@param filter the filter to use
@return this instance
@since 3.4.8

---

### Функция ID: 260

Appends an array placing separators between each value, but
not before the first or after the last.
Appending a null array will have no effect.
Each object is appended using {@link #append(Object)}.
@param array  the array to append
@param separator  the separator to use, null means no separator
@return {@code this} instance.

---

### Функция ID: 261

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

---

### Функция ID: 262

Return a report for all the properties that are no longer supported. If no such
properties were found, return {@code null}.
@return a report with the configurations keys that are no longer supported

---

### Функция ID: 263

Block until all pending requests from the given node have finished.
@param node The node to await requests from
@param timer Timer bounding how long this method can block
@return true If all requests finished, false if the timeout expired first

---

### Функция ID: 264

Internal helper method for acquiring a permit. This method checks whether currently a permit can be acquired and - if so - increases the internal
counter. The return value indicates whether a permit could be acquired. This method must be called with the lock of this object held.
@return a flag whether a permit could be acquired.

---

### Функция ID: 265

Gets the minimum of three {@code short} values.
@param a value 1.
@param b value 2.
@param c value 3.
@return the smallest of the values.

---

### Функция ID: 266

Creates a matcher from a set of characters.
@param chars  the characters to match, null or empty matches nothing.
@return a new matcher for the given char[].

---

### Функция ID: 267

Wrap the function to enable memoization.

Args:
    fn: The function to wrap.

Returns:
    A wrapped version of the function.

---

### Функция ID: 268

Add all property values from the given Map.
@param other a Map with property values keyed by property name,
which must be a String
@return this in order to allow for adding multiple property values in a chain

---

### Функция ID: 269

Converts an array of primitive ints to objects.
<p>This method returns {@code null} for a {@code null} input array.</p>
@param array  an {@code int} array.
@return an {@link Integer} array, {@code null} if null array input.

---

### Функция ID: 270

Return {@code true} if any element in the name is indexed.
@return if the element has one or more indexed elements
@since 2.2.10

---

### Функция ID: 271

Remove a proxied interface.
<p>Does nothing if the given interface isn't proxied.
@param ifc the interface to remove from the proxy
@return {@code true} if the interface was removed; {@code false}
if the interface was not found and hence could not be removed

---

### Функция ID: 272

Whether the property values should be injected.
@param pvs property values to check
@return whether the property values should be injected
@since 6.0.10

---

### Функция ID: 273

External access to context constructor.
@return A pointer to the context if successful; NULL if an error occurred.

---

### Функция ID: 274

@return Map of topics partitions received in a target assignment that have not been
reconciled yet because topic names are not in metadata or reconciliation hasn't finished.
The values in the map are the sets of partitions contained in the target assignment but
missing from the currently reconciled assignment, for each topic.
Visible for testing.

---

### Функция ID: 275

Convert color in #RGB (12 bits) format to #RRGGBB (32 bits), if it possible.

Otherwise, it returns the original value. Graphviz does not support colors in #RGB format.

:param color: Text representation of color
:return: Refined representation of color

---

### Функция ID: 276

Compares another object for equality with this object.
@param obj the object to compare to
@return {@code true}if equal to this instance

---

### Функция ID: 277

Dynamically creates a model proxy interface for a give name. For each prop
accessed on this proxy, it will lookup the dmmf to find if that model exists.
If it is the case, it will create a proxy for that model via {@link applyModel}.
@param client to create the proxy around
@returns a proxy to access models

---

### Функция ID: 278

Get state of the Glue job; the job state can be running, finished, failed, stopped or timeout.

.. seealso::
    - :external+boto3:py:meth:`Glue.Client.get_job_run`

:param job_name: unique job name per AWS account
:param run_id: The job-run ID of the predecessor job run
:return: State of the Glue job

---

### Функция ID: 279

@param sValue       Value to parse, which may be {@code null}.
@param defaultValue Value to return if {@code sValue} is {@code null}.
@param settingName  Name of the parameter or setting. On invalid input, this value is included in the exception message. Otherwise,
                    this parameter is unused.
@return The {@link TimeValue} which the input string represents, or {@code defaultValue} if the input is {@code null}.

---

### Функция ID: 280

Gets the number of steps needed to turn the source class into the destination class. This represents the number of steps in the object hierarchy graph.
@param srcClass  The source class.
@param destClass The destination class.
@return The cost of transforming an object.

---

### Функция ID: 281

Returns the value at {@code index} if it exists and is an int or can be coerced to
an int.
@param index the index to get the value from
@return the {@code value}
@throws JSONException if the value at {@code index} doesn't exist or cannot be
coerced to an int.

---

### Функция ID: 282

Strips leading zeros from version number.

This converts 1974.04.03 to 1974.4.3 as the format with leading month and day zeros is not accepted
by PIP versioning.

:param version: version number in CALVER format (potentially with leading 0s in date and month)
:return: string with leading 0s after dot replaced.

---

### Функция ID: 283

@return a future for the results of all described users with map keys (one per user) being consistent with the
contents of the list returned by {@link #users()}. The future will complete successfully only if all such user
descriptions complete successfully.

---

### Функция ID: 284

Check if the file hash is present in cache and its content has been modified. Optionally updates
the hash.

:param file_hash: hash of the current version of the file
:param cache_path: path where the hash is stored
:param update: whether to update hash if it is found different
:return: True if the hash file was missing or hash has changed.

---

### Функция ID: 285

Wrap the given bean if necessary, i.e. if it is eligible for being proxied.
@param bean the raw bean instance
@param beanName the name of the bean
@param cacheKey the cache key for metadata access
@return a proxy wrapping the bean, or the raw bean instance as-is

---

### Функция ID: 286

Check for calls which have timed out.
Timed out calls will be removed and failed.
The remaining milliseconds until the next timeout will be updated.
@param calls The collection of calls.
@return The number of calls which were timed out.

---

### Функция ID: 287

Returns the next value from the input.
@return a {@link JSONObject}, {@link JSONArray}, String, Boolean, Integer, Long,
Double or {@link JSONObject#NULL}.
@throws JSONException if the input is malformed.

---

### Функция ID: 288

Creates a new instance of the class. Required by Log4J2.
@param config the configuration
@param options the options
@return a new instance, or {@code null} if the options are invalid

---

### Функция ID: 289

Gets the minimum of three {@code long} values.
@param a value 1.
@param b value 2.
@param c value 3.
@return the smallest of the values.

---

### Функция ID: 290

Returns the index within {@code cs} of the first occurrence of the specified character, starting the search at the specified index.
<p>
If a character with value {@code searchChar} occurs in the character sequence represented by the {@code cs} object at an index no smaller than
{@code start}, then the index of the first such occurrence is returned. For values of {@code searchChar} in the range from 0 to 0xFFFF (inclusive), this
is the smallest value <em>k</em> such that:
</p>
<pre>
(this.charAt(<em>k</em>) == searchChar) &amp;&amp; (<em>k</em> &gt;= start)
</pre>
<p>
is true. For other values of {@code searchChar}, it is the smallest value <em>k</em> such that:
</p>
<pre>
(this.codePointAt(<em>k</em>) == searchChar) &amp;&amp; (<em>k</em> &gt;= start)
</pre>
<p>
is true. In either case, if no such character occurs inm {@code cs} at or after position {@code start}, then {@code -1} is returned.
</p>
<p>
There is no restriction on the value of {@code start}. If it is negative, it has the same effect as if it were zero: the entire {@link CharSequence} may
be searched. If it is greater than the length of {@code cs}, it has the same effect as if it were equal to the length of {@code cs}: {@code -1} is
returned.
</p>
<p>
All indices are specified in {@code char} values (Unicode code units).
</p>
@param cs         the {@link CharSequence} to be processed, not null.
@param searchChar the char to be searched for.
@param start      the start index, negative starts at the string start.
@return the index where the search char was found, -1 if not found.
@since 3.6 updated to behave more like {@link String}.

---

### Функция ID: 291

Sorts the given array into ascending order and returns it.
@param array the array to sort (may be null).
@return the given array.
@see Arrays#sort(float[])

---

### Функция ID: 292

Downloads the patch file for a given PR from the specified GitHub repository.

Args:
    pr_number (int): The pull request number.
    repo_url (str): The URL of the repository where the PR is hosted.
    download_dir (str): The directory to store the downloaded patch.

Returns:
    str: The path to the downloaded patch file.

Exits:
    If the download fails, the script will exit.

---

### Функция ID: 293

Returns the {@link Log} for the application. By default will be deduced.
@return the application log

---

### Функция ID: 294

Check whether a call should be timed out.
The remaining milliseconds until the next timeout will be updated.
@param call The call.
@return True if the call should be timed out.

---

### Функция ID: 295

Returns a Writer that sends all output to the given {@link Appendable} target. Closing the
writer will close the target if it is {@link Closeable}, and flushing the writer will flush the
target if it is {@link java.io.Flushable}.
@param target the object to which output will be sent
@return a new Writer object, unless target is a Writer, in which case the target is returned

---

### Функция ID: 296

Replaces all the occurrences of variables with their matching values
from the resolver using the given source array as a template.
The array is not altered by this method.
@param source  the character array to replace in, not altered, null returns null.
@return the result of the replace operation.

---

### Функция ID: 297

Check that a prefix exists in a bucket.

:param bucket_name: the name of the bucket
:param prefix: a key prefix
:param delimiter: the delimiter marks key hierarchy.
:return: False if the prefix does not exist in the bucket and True if it does.

---

### Функция ID: 298

Return the override for the given method, if any.
@param method the method to check for overrides for
@return the method override, or {@code null} if none

---

### Функция ID: 299

Reads a line of text. A line is considered to be terminated by any one of a line feed ({@code
'\n'}), a carriage return ({@code '\r'}), or a carriage return followed immediately by a
linefeed ({@code "\r\n"}).
@return a {@code String} containing the contents of the line, not including any
    line-termination characters, or {@code null} if the end of the stream has been reached.
@throws IOException if an I/O error occurs

---

### Функция ID: 300

Create an instance of the ApiException that contains the given error message.
@param message    The message string to set.
@return           The exception.

---

### Функция ID: 301

Matches and returns the ranges of any named captures.
@param text the text to match and extract values from.
@return a map containing field names and their respective ranges that matched or null if the pattern didn't match

---

### Функция ID: 302

Parse the arguments and run a suitable command.
@param args the arguments
@return the outcome of the command
@throws Exception if the command fails

---

### Функция ID: 303

Invokes a long supplier, and returns the result.
@param supplier The long supplier to invoke.
@param <E> The type of checked exception, which the supplier can throw.
@return The long, which has been created by the supplier

---

### Функция ID: 304

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

---

### Функция ID: 305

Compares two objects for equality.
@param obj  the object to compare to.
@return {@code true} if equal.

---

### Функция ID: 306

Gets the String content that the tokenizer is parsing.
@return the string content being parsed.

---

### Функция ID: 307

Gets the maximum of three {@code int} values.
@param a value 1.
@param b value 2.
@param c value 3.
@return the largest of the values.

---

### Функция ID: 308

Wrap the function to enable memoization with replay and record.

Args:
    fn: The function to wrap.

Returns:
    A wrapped version of the function.

---

### Функция ID: 309

Encodes {@code value} to this stringer.
@param value the value to encode
@return this stringer.
@throws JSONException if processing of json failed

---

### Функция ID: 310

Checks whether this range is after the specified element.
@param element  the element to check for, null returns false.
@return true if this range is entirely after the specified element.

---

### Функция ID: 311

Static method that can be used to find a single command from a collection.
@param commands the commands to search
@param name the name of the command to find
@return a {@link Command} instance or {@code null}.

---

### Функция ID: 312

Return a {@link String} containing the printed stack trace for a given
{@link Throwable}.
@param throwable the throwable that should have its stack trace printed
@return the stack trace string

---

### Функция ID: 313

Returns the value at {@code index} if it exists and is a long or can be coerced to
a long.
@param index the index to get the value from
@return the {@code value}
@throws JSONException if the value at {@code index} doesn't exist or cannot be
coerced to a long.

---

### Функция ID: 314

Get the table's primary key.

:param table: Name of the target table
:param schema: Name of the target schema, public by default
:return: Primary key columns list

---

### Функция ID: 315

Converts an array of primitive bytes to objects.
<p>This method returns {@code null} for a {@code null} input array.</p>
@param array  a {@code byte} array.
@return a {@link Byte} array, {@code null} if null array input.

---

### Функция ID: 316

Returns the minimum value in an array.
@param array an array, must not be null or empty.
@return the minimum value in the array.
@throws NullPointerException     if {@code array} is {@code null}.
@throws IllegalArgumentException if {@code array} is empty.
@since 3.4 Changed signature from min(byte[]) to min(byte...).

---

### Функция ID: 317

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

---

### Функция ID: 318

Set ``is_active=False`` on the DAGs for which the DAG files have been removed.

:param bundle_name: bundle for filelocs
:param rel_filelocs: relative filelocs for bundle
:param session: ORM Session
:return: True if any DAGs were marked as stale, False otherwise

---

### Функция ID: 319

Asynchronous variant of `top()` allowing for splitting up work in batches between which the event loop can run.
Returns the top N elements from the array.
Faster than sorting the entire array when the array is a lot larger than N.
@param array The unsorted array.
@param compare A sort function for the elements.
@param n The number of elements to return.
@param batch The number of elements to examine before yielding to the event loop.
@return The first n elements from array when sorted with compare.

---

### Функция ID: 320

Retrieves the first value that is defined in an array, going backwards from an index position.
To avoid repeating the same data (as when the "format" and "standalone" forms are the same)
add the first value to the locale data arrays, and add other values only if they are different.
@param data The data array to retrieve from.
@param index A 0-based index into the array to start from.
@returns The value immediately before the given index position.
@see [Internationalization (i18n) Guide](guide/i18n)

---

### Функция ID: 321

Removes and returns the value at {@code index}, or null if the array has no value
at {@code index}.
@param index the index of the value to remove
@return the previous value at {@code index}

---

### Функция ID: 322

Checks the contents of this builder against another to see if they
contain the same character content ignoring case.
@param other  the object to check, null returns false
@return true if the builders contain the same characters in the same order

---

### Функция ID: 323

Flatten the map keys using period separator.
@param map the map that should be flattened
@return the flattened map

---

### Функция ID: 324

@param sValue       Value to parse, which may be {@code null}.
@param defaultValue Value to return if {@code sValue} is {@code null}.
@param settingName  Name of the parameter or setting. On invalid input, this value is included in the exception message. Otherwise,
                    this parameter is unused.
@return The {@link TimeValue} which the input string represents, or {@code defaultValue} if the input is {@code null}.

---

### Функция ID: 325

Checks the contents of this builder against another to see if they
contain the same character content.
@param other  the object to check, null returns false
@return true if the builders contain the same characters in the same order

---

### Функция ID: 326

Helper method for generating a hash code for a member of an annotation.
@param name the name of the member
@param value the value of the member
@return a hash code for this member

---

### Функция ID: 327

Check whether calling a function raised a ``TypeError`` because
the call failed or because something in the factory raised the
error.

:param f: The function that was called.
:return: ``True`` if the call failed.

---

### Функция ID: 328

Returns {@code true} if this element is an ancestor (immediate or nested parent) of
the specified name.
@param name the name to check
@return {@code true} if this name is an ancestor

---

### Функция ID: 329

@return True if the member is preparing to leave the group (waiting for callbacks), or
leaving (sending last heartbeat). This is used to skip proactively leaving the group when
the poll timer expires.

---

### Функция ID: 330

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

---

### Функция ID: 331

Get the S3 bucket name and key.

From either:
- bucket name and key. Return the info as it is after checking `key` is a relative path.
- key. Must be a full s3:// url.

:param bucket: The S3 bucket name
:param key: The S3 key
:param bucket_param_name: The parameter name containing the bucket name
:param key_param_name: The parameter name containing the key name
:return: the parsed bucket name and key

---

### Функция ID: 332

Invokes a boolean supplier, and returns the result.
@param supplier The boolean supplier to invoke.
@param <E> The type of checked exception, which the supplier can throw.
@return The boolean, which has been created by the supplier

---

### Функция ID: 333

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

---

### Функция ID: 334

Invokes an int supplier, and returns the result.
@param supplier The int supplier to invoke.
@param <E> The type of checked exception, which the supplier can throw.
@return The int, which has been created by the supplier

---

### Функция ID: 335

Remove a proxied interface.
<p>Does nothing if the given interface isn't proxied.
@param ifc the interface to remove from the proxy
@return {@code true} if the interface was removed; {@code false}
if the interface was not found and hence could not be removed

---

### Функция ID: 336

Close this builder and return the resulting buffer.
@return The built log buffer

---

### Функция ID: 337

Tests whether the {@code /proc/N/environ} file at the given path string contains a specific line prefix.
@param envVarFile The path to a /proc/N/environ file.
@param key        The env var key to find.
@return value The env var value or null.

---

### Функция ID: 338

Finds the most frequently occurring item.
@param <T> type of values processed by this method.
@param items to check.
@return most populous T, {@code null} if non-unique or no items supplied.
@since 3.0.1

---

### Функция ID: 339

Returns the filter name that will be registered.
@return the filter name
@since 3.2.0

---

### Функция ID: 340

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

---

### Функция ID: 341

Returns the value mapped by {@code name} if it exists and is a double or can be
coerced to a double.
@param name the name of the property
@return the value
@throws JSONException if the mapping doesn't exist or cannot be coerced to a
double.

---

### Функция ID: 342

Gets the base timestamp of the batch which is used to calculate the record timestamps from the deltas.
@return The base timestamp

---

### Функция ID: 343

Create a new {@link Options} instance that contains the options in this set
including the given option.
@param option the option to include
@return a new {@link Options} instance

---

### Функция ID: 344

Gets the list of {@link Throwable} objects in the
exception chain.
<p>A throwable without cause will return a list containing
one element - the input throwable.
A throwable with one cause will return a list containing
two elements. - the input throwable and the cause throwable.
A {@code null} throwable will return a list of size zero.</p>
<p>This method handles recursive cause chains that might
otherwise cause infinite loops. The cause chain is processed until
the end, or until the next item in the chain is already
in the result list.</p>
@param throwable  the throwable to inspect, may be null.
@return the list of throwables, never null.
@since 2.2

---

### Функция ID: 345

Multiply two fixed-precision rational numbers.
@param a
@param b
@return Returns a * b

---

### Функция ID: 346

Create an array of primitive type from an array of wrapper types.
<p>
This method returns {@code null} for a {@code null} input array.
</p>
@param array  an array of wrapper object.
@return an array of the corresponding primitive type, or the original array.
@since 3.5

---

### Функция ID: 347

Unset the preferred read replica. This causes the fetcher to go back to the leader for fetches.
@param tp The topic partition
@return the removed preferred read replica if set, Empty otherwise.

---

### Функция ID: 348

Appends an iterable placing separators between each value, but
not before the first or after the last.
Appending a null iterable will have no effect.
Each object is appended using {@link #append(Object)}.
@param iterable  the iterable to append
@param separator  the separator to use, null means no separator
@return {@code this} instance.

---

### Функция ID: 349

Tests whether the specified {@link Throwable} is unchecked and throws it if so.
@param <T> The Throwable type.
@param throwable the throwable to test and throw or return.
@return the given throwable.
@since 3.14.0

---

### Функция ID: 350

Is this method on an introduced interface?
@param mi the method invocation
@return whether the invoked method is on an introduced interface

---

### Функция ID: 351

Compares this range to another object to test if they are equal.
<p>To be equal, the minimum and maximum values must be equal, which
ignores any differences in the comparator.</p>
@param obj the reference object with which to compare.
@return true if this object is equal.

---

### Функция ID: 352

Repeats a String {@code repeat} times to form a new String, with a String separator injected each time.
<pre>
StringUtils.repeat(null, null, 2) = null
StringUtils.repeat(null, "x", 2)  = null
StringUtils.repeat("", null, 0)   = ""
StringUtils.repeat("", "", 2)     = ""
StringUtils.repeat("", "x", 3)    = "xx"
StringUtils.repeat("?", ", ", 3)  = "?, ?, ?"
</pre>
@param repeat    the String to repeat, may be null.
@param separator the String to inject, may be null.
@param count     number of times to repeat str, negative treated as zero.
@return a new String consisting of the original String repeated, {@code null} if null String input.
@since 2.5

---

### Функция ID: 353

Gets a {@link List} of superclasses for the given class.
@param cls the class to look up, may be {@code null}.
@return the {@link List} of superclasses in order going up from this one {@code null} if null input.

---

### Функция ID: 354

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

---

### Функция ID: 355

Parse certificates from the specified string.
@param text the text to parse
@return the parsed certificates

---

### Функция ID: 356

Calculate how many Dag Runs will be created with given performance DAG configuration.

:param performance_dag_conf: dict with environment variables as keys and their values as values

:return: total number of Dag Runs
:rtype: int

---

### Функция ID: 357

Get an injectable argument instance for the given type. This method can be used
when manually instantiating an object without reflection.
@param <A> the argument type
@param type the argument type
@return the argument to inject or {@code null}
@since 3.4.0

---

### Функция ID: 358

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

---

### Функция ID: 359

Returns {@code true} if this element is an ancestor (immediate or nested parent) of
the specified name.
@param name the name to check
@return {@code true} if this name is an ancestor

---

### Функция ID: 360

Trims the builder by removing characters less than or equal to a space
from the beginning and end.
@return {@code this} instance.

---

### Функция ID: 361

Returns the value at {@code index} if it exists and is an int or can be coerced to
an int. Returns {@code fallback} otherwise.
@param index the index to get the value from
@param fallback the fallback value
@return the value at {@code index} of {@code fallback}

---

### Функция ID: 362

Create an ignore for a property with the given name.
@param name the name
@return the item ignore

---

### Функция ID: 363

Estimates the rank of a given value in the distribution represented by the histogram.
In other words, returns the number of values which are less than (or less-or-equal, if {@code inclusive} is true)
the provided value.
@param histo the histogram to query
@param value the value to estimate the rank for
@param inclusive if true, counts values equal to the given value as well
@return the number of elements less than (or less-or-equal, if {@code inclusive} is true) the given value

---

### Функция ID: 364

Resolve the cache to use.
@param context the invocation context
@return the cache to use (never {@code null})

---

### Функция ID: 365

Creates a new instance of this Tokenizer. The new instance is reset so that
it will be at the start of the token list.
@return a new instance of this Tokenizer which has been reset.
@throws CloneNotSupportedException if there is a problem cloning.

---

### Функция ID: 366

Returns {@code true} if this element is an ancestor (immediate or nested parent) of
the specified name.
@param name the name to check
@return {@code true} if this name is an ancestor

---

### Функция ID: 367

Returns an array with the values corresponding to {@code names}. The array contains
null for names that aren't mapped. This method returns null if {@code names} is
either null or empty.
@param names the names of the properties
@return the array

---

### Функция ID: 368

Lax-ly parses a string that (ideally) looks like 'AS123' into a Long like 123L (or null, if such parsing isn't possible).
@param asn a potentially empty (or null) ASN string that is expected to contain 'AS' and then a parsable long
@return the parsed asn

---

### Функция ID: 369

Invokes a supplier, and returns the result.
@param supplier The supplier to invoke.
@param <O> The suppliers output type.
@param <T> The type of checked exception, which the supplier can throw.
@return The object, which has been created by the supplier
@since 3.10

---

### Функция ID: 370

Encodes {@code value} to this stringer.
@param value the value to encode
@return this stringer.
@throws JSONException if processing of json failed

---

### Функция ID: 371

Gets a {@link List} of all interfaces implemented by the given class and its superclasses.
<p>
The order is determined by looking through each interface in turn as declared in the source file and following its
hierarchy up. Then each superclass is considered in the same way. Later duplicates are ignored, so the order is
maintained.
</p>
@param cls the class to look up, may be {@code null}.
@return the {@link List} of interfaces in order, {@code null} if null input.

---

### Функция ID: 372

Encodes {@code value} to this stringer.
@param value the value to encode
@return this stringer.
@throws JSONException if processing of json failed

---

### Функция ID: 373

Retrieves the data contained in the regular files named by {@code keys} in the directory given by {@code path}.
Non-regular files (such as directories) in the given directory are silently ignored.
@param path the directory where data files reside.
@param keys the keys whose values will be retrieved.
@return the configuration data.

---

### Функция ID: 374

Gets a fraction that is the positive equivalent of this one.
<p>
More precisely: {@code (fraction &gt;= 0 ? this : -fraction)}
</p>
<p>
The returned fraction is not reduced.
</p>
@return {@code this} if it is positive, or a new positive fraction instance with the opposite signed numerator

---

### Функция ID: 375

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

---

### Функция ID: 376

Tests whether the given type arrays are equal.
@param type1 LHS.
@param type2 RHS.
@return Whether the given type arrays are equal.

---

### Функция ID: 377

If there is no pending task, set the pending task active.
 If wakeup was called before setting an active task, the current task will complete exceptionally with
 WakeupException right away.
 If there is an active task, throw exception.
@param currentTask
@param <T>
@return

---

### Функция ID: 378

Search forward for the first message that meets the following requirements:
- Message's timestamp is greater than or equals to the targetTimestamp.
- Message's position in the log file is greater than or equals to the startingPosition.
- Message's offset is greater than or equals to the startingOffset.
@param targetTimestamp The timestamp to search for.
@param startingPosition The starting position to search.
@param startingOffset The starting offset to search.
@return The timestamp and offset of the message found. Null if no message is found.

---

### Функция ID: 379

Replaces all the occurrences of variables with their matching values
from the resolver using the given source as a template.
The source is not altered by this method.
<p>
Only the specified portion of the buffer will be processed.
The rest of the buffer is not processed, and is not returned.
</p>
@param source  the buffer to use as a template, not changed, null returns null.
@param offset  the start offset within the array, must be valid.
@param length  the length within the array to be processed, must be valid.
@return the result of the replace operation.
@since 3.2

---

### Функция ID: 380

Retrieves the meta-data of the service at the specified URL.
@param url the URL
@return the response

---

### Функция ID: 381

Flatten the map keys using period separator.
@param map the map that should be flattened
@return the flattened map

---

### Функция ID: 382

Returns the output keys produced by the instance (excluding named skip keys),
e.g. for the pattern <code>"%{a} %{b} %{?c}"</code> the result is <code>[a, b]</code>.
<p>
The result is an ordered set, where the entries are in the same order as they appear in the pattern.
<p>
The reference keys are returned with the name they have in the pattern, e.g. for <code>"%{*x} %{&amp;x}"</code>
the result is <code>[x]</code>.
@return the output keys produced by the instance.

---

### Функция ID: 383

Reads a sequence of bytes from this channel into a subsequence of the given buffers.
@param dsts - The buffers into which bytes are to be transferred
@param offset - The offset within the buffer array of the first buffer into which bytes are to be transferred; must be non-negative and no larger than dsts.length.
@param length - The maximum number of buffers to be accessed; must be non-negative and no larger than dsts.length - offset
@return The number of bytes read, possibly zero, or -1 if the channel has reached end-of-stream.
@throws IOException if some other I/O error occurs

---

### Функция ID: 384

Handles the resolution step where esbuild resolves the imports before
bundling them. This allows us to inject a filler via its `path` if it was
provided. If not, we proceed to the next `onLoad` step.
@param fillers to use the path from
@param args from esbuild
@returns

---

### Функция ID: 385

Atomically adds the given value to the element at index {@code i}.
@param i the index
@param delta the value to add
@return the previous value

---

### Функция ID: 386

Returns the value mapped by {@code name} if it exists and is an int or can be
coerced to an int. Returns {@code fallback} otherwise.
@param name the name of the property
@param fallback a fallback value
@return the value or {@code fallback}

---

### Функция ID: 387

Finalize the state of a batch. Final state, once set, is immutable. This function may be called
once or twice on a batch. It may be called twice if
1. An inflight batch expires before a response from the broker is received. The batch's final
state is set to FAILED. But it could succeed on the broker and second time around batch.done() may
try to set SUCCEEDED final state.
2. If a transaction abortion happens or if the producer is closed forcefully, the final state is
ABORTED but again it could succeed if broker responds with a success.
Attempted transitions from [FAILED | ABORTED] --> SUCCEEDED are logged.
Attempted transitions from one failure state to the same or a different failed state are ignored.
Attempted transitions from SUCCEEDED to the same or a failed state throw an exception.
@param baseOffset The base offset of the messages assigned by the server
@param logAppendTime The log append time or -1 if CreateTime is being used
@param topLevelException The exception that occurred (or null if the request was successful)
@param recordExceptions Record exception function mapping batchIndex to the respective record exception
@return true if the batch was completed successfully and false if the batch was previously aborted

---

### Функция ID: 388

If there is no pending task, set the pending task active.
 If wakeup was called before setting an active task, the current task will complete exceptionally with
 WakeupException right away.
 If there is an active task, throw exception.
@param currentTask
@param <T>
@return

---

### Функция ID: 389

Invokes a boolean supplier, and returns the result.
@param supplier The boolean supplier to invoke.
@param <T> The type of checked exception, which the supplier can throw.
@return The boolean, which has been created by the supplier

---

### Функция ID: 390

Check if the transaction is in the prepared state.
@return true if the current state is PREPARED_TRANSACTION

---

### Функция ID: 391

Appends each item in an iterator to the builder without any separators.
Appending a null iterator will have no effect.
Each object is appended using {@link #append(Object)}.
@param it  the iterator to append
@return {@code this} instance.
@since 2.3

---

### Функция ID: 392

Returns {@code true} if this element is an ancestor (immediate or nested parent) of
the specified name.
@param name the name to check
@return {@code true} if this name is an ancestor

---

### Функция ID: 393

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

---

### Функция ID: 394

Get the relative file location for a given filepath.

:param filepath: Absolute path to the file
:return: Relative path from bundle_path, or original filepath if no bundle_path

---

### Функция ID: 395

Returns {@code true} if this element is an ancestor (immediate or nested parent) of
the specified name.
@param name the name to check
@return {@code true} if this name is an ancestor

---

### Функция ID: 396

Returns the value at {@code index} if it exists and is a double or can be coerced
to a double.
@param index the index to get the value from
@return the {@code value}
@throws JSONException if the value at {@code index} doesn't exist or cannot be
coerced to a double.

---

### Функция ID: 397

Sets the given bucket of the negative buckets. If the bucket already exists, it will be replaced.
Buckets may be set in arbitrary order. However, for best performance and minimal allocations,
buckets should be set in order of increasing index and all negative buckets should be set before positive buckets.
@param index the index of the bucket
@param count the count of the bucket, must be at least 1
@return the builder

---

### Функция ID: 398

Creates a new instance of the class. Required by Log4J2.
@param config the configuration
@param options the options
@return a new instance, or {@code null} if the options are invalid

---

### Функция ID: 399

Get EC2 instance by id and return it.

:param instance_id: id of the AWS EC2 instance
:param filters: List of filters to specify instances to get
:return: Instance object

---

### Функция ID: 400

Tests whether the given types are equal.
@param type1 The first type.
@param type2 The second type.
@return Whether the given types are equal.
@since 3.2

---


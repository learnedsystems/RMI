# RMI

This is a reference implementation of recursive model indexes (RMIs). A prototype RMI was initially described in [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208) by Kraska et al. in 2017.

![Fig 1 from the Case for Learned Index Structures](http://people.csail.mit.edu/ryanmarcus/rmi.png)

## RMI basics

Like binary search trees, an RMI is a structure to help search through sorted data. Given a sorted array, an RMI is a function that maps a key to an approximate index. This approximate index can be used as a starting point for a linear, exponential, or binary search. The [SOSD benchmark](https://learned.systems/sosd) demonstrates that RMIs can outperform binary search and many other standard approaches as well.

Unlike a binary search tree, an RMI uses machine learning techniques to build this approximation function. The result is normally a small, compact mathematical function that can be evaluated quickly. RMIs are a good tool when you need to search the same sorted data many times. Compared to other structures RMIs:

* (➕) Offer faster lookup times (when properly tuned)
* (➕) Are generally much smaller than traditional structures like B-Trees or radix trees
* (➖) Must be trained ahead of time on a dataset
* (➖) Does not support inserts (without retraining the model)

Many more details can be found in [the original paper](https://arxiv.org/abs/1712.01208).

## Using this implementation

To use the reference implementation, clone this repository and [install Rust](https://rustup.rs/).

The reference RMI implementation is a *compiler.* It takes a dataset as input, and produces C/C++ source files as outputs. The data input file must contain:

1. The number of items, as a 64-bit unsigned integer
2. The data items, either 32-bit or 64-bit unsigned integers.

If the input file contains 32-bit integers, the filename must end with `uint32`. If the input file contains 64-bit integers, the filename must end with `uint64`.

In addition to the input dataset, you must also provide a model structure. For example, to build a 2-layer RMI on the data file `books_200M_uint32` (available from [the Harvard Dataverse](https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/JGVF9A/MZZUP2&version=4.0)) with a branching factor of 100, we could run:

```
cargo run --release -- books_200M_uint32 my_first_rmi linear,linear 100
```

This command would produce C/C++ source files in the current directory. The C/C++ sources contain a few publicly-exposed fields:

```C++
#include <cstddef>
#include <cstdint>
namespace my_first_rmi {
    const size_t RMI_SIZE = 12928;
    const uint64_t BUILD_TIME_NS = 13622048811;
    const char NAME[] = "my_first_rmi";
    uint64_t lookup(uint64_t key);
}

```

The `RMI_SIZE` constant represents the size of the constructed model in bytes. The `BUILD_TIME_NS` field records how long it took to build the RMI, in nanoseconds. The `NAME` field is a constant you specify (and always matches the namespace name). The `lookup` function takes in an unsigned, 64-bit integer key and produces an estimate of the offset. 

If you run the compiler with the `-e` flag, the API will change to report the maximum possible error of each lookup.

```c++
uint64_t lookup(uint64_t key, size_t* err);
```

This lookup error can be used to perform a bounded binary search. If the error of the trained RMI is low enough, linear search may give better performance.

Logging useful diagnostic information can be enabled by setting the `RUST_LOG` environmental variable to `trace`: `export RUST_LOG=trace`.

Currently, the following types of RMI layers are supported:

* `linear`, simple linear regression
* `linear_spline`, connected linear spline segments
* `cubic`, connected cubic spline segments
* `loglinear`, simple linear regression with a log transform
* `normal`, normal CDF with tuned mean, variance, and scale.
* `lognormal`, normal CDF with log transform
* `radix`, eliminates common prefixes and returns a fixed number of significant bits based on the branching factor
* `bradix`, same as radix, but attempts to choose the number of bits based on balancing the dataset
* `histogram`, partitions the data into several even-sized blocks (based on the branching factor)
* `plr`, performs piecewise linear regression with a fixed error bound (64)
* `pgm`, constructs a [piecewise geometric model (PGM) index](https://arxiv.org/abs/1910.06169)

Tuning an RMI is critical to getting good performance. A good place to start is a `radix` layer followed by a large linear layer, for example: `radix,linear 262144`. 


This work is freely available under the terms of the MIT license.

### Contributors

* [Ryan Marcus](https://rmarcus.info)

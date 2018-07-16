# dynet-c: Plain-c bindings for DyNet

> **Attention:**
> * This project is still on early stage, more need to be done! Contribution are welcomed!
> * This is a **temporary** placement for dynet c-api, a pull request will be made to the main repository when basic functions are fully implemented!(Of course, after some discusssion and modification)
> * Directly using C api to write dynet program is not so convenient, but you still can if you are fond of C.☺️ Check the example.The main usage of this project is for creating bindings for other languages, such as Rust, Swift or Go.

## Building from source
1. Make sure **clang** is already installed.
2. Build **DyNet** library first. Please follow the instruction in section "C++ installation" on [DyNet project page](https://github.com/clab/dynet).
3. Download this project:
```bash
git clone git@github.com:xbainbain/dynet-c.git
```  
4. Build dynet for c with following instruction:
```bash
clang++ c_api.cc -I/path/to/dyent/ -L/path/to/dynet/build/dynet -ldynet -dynamiclib -o libcdynet.dylib -std=c++11
```
5. (Optional) Build the example:
```bash
clang test.c -o test -lcdynet -L//Users/albain/Developer/ML/dynet/contrib/c
./test
```

## Implementation details
The choice of the exposed API is based on the [Python API](http://dynet.readthedocs.io/en/latest/python_ref.html) for dynet and the end-user usage situation of dynet.

The name convention follows [tensorflow c-api project](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/c).

## To do
- [ ] Basic functions' implementation.
- [ ] Error handling
- [ ] Documentation


## Contribution
Any suggestion and discussion about the choices of API or implementation details are welcomed!




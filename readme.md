# Run code

```shell
!git clone https://github.com/litus324/LTSS-final_project.git
%cd content/LTSS-final_project
!mkdir build
%cd build
!cmake .. -DCMAKE_{C,CXX}_FLAGS="-O3 -march=native"
!make -j$(nproc)
```

Run `./demo`.

Result: 

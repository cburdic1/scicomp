Before replacing laplacian:
65.21


real    0m20.705s
user    0m20.643s
sys     0m0.003s
After laplacium:
65.21


real    0m0.180s
user    0m0.176s
sys     0m0.000s


1. How much of a speedup do you get? Why do you think it’s so dramatic? 
Speedup went from 20.705s to 0.180s so it is 115x faster which is 99.13% faster. This reduces any copying of the 2-D vector (a lot of copies were prevented, hence the speedup) and  modifications inside the cell are prevented. Also, the output of optimize is still 65.21
After changing the rows to 800:
5.39
real    0m11.715s
user    0m8.633s
sys     0m3.040s
After fixing other functions:
5.39
real    0m11.168s
user    0m8.301s
sys     0m2.830s


2. How much more of a speedup do these changes yield? Why isn’t it as significant?
When the rows increase to 800 we see a slow down, so then i fixed functions that should take arguments by const reference rather than by value. This speedup goes from 11.715s to 11.168s, so the speedup is about 1.05x faster or a 4.7% speedup. This is not as significant, because the big copies have already been reduced. Then the later tweaks remove far fewer and smaller copies. 
After changing order of iteration on these loops:
5.39


real    0m2.970s
user    0m2.950s
sys     0m0.007s
3.How much faster does optimize run with this change? Why do you think that is?
I did not only make this change here, I also added a non-const overload of interior, because it looks like my earlier code changes were causing things to move slower. This did not make anything ste rby itself, but allowed for writtbalespans and loop reordering properly. Here the speed went from 11/168 s to 2.970 s which is 3.76x faster or 73.4% faster. I think this is this easy because this allows the program to access memory in a more contiguous and cache friendly order, reducing cache misses and speeding things. 


This is the new way to compile:
g++ -std=c++20 -Ofast -o optimize optimize.cpp # the "O" in "-O3" is the letter
time ./optimize
After new way to compile:
5.39

real    0m2.569s
user    0m2.553s
sys     0m0.004s


4. How much of a speedup resulted from changing -O3 to -Ofast?
The speed used to be 2.970s to 2.569 and is 1.156c faster, roughly 15.6%.
After remove Remove the if (...) continue;:
5.39


real    0m1.420s
user    0m1.405s
sys     0m0.006s
5. How much of a speedup results, and why?
This sped things up from 2.569s to 1.420s, which is 1.81 x faster, 44.7% faster. THis got faster because it eliminates per-iteration branch checks and mispreditions, tightens loop bounds, reduces total iterations, and  improves CPU throughput.
New compiler thing:
g++ -Ofast -std=c++20 -fopenmp optimize.cpp -o optimize
export OMP_NUM_THREADS=2  # or 4, 8, 16, etc.
time ./optimize
2:
5.39


real    0m1.431s
user    0m1.409s
sys     0m0.007s


4:
5.39


real    0m1.448s
user    0m1.417s
sys     0m0.008s


8:
5.39


real    0m1.435s
user    0m1.414s
sys     0m0.004s


16:
5.39


real    0m1.428s
user    0m1.410s
sys     0m0.007s
32:
5.39


real    0m1.432s
user    0m1.414s
sys     0m0.006s


6. What sort of speedup do you get when you run with 2 threads? 4? 8? 16? 32? Why do you think the diminishment of returns is so severe? 
The speed for 2=1.42, 4=1.448, 8=1.435, 16=1.428 and 32=1.432. So the speed really remained the same, with minor speed ups until 15, then from 15-32 no speed ups. The diminishment of returns were severe because adding more threads does not increase useful computation, it just increases memory contention and coordination overhead. Also, the workload is already simple and bandwidth-limited.


g++ -Ofast -march=native -mtune=native -flto -std=c++20 -fopenmp optimize.cpp -o optimize


New bash:
rm -f optimize
g++ -Ofast -std=c++20 -fopenmp -march=native -mtune=native optimize.cpp -o optimize
export OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores
time ./optimize


5.39


real    0m0.667s
user    0m5.145s
sys     0m0.012s


7. How much more of a speedup were you able to get? What seemed to help the most, and why do you think that’s the case?
The speed up went from 1.432 s to 0667 s with a speedup of 2.15 s or 53.4% faster speed. The optimizations I did to make this go faster was that OpenMP parallelization to run outer loops on multiple CPU cores and SIMD vectorization to process multiple data elements. Also, using this to compile/run code drastically reduced run time “rm -f optimize
g++ -Ofast -std=c++20 -fopenmp -march=native -mtune=native optimize.cpp -o optimize
export OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores
time ./optimize”. This new compiler+runtime includes previous optimizations but also adds aspects that create more optimizations per cycle and many more cycles are in parallel. 
8. Did you find out anything interesting or unexpected during the course of this optimization?
I was surprised about such small differences making such big differences. Also, I was surprised that the compiler and runtime command sequence makes such a huge difference with run time. I would have thought that what is in the C++ code would make the biggest difference. 


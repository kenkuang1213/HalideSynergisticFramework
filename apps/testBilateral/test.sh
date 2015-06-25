#!/bin/bash
#for ((i=10;i<100;i=i+10))
#do
#	./bin/cuda/run ../images/gray.png 0.1 out.png $i &> cuda_$i.txt
#done
for ((i=10;i<100;i=i+10))
do
        ./bin/release/run ../images/gray.png 0.1 out.png $i &> opencl_$i.txt
done


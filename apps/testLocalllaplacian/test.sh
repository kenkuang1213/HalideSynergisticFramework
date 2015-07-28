#!/bin/bash
#for ((i=10;i<100;i=i+10))
#do
#	./bin/cuda/run ../images/rgb.png 8 1 1 out.png $i &> cuda_$i.txt
#done
for ((i=10;i<100;i=i+10))
do
        ./bin/release/run ../images/rgb.png 8 1 1 out.png $i -n &> opencl_$i.txt
done


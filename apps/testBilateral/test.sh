#!/bin/bash
for ((i=10;i<100;i=i+10))
do
	./bin/release/run ../images/gray.png 0.1 out.png $i &> $i.txt
done

#!/bin/bash
adb push bin/android/run /data/ken
for ((i=10;i<100;i=i+10))
do
	adb shell LD_LIBRARY_PATH=/data/ken ./data/ken/run /data/ken/gray.png 0.1 /data/ken/out.png $i -n &>android_$i.txt
done

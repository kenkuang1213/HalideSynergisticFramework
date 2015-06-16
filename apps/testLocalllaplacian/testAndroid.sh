#!/bin/bash
make Android
adb push bin/android/run /data/ken/
for ((i=10;i<100;i=i+10))
do
	adb shell LD_LIBRARY_PATH=/data/ken ./data/ken/run /data/ken/rgb.png 8 1 1  /data/ken/out.png $i &>android_$i.txt
done

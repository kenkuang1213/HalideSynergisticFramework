#!/usr/bin/env python
import re

import argparse
parser=argparse.ArgumentParser(prog="profile.py")
parser.add_argument("--file",help="file Name",required=True)
args=parser.parse_args()
f=open(args.file,'r')
contents= [i for i in f]

contents_sec=[]
APIName=set()
APITime={}
start=False
for i in range(0,len(contents)):
	#print contents[i]
	if contents[i]=="sec\r\n" or contents[i]=="sec\r" or contents[i]=="sec\n" : 

		if start:
			break
		else :
			start = True
	if start == True:
		contents_sec.append(contents[i])


title=""
for i in range(0,len(contents_sec)):
	tmp=contents_sec[i].split()
	if tmp:
		if tmp[0] == "CL:":
			title=tmp[1]
		elif tmp[0]=="CUDA:":
			title=tmp[1]
		elif tmp[0]=="Time:":
			if title not in APITime:
				APITime[title]=0
			APITime[title]+=float(tmp[1])
totalTime=0
for i in APITime:
	print i 
	totalTime+=APITime[i]
for i in APITime:
	print str(APITime[i])
print "Total time : "+str(totalTime)

###
'''
Title : how to get whole_list.txt
step1 : download OTB from shell blow 
      eg , --OTB 
            --Basketball
            --...
            --Woman
            --datasets.html
            --files.txt
            --whole_list_script.py
step2 :python whole_list_script.py

cat whole_list.txt 
###
'''

import os
## how to get file_list.txt
'''
#!/bin/bash
baseurl="http://cvlab.hanyang.ac.kr/tracker_benchmark"
wget "$baseurl/datasets.html"
cat datasets.html | grep '\.zip' | sed -e 's/\.zip".*/.zip/' | sed -e s'/.*"//' >files.txt
cat files.txt | xargs -n 1 -P 8 -I {} wget -c "$baseurl/{}"
'''
root = "."
li=[]
with open("files.txt","r") as f:
    li = f.readlines()
    li = [x.strip() for x in li ]

    #li =[x.split("/")[-1].split(".")[0] for x in li ]
    li =[x.split("/")[-1].split(".")[0] for x in li ]


print ("total find %d \n"%(len(li)))

def getimg_count(root,seq):
    fn0 = os.path.join(root,seq,"groundtruth_rect.txt")
    fn1 = os.path.join(root,seq,"groundtruth_rect.1.txt")
    fn2 = os.path.join(root,seq,"groundtruth_rect.2.txt")
    for fn in [fn0,fn1,fn2,None]:
        if os.path.isfile(fn) and os.stat(fn).st_size>0 :
            break
        if fn is None:
            raise Exception ("hi guy ,download zip in wrong format::"+seq)
            
    ''' 
    human4 
     groundtruth_rect.1.txt
     groundtruth_rect.2.txt
    '''
    print (fn)
    with open(fn,"r") as f :
        fl = f.readline()
    return len(fl)

with open ("whole_list.txt","w") as f2 :
    for seq in li :
        c = getimg_count(root ,seq)
        fn = os.path.abspath(os.path.join(root, seq))
        f2.write("%s\t%d\n"%(fn,c))

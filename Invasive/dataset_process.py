imglist_file=open('train_imglist.lst', 'w')
i = 0
for line in open('train_labels.csv'):   
    if i == 0:
        i+=1
        continue
    i+=1
    splitted = line.strip().split(",")
    imglist_file.write(splitted[0]+"\t"+splitted[1]+"\t"+splitted[0]+".jpg"+"\n")

imglist_file.close()

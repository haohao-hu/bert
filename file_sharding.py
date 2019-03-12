#coding=utf-8
def read_rdc_train(number_of_sharded_files,is_train=True):
    if is_train:
        f = open("rdc_dataset/train.tsv") #,encoding="utf-8")
        total_len=600000 #, encoding="latin-1")
        sharded_folder="sharded_train"
    else:
        f = open("rdc_dataset/val.tsv") #,encoding="utf-8")
        total_len=200000
        sharded_folder="sharded_val"
    rows = []
    index = []
    item_id=1

    len_sharded=total_len/number_of_sharded_files
    fwriter=[[] for i in range(number_of_sharded_files)]
    for i in range(number_of_sharded_files):
        fwriter[i] = open("%s/%d.tsv" % (sharded_folder,i), 'w')

    for j,line in enumerate(f):
        for k in range(number_of_sharded_files):
            if j>=k*len_sharded and j<(k+1)*len_sharded:
                fwriter[k].write('%s' % (line))
                break
        #f2.write('%s' % (line))

read_rdc_train(5, False)

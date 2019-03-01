import os

path = "flickr30k_images"
for (_, _, fnames) in os.walk(path):
    newpath = "flickr_train"
    for i in range(6000):
        fname = fnames[i]
        fpath = path + "/" + fname
        os.rename(fpath, newpath + "/" + fname)

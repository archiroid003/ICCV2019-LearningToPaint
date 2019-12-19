import os
import cv2
import numpy as np
import struct

train_image = 'data/kkanji/train-images-idx3-ubyte'
train_label = 'data/kkanji/train-labels-idx1-ubyte'
# test_image = 't10k-images-idx3-ubyte'
# test_label = 't10k-labels-idx1-ubyte'

# for f in [train_image, train_label, test_image, test_label]:
# 	os.system('wget --no-check-certificate http://yann.lecun.com/exdb/mnist/%s.gz' % (f,))
	
# for f in [train_image, train_label, test_image, test_label]:
# 	os.system('gunzip %s.gz' % (f,))


def get_data(img_file, label_file):
    with open(img_file, 'rb') as file:
        # ヘッダ16バイトを読みこみ、4つの uint8 として解釈する。
        magic, num, rows, cols = struct.unpack(">4I", file.read(16))
        print("magic={}, num={}, rows={}, cols={}".format(magic, num, rows, cols))
 
        # 残り全部を読み込み、1次元配列を作成した後、num x rows x cols に変形する。
        imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
 
    with open(label_file, 'rb') as file:
        # ヘッダ8バイトを読みこみ、2つの uint8 として解釈する。
        magic, num = struct.unpack(">2I", file.read(8))
        print("magic={}, num={}".format(magic, num))
 
        # 残り全部を読み込み、1次元配列を作成する。
        labels = np.fromfile(file, dtype=np.uint8)
    # return [{"img": img, "label": label} for img, label in zip(imgs, labels)]
    return imgs,labels


# for image_f, label_f in [(train_image, train_label), (test_image, test_label)]:
for image_f, label_f in [(train_image, train_label)]:
	# with open(image_f, 'rb') as f:
	# 	images = f.read()
	# with open(label_f, 'rb') as f:
	# 	labels = f.read()
        
	# images = [ord(d) for d in images[16:]]
	# images = np.array(images, dtype=np.uint8)
	# images = images.reshape((-1,28,28))

	images, labels = get_data(image_f, label_f)

    print(labels)
    
	outdir = image_f + "_folder"
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	for k,image in enumerate(images):
		cv2.imwrite(os.path.join(outdir, '%05d.png' % (k,)), image)
	
	labels = [outdir + '/%05d.png %d' % (k, ord(l)) for k,l in enumerate(labels[8:])]
	with open('%s.txt' % label_f, 'w') as f:
		f.write(os.linesep.join(labels))
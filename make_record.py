import argparse
import cv2
import os

p = argparse.ArgumentParser()
p.add_argument("--images-path", required=True, type=str, help='input folder with images')
p.add_argument("--record-path", required=True, type=str,  help='output TFRecord')
p.add_argument("--depth", required=False, type=int, choices=set((3, 1)), default=3, help='image output depth')
p.add_argument("--resize", required=False, type=str, help='image output size wxh')
args = p.parse_args()

resize = args.resize
depth = args.depth
images_path = args.images_path
record_path = args.record_path

if resize is not None:
	width, height = [int(val) for val in resize.split('x')]

with open(record_path, 'w') as record:
	for img_path in os.listdir(images_path):
		
		img=cv2.imread(os.path.join(images_path, img_path))

		if depth == 1:
			img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				
		if resize is not None:
			img=cv2.resize(img, (width, height))

		record.write(img.tostring())

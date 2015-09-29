from PIL import Image 
import os
import sys

if __name__ == "__main__":
	if(len(sys.argv) != 4):
		print("the number of parameters must be 3, like this(if you want to convert image to png format):\n  ConvertImageFormat JpegDir PngDir png\n")
		exit()	
	if(sys.argv[1][0] != "/"):
		src_dir = os.getcwd()+"/"+sys.argv[1]
	else: 
		src_dir = sys.argv[1]
	if(sys.argv[2][0] != "/"):
		dst_dir = os.getcwd()+"/"+sys.argv[2]
	else:
		dst_dir = sys.argv[2]

	print("converting from " + src_dir + " to " + dst_dir)
	src_lists = os.listdir(src_dir)
	for src_list in src_lists:
		print("processing " + src_list + "...\n")
		img = Image.open(src_dir+"/"+src_list)
		img.save(dst_dir+"/"+src_list.split(".")[0]+"."+sys.argv[3])		

	print("Done!\n")

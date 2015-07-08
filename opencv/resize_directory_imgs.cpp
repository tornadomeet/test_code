/*
	use opencv resizing images in one directory to another directory, and save another image type
	you can set the resized width, height, image type
	up now, we can only support one-level mkdir, not recurive
	like: tresize_directory_imgs src_dir dst_dir 48 56 jpg

TODO: support recurive mkdir if directory in src_dir is recurive exist 
*/

#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>  // GetDir
#include <sys/stat.h>  // Mkdir 
#include <sys/stat.h>  // Mkdir
//#include <libgen.h>  // dirname, not very friendly
#include <opencv2/core/core.hpp>
#include <highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

#define RWIDTH 48
#define RHEIGHT 48

void GetDir(const string &dir, vector<string> *files) {
        DIR *dp;
	struct dirent *dirp;
        if((dp = opendir(dir.c_str())) == NULL) {
        	std::cout << "Error open " << dir << std::endl; 
        }   
        while((dirp = readdir(dp)) != NULL && string(dirp->d_name) != ".." && string(dirp->d_name) != ".") {
		if(dirp->d_type == DT_DIR) {
			std::cout << "in subdirectory" << std::endl;
			GetDir(string(dir + "/" + dirp->d_name), files);	
		} else {
	                files->push_back(string(dir + "/" + dirp->d_name)); 
		}
        } 
        closedir(dp);
} 

// cannot create recursion directory
void Mkdir(const string &dirname)
{
	struct stat st = {0};
	if(stat(dirname.c_str(), &st) == -1) {
		std::cout << "creating directory " << dirname << " ";
		mkdir(dirname.c_str(), 0755);
	}
}

void ReizeDirImages(const string &in_dirname, const string &out_dirname, int rwidth, int rheight, string type)
{
	vector<string> files;
	GetDir(in_dirname, &files);
	for(unsigned int i = 0; i < files.size(); i++) {
		string in_filename = files[i];	
		std::cout << "Processing " << files[i]; 

		cv::Mat img = cv::imread(in_filename);
		if(img.empty()) {
			std::cout << "open image " << in_filename << " failed." << std::endl;
		}
		cv::resize(img, img, cv::Size(rwidth, rheight));

		size_t found = in_filename.find("/");  // 0-based
		string out_filename = out_dirname + "/" + files[i].replace(files[i].begin(), files[i].begin()+found+1, "");	
		found = out_filename.find(".");
		out_filename = out_filename.replace(out_filename.begin()+found+1, out_filename.end(), type);

		// 1:
		found = out_filename.rfind("/");  // 0-based
		string create_dir = out_filename.substr(0, found);

		// 2: why dirname change the value of out_filename?
		//string tmp_filename(out_filename);  // deep copy
		//string create_dir = dirname(const_cast<char*>(tmp_filename.c_str()));

		//std::cout << ", create_dir = " << create_dir;
		Mkdir(create_dir);
		std::cout << ", saving " << out_filename;
		cv::imwrite(out_filename, img);
		std::cout << "\tDone" << std::endl;
	}
}


int main(int argc, char **argv)
{
	if(argc != 3 && argc != 4 && argc !=5 && argc != 6) {
		std::cout << "\terror! the number of parameter is not correct, it's should be 3 or 4 or 5 or 6" << std::endl;
		std::cout << "\tusage:\tresize_directory_imgs train_img train(default width=height=48, type=jpg)" << std::endl;
		std::cout << "\t      \ttresize_directory_imgs train_img train 48(default heigh=48, type=jpg)" << std::endl;
		std::cout << "\t      \ttresize_directory_imgs trian_img train 48 56(default type=jpg)" << std::endl;
		std::cout << "\t      \ttresize_directory_imgs trian_img train 48 56 jpg" << std::endl;
		exit(0);
	}
	string in_dirname = string(argv[1]);
	string out_dirname = string(argv[2]);
	string type = "jpg";

	int rwidth = RWIDTH, rheight = RHEIGHT;
	if(argc == 4) { rwidth = atoi(argv[3]); rheight = rwidth; }
	if(argc == 5) { rwidth = atoi(argv[3]); rheight = atoi(argv[4]); }
	if(argc == 6) { rwidth = atoi(argv[3]); rheight = atoi(argv[4]); type = argv[5]; }
	ReizeDirImages(in_dirname, out_dirname, rwidth, rheight, type);

	std::cout << "success!" << std::endl;
	return 0;
}

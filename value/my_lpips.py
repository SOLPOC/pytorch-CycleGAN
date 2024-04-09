import argparse
import os
import lpips
def get_lpips(dir):
    rename("results/"+dir+"/test_latest/images/B_fake",
           "results/"+dir+"/test_latest/images/B_real")
#     rename("results/shuimo_unet/test_latest/images/A_real","results/shuimo_unet/test_latest/images/B_real")
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--dir0', type=str, default='results/shuimo_unet/test_latest/images/B_fake')
#     parser.add_argument('--dir1', type=str, default='results/shuimo_unet/test_latest/images/B_real')
#     parser.add_argument('-v', '--version', type=str, default='0.1')
#     opt = parser.parse_args()
    dir0="results/"+dir+"/test_latest/images/B_fake"
    dir1="results/"+dir+"/test_latest/images/B_real"
    ## Initializing the model
#     loss_fn = lpips.LPIPS(net='alex', version=opt.version)
    loss_fn = lpips.LPIPS(net='alex', version="0.1")
    # the total list of images
#     files = os.listdir(opt.dir0)
    files = os.listdir(dir0)
    i = 0
    total_lpips_distance = 0
    average_lpips_distance = 0

    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(dir+"\n")

    for file in files:

        try:
            # Load images
#             img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0, file)))
#             img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1, file)))
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(dir0, file)))
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(dir1, file)))
#             if (os.path.exists(os.path.join(opt.dir0, file)), os.path.exists(os.path.join(opt.dir1, file))):
            if (os.path.exists(os.path.join(dir0, file)),
                os.path.exists(os.path.join(dir1, file))):
                i = i + 1

            # Compute distance
            current_lpips_distance = loss_fn.forward(img0, img1)
            total_lpips_distance = total_lpips_distance + current_lpips_distance

            print('%s: %.4f' % (file, current_lpips_distance))

            with open("results.txt", "a", encoding="utf-8") as f:
                f.write('%s: %.4f' % (file, current_lpips_distance))
                f.write("\n")




        except Exception as e:
            print(e)

    average_lpips_distance = float(total_lpips_distance) / i
#     print(average_lpips_distance)
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write("average : ")
        f.write('%.4f' % average_lpips_distance)
        f.write("\n\n\n")

def rename( dir1 = "/path/to/directory1", dir2 = "/path/to/directory2"):


    # Get the list of files in each directory
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    # Make sure the number of files in each directory is the same
    if len(files1) != len(files2):
        print("The number of files in each directory is not the same. Cannot proceed.")
    else:
        # Iterate over each file in the first directory
        for i in range(len(files1)):
            # Get the old and new file paths
            old_file_path = os.path.join(dir1, files1[i])
            new_file_path = os.path.join(dir1, files2[i])

            # Rename the file
            os.rename(old_file_path, new_file_path)

        # print("Files in directory1 have been successfully renamed to match the files in directory2.")
if __name__ == '__main__':
    dir_list = os.listdir("./results")
    for subdir in dir_list:
#         print(subdir)
        get_lpips(subdir)
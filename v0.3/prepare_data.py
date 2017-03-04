import glob
import os
import fnmatch
from PIL import Image
import SimpleITK as sitk
import numpy as np
import pylab
import matplotlib.pyplot as plt

class BRATS(object):
    """
    flow of class working
    1. read files
    2. make patches of files
        2.1 make file name of each patch that also attaches the class
    3. read patches and their labels.
        3.1 convert the class data to one-hot encoding
    """
        # 'BRATS-Training/**/**/**/*.mha'
    def __init__(self):
        self.train_data_path= ''   # to be used with glob package
        self.test_data_path=''

    def set_training_path(self,path):
        self.train_data_path=path

    def get_training_path(self):
        return self.train_data_path

    def set_test_path(self,path):
        self.test_data_path=path

    def get_test_path(self):
        return self.test_data_path

    def read_train_files(self):
        labels = glob.glob(self.train_data_path)
        file_names_t1 = []
        file_names_t2=[]
        file_names_t1c=[]
        file_names_flair=[]
        file_names_gt=[]

        # we suppose that images are read in sequence. Afte 4 files, 5th is ground truth
        for i in range(len(labels)):
            fname = os.path.basename(labels[i])
            if fnmatch.fnmatch(labels[i], '*T1c.*'):
                file_names_t1c.append(labels[i])
            elif fnmatch.fnmatch(labels[i], '*T1.*'):
                file_names_t1.append(labels[i])
            elif fnmatch.fnmatch(labels[i], '*T2.*'):
                file_names_t2.append(labels[i])
            elif fnmatch.fnmatch(labels[i], '*Flair.*'):
                 file_names_flair.append(labels[i])
            elif fnmatch.fnmatch(labels[i], '*_3more*'):
                 file_names_gt.append(labels[i])

        return [file_names_t1,file_names_t1c,file_names_t2,file_names_flair,file_names_gt]

    def make_train_patches(self,data_files_list, label_map_list,patch_size):  # i.e. patch_size=[25,25]
        """

        steps:   1 -read a picture
                    2- make its slices
                    3- for each slide create 25x25 patch
                    4- save patches with proper name and classes
        """

        # we are taking only two files to test code, in production, it should be full data_files_list etc.
        flist = [data_files_list[0], data_files_list[1]]
        mlist = [label_map_list[0], label_map_list[1]]

        for i in range(len(flist)):
            image = sitk.ReadImage(flist[i])
            image_label = sitk.ReadImage(mlist[i])

            img_arr = sitk.GetArrayFromImage(image)
            label_arr = sitk.GetArrayFromImage(image_label)

            slices = img_arr.shape[0]
            for slice_idx in range(slices):  # dimensions are changed and now first dimention gives depth

                new_cols = np.zeros([240, 10])
                new_rows = np.zeros([10, 250])

                img_slice = img_arr[slice_idx, :, :]  # the image is 240x240, we make it to 250x250
                img_slice = np.hstack((img_slice, new_cols))
                img_slice = np.vstack((img_slice, new_rows))

                label_slice = label_arr[slice_idx, :, :]  # the image is 240x240, we make it to 250x250
                label_slice = np.hstack((label_slice, new_cols))
                label_slice = np.vstack((label_slice, new_rows))

                image_patches = []
                label_patches = []
                patch_class = []
                r, c ,r_sz,c_sz= 0, 0, patch_size[0], patch_size[1]
                rst, ren, cst, cen = 0, r_sz, 0, c_sz

                while (cen <= 250):
                    # here are four cases 1. [row,cols are valid],[rows invalid],[cols invalid],[both invalid]
                    # case 1: both are valid
                    ptim = []
                    ptlbl = []
                    if (ren < 240 and cen < 240):
                        ptim = img_slice[rst:ren, cst:cen]
                        ptlbl = label_slice[rst:ren, cst:cen]
                        cst, cen = cen, cen + c_sz

                    elif (cen > 240 and ren < 240):
                        ptim = img_slice[rst:ren, cst: 250]
                        ptlbl = label_slice[rst:ren, cst: 250]
                        rst, ren = ren, ren + r_sz
                        cst, cen = 0, c_sz

                    elif (cen < 240 and ren > 240):
                        ptim = img_slice[rst:250, cst: cen]
                        ptlbl = label_slice[rst:ren, cst: 250]
                        cst, cen = cen, cen + c_sz
                    else:
                        ptim = img_slice[rst:250, cst:250]
                        ptlbl = label_slice[rst:ren, cst: 250]
                        break;

                    image_patches.append(ptim)
                    try:
                        central_point=np.int(r_sz/2)+1
                        pt_class = ptlbl[central_point, central_point]  # class of image patch = class of map patch center pixel
                        patch_class.append(pt_class)
                    except:
                        print('some error in finding training patch class')
                        exit()

                fn = ''
                # do for all patches of this slice
                for j in range(len(image_patches)):
                    fname = self.make_file_name(data_files_list[i], slice_idx, j, patch_class[j])
                    fn = fname + '.npy'
                    try:
                        np.save('train_patches/' + fn, image_patches[j])
                        #print(fn)
                    except:
                        print('some error in saving training patches')
                        exit()
            print('training patches created successfully')

    def make_test_patches(self,data_files_list, label_map_list,patch_size):  # i.e. patch_size=[25,25]
        """

        steps:   1 -read a picture
                    2- make its slices
                    3- for each slide create 25x25 patch
                    4- save patches with proper name and classes
        """

        # we are taking only one files for testing, in production, it should be full data_files_list etc.
        flist = [data_files_list[2]]
        mlist = [label_map_list[2]]

        for i in range(len(flist)):
            image = sitk.ReadImage(flist[i])
            image_label = sitk.ReadImage(mlist[i])

            img_arr = sitk.GetArrayFromImage(image)
            label_arr = sitk.GetArrayFromImage(image_label)

            slices = img_arr.shape[0]
            for slice_idx in range(slices):  # dimensions are changed and now first dimention gives depth

                new_cols = np.zeros([240, 10])
                new_rows = np.zeros([10, 250])

                img_slice = img_arr[slice_idx, :, :]  # the image is 240x240, we make it to 250x250
                img_slice = np.hstack((img_slice, new_cols))
                img_slice = np.vstack((img_slice, new_rows))

                label_slice = label_arr[slice_idx, :, :]  # the image is 240x240, we make it to 250x250
                label_slice = np.hstack((label_slice, new_cols))
                label_slice = np.vstack((label_slice, new_rows))

                image_patches = []
                label_patches = []
                patch_class = []
                r, c ,r_sz,c_sz= 0, 0, patch_size[0], patch_size[1]
                rst, ren, cst, cen = 0, r_sz, 0, c_sz

                while (cen <= 250):
                    # here are four cases 1. [row,cols are valid],[rows invalid],[cols invalid],[both invalid]
                    # case 1: both are valid
                    ptim = []
                    ptlbl = []
                    if (ren < 240 and cen < 240):
                        ptim = img_slice[rst:ren, cst:cen]
                        ptlbl = label_slice[rst:ren, cst:cen]
                        cst, cen = cen, cen + c_sz

                    elif (cen > 240 and ren < 240):
                        ptim = img_slice[rst:ren, cst: 250]
                        ptlbl = label_slice[rst:ren, cst: 250]
                        rst, ren = ren, ren + r_sz
                        cst, cen = 0, c_sz

                    elif (cen < 240 and ren > 240):
                        ptim = img_slice[rst:250, cst: cen]
                        ptlbl = label_slice[rst:ren, cst: 250]
                        cst, cen = cen, cen + c_sz
                    else:
                        ptim = img_slice[rst:250, cst:250]
                        ptlbl = label_slice[rst:ren, cst: 250]
                        break;

                    image_patches.append(ptim)
                    try:
                        central_point=np.int(r_sz/2)+1
                        pt_class = ptlbl[central_point, central_point]  # class of image patch = class of map patch center pixel
                        patch_class.append(pt_class)
                    except:
                        print('some error in finding patch class')
                        exit()

                fn = ''
                # do for all patches of this slice
                for j in range(len(image_patches)):
                    fname = self.make_file_name(data_files_list[i], slice_idx, j, patch_class[j])
                    fn = fname + '.npy'
                    try:
                        np.save('test_patches/' + fn, image_patches[j])
                        #print(fn)
                    except:
                        print('some error in saving test patches')
                        exit()
                print('test patches created successfully')

    def make_file_name(self,fname, slice_idx, patch_idx, patch_class):
        fname = os.path.basename(fname)
        fname, _ = os.path.splitext(fname)
        fname = fname + '--S_' + np.str(slice_idx) + '--P_' + np.str(patch_idx) + '--C_' + np.str(np.int(patch_class))
        return fname

    def read_data_n_labels_4m_patches(self,path):
        patches=glob.glob(path+'/*.npy')
        patch_labels=[]
        for i in range(len(patches)):

            fname,ext=os.path.splitext(patches[i])
            pclass=fname[-1]                                        # extract the class label
            patch_labels.append(self.make_1hot_encoding(pclass))

        return [patches,patch_labels]

    def make_1hot_encoding(self,pclass):
        label=[]
        pclass=np.int(pclass)
        if(pclass==0):
            label = [0, 0, 0, 0, 1]
        elif(pclass==1):
            label = [0, 0, 0, 1, 0]
        elif (pclass == 2):
            label = [0, 0, 1, 0, 0]
        elif (pclass == 3):
            label = [0, 1, 0, 0, 0]
        elif (pclass == 4):
            label = [1, 0, 0, 0, 0]
        return label

    def get_test_n_train_data(self,train_path):
        self.set_training_path(train_path)
        tfiles = self.read_train_files()

        # Tc1=tfiles[0], T1=tfiles[1], T2=tfiles[2], Flair=tfiles[3], ground_truth=tfiles[4]
        # if patches are not prepared already, uncomment following two lines
        #self.make_train_patches(tfiles[0], tfiles[4], [25, 25])
        #self.make_test_patches(tfiles[0], tfiles[4], [25, 25])

        train_X, train_Y = self.read_data_n_labels_4m_patches('train_patches')
        test_X, test_Y = self.read_data_n_labels_4m_patches('test_patches')

        return train_X,train_Y,test_X,test_Y

    def read_test_patches(self,test_file_names):
        im_patches = np.empty([0, 625], dtype=float)
        # read images
        for i in range(len(test_file_names)):
            patch_data = np.load(test_file_names[i])  # batch_x contains the file paths
            patch_data = patch_data.flatten()

            im_patches = np.vstack((im_patches, patch_data))
        return im_patches

# ==============================================================================

class Training_batch_iterator(object):
    """
    flow of class work
    1. set training data and label paths
    2. get next batch
    """
    def __init__(self):
        self.batch_pointer = 0
        self.train_data = ''
        self.train_label = ''

    def get_next_batch(self,X,Y,batch_size):
        im_patches = np.empty([0, 625], dtype=float)
        x, y = X, Y
        batch_x = x[self.batch_pointer:self.batch_pointer + batch_size]
        batch_y = y[self.batch_pointer:self.batch_pointer + batch_size]
        self.batch_pointer += batch_size

        # read images
        for i in range(len(batch_x)):
            patch_data = np.load(batch_x[i])     # batch_x contains the file paths
            patch_data = patch_data.flatten()

            im_patches = np.vstack((im_patches, patch_data))

        return im_patches, batch_y















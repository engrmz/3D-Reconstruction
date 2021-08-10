'''

Goals:          Building an data loading pipeline (images, labels from npz files)

Description:    The script loads images from "images" folder
                lables (points, occupancies from npz files)

input::         Images from folder "img_choy2016"
                dataset_dir = 'dataset/mini_shapenet/ShapeNet'


output::        Loads images and corresponding GT points simultaneously


Mohammad Zohaib
PAVIS | IIT, Genova, Italy
mohammad.zohaib@iit.it
engr.mz@hotmail.com
2020/11/15

'''

# CvxNet Code
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


# My updates
import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import matplotlib.pyplot as plt
import numpy as np
import yaml
import random
import csv

#to add
import cv2
import glob
import os
from tqdm import tqdm
from collections import namedtuple
Image = namedtuple('Image', 'rows cols depth data')
Occupancy = namedtuple('Occupancies', 'len data')
Point = namedtuple('Points', 'len coord data')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class DataLoader:
    def __init__(self, split, argv):
        self.total_points = 100000
        self.image_h = argv.image_h
        self.image_w = argv.image_w
        self.image_d = argv.image_d
        self.depth_h = argv.depth_h
        self.depth_w = argv.depth_w
        self.depth_d = argv.depth_d
        self.split = split
        ## /data/mzohaib/code/Initial_cvxnet/graphics/tensorflow_graphics/projects/cvxnet
        # self.dataset_dir = 'dataset/mini_shapenet_delete/ShapeNet'
        # self.dataset_dir = '/data/mzohaib/code/onet_shapenet/ShapeNet'
        # self.dataset_dir = '/data/mzohaib/code/dataset/pix3d_masked/ShapeNet'
        # self.dataset_dir = '/data/mzohaib/code/dataset/masked_images/ShapeNet'
        # self.dataset_dir = '/data/mzohaib/code/dataset/mini_pix3d/yolact_masks'
        # self.dataset_dir = '/data/mzohaib/code/dataset/mini_pix3d_all_segments/ShapeNet_masked'
        ##self.dataset_dir = '/data/mzohaib/code/dataset/icip_comparison/common_yolact_masked/ShapeNet??'
        ##self.dataset_dir = '/data/mzohaib/code/datasets/mini_shapenet/ShapeNet'
        
        ####self.dataset_dir = '/data/mzohaib/code/datasets/Complete_onet_shapenet/onet_rendering'
        # self.dataset_dir = '/data/mzohaib/code/dataset/pix3d_3_categories_yolact/ShapeNet'
        # self.dataset_dir = '/home/vsanguineti/Documents/Code/mini_shapenet/cvxnet_test_/ShapeNet'

        self.dataset_dir = "/data/mzohaib/code/datasets/Complete_onet_shapenet/Test set/mimi_test_set_2"
        ##self.metadata_file = 'metadata_selected.yaml'
        self.metadata_file = 'metadata_selected_2.yaml'
        ##self.metadata_file = 'metadata_onet.yaml'
        self.image_folder = 'img_choy2016'
        self.tfrecords, self.dirlist = self.create_directory()
        self.class_dir = self.get_class_dir()
        self.train_set, self.test_set, self.valid_set = self.split_data()
        self.split_file = open(self.dirlist + '/' + self.split + '_split.txt', 'w')

    def create_directory(self):
        tfrecords_dir = '{}/tf_test_set2'.format(os.path.split(self.dataset_dir)[0])
        list_dir = '{}/list'.format(tfrecords_dir)
        if not os.path.exists(tfrecords_dir):
            os.makedirs(tfrecords_dir)
        if not os.path.exists(list_dir):
            os.makedirs(list_dir)
        return tfrecords_dir, list_dir

    ''' Read "metadata.yaml" file and generates list containing paths of all the classes '''
    def get_class_dir(self):
        with open(os.path.join(self.dataset_dir, self.metadata_file)) as file:
            yaml_file = yaml.full_load(file)
        k = [x for x in yaml_file.keys()]
        return [os.path.join(self.dataset_dir, x) for x in k]

    ''' Read .lst files from all the classes and separates train, test and validation set '''
    def split_data(self):
        train_set = []
        test_set = []
        valid_set = []
        for clas in self.class_dir:
            with open(os.path.join(clas, 'train.lst')) as f:
                train_list = f.read().splitlines()
            with open(os.path.join(clas, 'test.lst')) as f:
                test_list = f.read().splitlines()
            with open(os.path.join(clas, 'val.lst')) as f:
                valid_list = f.read().splitlines()

            for i in [os.path.join(clas, x) for x in train_list]:   train_set.append(i)
            for i in [os.path.join(clas, x) for x in test_list]:   test_set.append(i)
            for i in [os.path.join(clas, x) for x in valid_list]:   valid_set.append(i)
            print("class: {} - train: {}, test: {}, valid: {} ".format(clas, len(train_list), len(test_list),
                                                                       len(valid_list)))
        return [train_set, test_set, valid_set]


    '''Decode the occupancies (labels) 8-bit binary to 100k decimal points'''
    def decode_occupancies(self, data, occ):
        decoded_occupancies = []
        for element in data[occ]:  # Reading encoded occupancies/labels
            binary_list = "{0:08b}".format(element)  # 236 => '11101100'
            int_list = [int(value) for value in binary_list]  # '11101100' => [1,1,1,0,1,1,0,0]
            for digit in int_list:
                decoded_occupancies.append(digit)
        return decoded_occupancies

    ''' Reading data => image, labels, name etc. '''
    def get_data(self, label, filename):
        data = np.load(label)  # Loading npz file
        keys = [key for key in data]
        occupancies = self.decode_occupancies(data, keys[1])  # separating occupancies
        occupancies = np.array(occupancies)
        occupancies = occupancies.astype(np.float32) # TODO -> updated np.unit8 to np.float32
        lenocc = len(occupancies)
        occupanciesserialized = occupancies.tostring()
        points = data[keys[0]]
        points = points.astype(np.float32)
        pointsheight = points.shape[0]
        pointswidth = points.shape[1]
        pointsserialized = points.tostring()  # separating label points
        #print("occ{}point{}{}".format(lenocc, pointswidth, pointsheight))
        dir_lst = filename.split('/')
        last = dir_lst[len(dir_lst) - 3]
        first = dir_lst[len(dir_lst) - 4]
        image_name = first + '-' + last
        return Point(len=pointsheight, coord=pointswidth, data=pointsserialized), Occupancy(len=lenocc, data=occupanciesserialized), image_name, last, first

    def get_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.depth_w, self.depth_h), interpolation=cv2.INTER_CUBIC)
        image_serialized = image.tostring()
        return Image(rows=image.shape[0], cols=image.shape[1], depth=image.shape[2], data=image_serialized)

    def show_data_details(self, img, depth, pts, occ, name):
        inner_points = np.array([pts[i] for i in range(len(pts)) if occ[i] == 1])
        outer_points = np.array([pts[i] for i in range(len(pts)) if occ[i] == 0])


        fig = plt.figure(1)
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(inner_points[:, 2], inner_points[:, 0], inner_points[:, 1], c='r', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(outer_points[:, 2], outer_points[:, 0], outer_points[:, 1], c='b', marker='o')
        plt.axis('off')
        plt.show()

        fig2 = plt.figure(2)
        plt.imshow(img)
        # fig2, axis = plt.subplots(2)
        # fig2.subtitle(name)
        # axis[0].imshow(img)
        # axis[1].imshow(depth)

    '''Load the data and create TFRecords'''
    def to_TFRecord(self):
        # dataset = sorted(glob.glob('{}/*/*/'.format(self.dataset_dir)))
        if self.split == 'train':
            dataset = self.train_set
        elif self.split == 'test':
            dataset = self.test_set
        elif self.split == 'valid':
            dataset = self.valid_set
        else:
            print('*** Wrong Split is selected:: Please select "train", "test" or "valid"***')
            raise Exception('*** Wrong Split is selected:: Please select "train", "test" or "valid"***')

        imgs = []
        path = []
        idn = ''
        classn = ''
        idx=0
        for dir_ in tqdm(dataset):
            img_path = os.path.join(dir_, self.image_folder)
            img_list = [x for x in os.listdir(img_path) if not x.endswith('.npz')]
            img_list.sort()
            img_list = [os.path.join(dir_, self.image_folder, x) for x in img_list]
            pts_path = os.path.join(dir_, "points.npz")
            points, occupancies, image_name, last, first = self.get_data(pts_path, img_list[0])
            imageslist = []
            pathlist = []
            for i in range(len(img_list)):
                imgs.append(img_list[i])
                imageslist.append(self.get_image(img_list[i]))
                pathlist.append(img_list[i])
            if idn!=last:
                idn = last
                out_data_dir = '{}/{}/{}'.format(self.tfrecords, first, last)
                out_filename = '{}/Data.tfrecord'.format(out_data_dir)
                self.split_file.write(out_filename + '\n')
                if not os.path.exists(out_data_dir):
                    os.makedirs(out_data_dir)
                with tf.python_io.TFRecordWriter(out_filename, options=tf.python_io.TFRecordOptions(
                        compression_type=tf.python_io.TFRecordCompressionType.GZIP)) as writer:
                    # Store audio and video data properties as context features, assuming all sequences are the same size
                    feature = {
                        'nameobject': _bytes_feature(str.encode(image_name)),
                    }
                    feature.update({
                            'video/height': _int64_feature(imageslist[0].rows),
                            'video/width': _int64_feature(imageslist[0].cols),
                            'video/depth': _int64_feature(imageslist[0].depth),
                        })
                    feature.update({
                        'point/height': _int64_feature(points.len),
                        'point/width': _int64_feature(points.coord)
                    })
                    feature.update({
                        'occupancies/height': _int64_feature(occupancies.len)
                    })
                    feature_list = {}
                    feature_list.update({
                        'video/image': tf.train.FeatureList(
                            feature=[_bytes_feature(video_image.data) for video_image in imageslist])
                        })
                    feature_list.update({
                        'point/data': tf.train.FeatureList(
                            feature=[_bytes_feature(points.data)])
                    })
                    feature_list.update({
                        'occupancies/data': tf.train.FeatureList(
                            feature=[_bytes_feature(occupancies.data)])
                    })
                    feature_list.update({
                        'image_path': tf.train.FeatureList(
                            feature=[_bytes_feature(str.encode(path)) for path in pathlist])
                    })
                    context = tf.train.Features(feature=feature)
                    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
                    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
                    writer.write(sequence_example.SerializeToString())

        print("Total number of loaded images: ", len(imgs))
        self.split_file.close()


    #
    # '''main function for loading data'''
    # def load_dataset(self):
    #     self.create_TFRecords()
    #         # self.show_data_details(image, depth, points, occupancies, image_name)
    #         # TODO Write this data () to TF record


def create_TFRecords(FLAGS):
   ## DataLoader('train',FLAGS).to_TFRecord()
    DataLoader('test',FLAGS).to_TFRecord()
    # DataLoader('valid',FLAGS).to_TFRecord()


def main():
    flags = tf.app.flags
    tf.logging.set_verbosity(tf.logging.INFO)

    # Dataset flags
    flags.DEFINE_integer("image_h", 224, "The height of the color images.")  # 137
    flags.DEFINE_integer("image_w", 224, "The width of the color images.")  # 137
    flags.DEFINE_integer("image_d", 3, "The channels of color images.")
    flags.DEFINE_integer("depth_h", 224, "The height of depth images.")
    flags.DEFINE_integer("depth_w", 224, "The width of depth images.")
    flags.DEFINE_integer("depth_d", 1, "The number of depth views.")
    flags.DEFINE_integer("n_views", 24, "The number of color images views.")
    flags.DEFINE_string("data_dir", None, "The base directory to load data from.")


    FLAGS = flags.FLAGS
    create_TFRecords(FLAGS)


if __name__ == "__main__":
    main()
    # load_images()




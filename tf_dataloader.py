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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
import numpy as np
import yaml
import random


class DataLoader:
    def __init__(self, split, args):
        self.total_points = 100000
        self.sample_bbx = args.sample_bbx
        self.split = split
        if self.split != "train":
            self.sample_bbx = self.total_points
        self.sample_surf = args.sample_surf
        if self.split != "train":
            self.sample_surf = 0
        self.image_h = args.image_h
        self.image_w = args.image_w
        self.image_d = args.image_d
        self.n_views = args.n_views
        self.depth_h = args.depth_h
        self.depth_w = args.depth_w
        self.depth_d = args.depth_d
        self.batch_size = args.batch_size if self.split == "train" else 1
        # self.dataset_dir = 'dataset/mini_shapenet_delete/ShapeNet'
        # self.dataset_dir = '/data/mzohaib/code/dataset/real_images/ShapeNet'
        # self.dataset_dir = '/data/mzohaib/code/dataset/masked_images/ShapeNet'
        # self.dataset_dir = '/data/mzohaib/code/onet_shapenet/ShapeNet'
        # self.dataset_dir = '/data/mzohaib/code/dataset/pix3d_masked/ShapeNet'
        # self.dataset_dir = '/data/mzohaib/code/dataset/mini_pix3d/yolact_masks'
        # self.dataset_dir = '/data/mzohaib/code/dataset/cvxnet_test_/ShapeNet'

        # self.dataset_dir = '/data/mzohaib/code/dataset/icip_comparison/pix3d_3_categories_yolact/ShapeNet'
        # self.dataset_dir = '/data/mzohaib/code/dataset/icip_comparison/common_yolact_real/ShapeNet_yolact_real'
        ##self.dataset_dir = '/data/mzohaib/code/datasets/mini_shapenet/onet_shapenet/ShapeNet'
        ##self.dataset_dir = '/data/mzohaib/code/datasets/Complete_onet_shapenet/onet_shapenet/ShapeNet'
        self.dataset_dir = '/data/mzohaib/code/datasets/Complete_onet_shapenet/onet_shapenet_1_tele_air/ShapeNet'
       
        ####self.dataset_dir = '/data/mzohaib/code/datasets/Complete_onet_shapenet/onet_shapenet_2/ShapeNet'

        # # self.dataset_dir = 'dataset/test_shapenet/ShapeNet'
        ### /data/mzohaib/code/Initial_cvxnet/graphics/tensorflow_graphics/projects/cvxnet/
        self.list_path = self.get_list_path()  # 'dataset/mini_shapenet/tfrecord/list/listall.txt'
#
    def get_list_path(self):
        if self.split == 'train':
            list_path = '{}/list/train_split.txt'.format(os.path.split(self.dataset_dir)[0])
        elif self.split == 'test':
            list_path = '{}/list/test_split.txt'.format(os.path.split(self.dataset_dir)[0])
        elif self.split == 'valid':
            list_path = '{}/list/valid_split.txt'.format(os.path.split(self.dataset_dir)[0])
        else:
            print('\n\n\n*** Wrong Split is selected:: Please select "train", "test" or "valid"***')
            raise Exception('\n*** Wrong Split is selected:: Please select "train", "test" or "valid"***\n\n\n')

        return list_path

    '''Format the data w.r.t CvxNet requirement'''
    def sampler(self, img, name, occupancies, points, image_path):
        image = tf.reshape(img, [224, 224, self.image_d])
        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        # image = tf.cast(image, tf.float32)
        depth = tf.zeros([224, 224, 1])
        # depth = tf.zeros_like(image)
        # depth = tf.reduce_sum(depth, axis=-1, keep_dimensions=True)
        # depth = tf.reshape(img[:, :, 0], [224, 224, 1])
        labels = []
        pts = []
        if self.split == "train":
            indices_bbx = tf.random.uniform([2048],
                                            minval=0,
                                            maxval=self.total_points,
                                            dtype=tf.int32)
            pts = tf.gather(points, indices_bbx, axis=0)
            pts = tf.reshape(pts, [2048, 3])
            labels = tf.gather(occupancies, indices_bbx, axis=0)
            labels = tf.reshape(labels, [2048, 1])
        else:
            pts = tf.reshape(points, [100000, 3])
            labels = tf.reshape(occupancies, [100000, 1])

        pts = tf.cast(pts, tf.float32)
        labels = tf.cast(labels, tf.float32)

        return {
            "image": image,
            "depth": depth,
            "point": pts,
            "point_label": labels,
            "name": name,
            "path": image_path,
        }

    ''' Read the images path from te list -> 'listall' '''
    def _read_txt_file(self):
        """Read the content of the text file and store it into a list."""
        self.img_paths = []
        with open(self.list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path = line.rstrip('\n')
                self.img_paths.append(img_path)
        if self.split == 'train':   # Shuffling the list for training
            random.shuffle(self.img_paths)
        return len(self.img_paths)

    ''' Shuffle and Repeat the lists for for training '''
    def _shuffle_and_repeat_lists(self, num_epochs, num_samples):
        """Shuffle and repeat the list of paths."""
        self.files = self.files.shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)
        self.files = self.files.repeat(num_epochs)
        self.files = self.files.apply(tf.contrib.data.unbatch())

    ''' Load data from TF Records'''
    def _parse_sequence(self, sequence_example_proto):
        """Input parser for samples of the training set."""

        context_features = {'nameobject': tf.FixedLenFeature([], dtype=tf.string)}
        sequence_features = {}

        context_features.update({
            'video/height': tf.FixedLenFeature([], tf.int64),
            'video/width': tf.FixedLenFeature([], tf.int64),
            'video/depth': tf.FixedLenFeature([], tf.int64)
        })
        sequence_features.update({
            'video/image': tf.FixedLenSequenceFeature([], dtype=tf.string)
        })

        context_features.update({
            'point/height': tf.FixedLenFeature([], tf.int64),
            'point/width': tf.FixedLenFeature([], tf.int64)
        })
        sequence_features.update({
            'point/data': tf.FixedLenSequenceFeature([], dtype=tf.string)
        })

        context_features.update({
            'occupancies/height': tf.FixedLenFeature([], tf.int64)
        })
        sequence_features.update({
            'occupancies/data': tf.FixedLenSequenceFeature([], dtype=tf.string)
        })
        sequence_features.update({
            'image_path': tf.FixedLenSequenceFeature([], dtype=tf.string)
        })
        # Parse single example
        parsed_context_features, parsed_sequence_features = tf.parse_single_sequence_example(sequence_example_proto,
                                                                                             context_features=context_features,
                                                                                             sequence_features=sequence_features)

        objectname = tf.cast(parsed_context_features['nameobject'], tf.string)
        objectname = tf.reshape(objectname, [-1])
        # Retrieve parsed video image features
        video_image_decoded = tf.decode_raw(parsed_sequence_features['video/image'], tf.uint8)
        # Retrieve parsed context features
        video_height = tf.cast(parsed_context_features['video/height'], tf.int32)
        video_width = tf.cast(parsed_context_features['video/width'], tf.int32)
        video_depth = tf.cast(parsed_context_features['video/depth'], tf.int32)
        # Reshape decoded video image
        video_image_shape = tf.stack([-1, video_height, video_width, video_depth])  # 224, 298, 3
        video_images = tf.reshape(video_image_decoded, video_image_shape)

        point_decoded = tf.decode_raw(parsed_sequence_features['point/data'], tf.float32)
        # Retrieve parsed context features
        point_height = tf.cast(parsed_context_features['point/height'], tf.int32)
        point_width = tf.cast(parsed_context_features['point/width'], tf.int32)
        # Reshape decoded video image
        point_shape = tf.stack([-1, point_height, point_width])  # 100000, 3
        points = tf.reshape(point_decoded, point_shape)
        occupancies_decoded = tf.decode_raw(parsed_sequence_features['occupancies/data'], tf.float32) # TODO -> updated tf.uint8 to tf.float32
        # Retrieve parsed context features
        occupancies_height = tf.cast(parsed_context_features['occupancies/height'], tf.int32)
        # Reshape decoded video image
        occupancies_shape = tf.stack([-1, occupancies_height])  # 224, 298, 3
        occupancies = tf.reshape(occupancies_decoded, occupancies_shape)
        #image path
        image_path = tf.cast(parsed_sequence_features['image_path'], tf.string)
        image_path = tf.reshape(image_path, [-1, 1])
        return video_images, objectname, occupancies, points, image_path

    def _map_pick_all(self, video_images, objectname, occupancies, points, image_path):
        """Reply name, occupancies, points"""
        # n = np.shape(image_path)[0]
        print("tf.shape(video_images)[0] :: ",tf.shape(video_images)[0])
        objectname = tf.expand_dims(objectname, 0)
        objectname = tf.tile(objectname, (1, tf.shape(video_images)[0])) # 1, 24
        objectname = tf.reshape(objectname, (-1, 1))

        occupancies = tf.expand_dims(occupancies, 0)
        occupancies = tf.tile(occupancies, (tf.shape(video_images)[0], 1, 1)) # 24, 1, 1
        occupancies = tf.reshape(occupancies, (-1, 100000))

        points = tf.expand_dims(points, 0)
        points = tf.tile(points, (tf.shape(video_images)[0], 1, 1, 1 )) # 24, 1, 1, 1
        points = tf.reshape(points, (-1, 100000, 3))

        return video_images, objectname, occupancies, points, image_path

    '''main function for loading data'''
    def load_dataset(self):
        self.data_size = self._read_txt_file()
        print("\n\nbatch size is: {}".format(self.batch_size))
        print("\nLoading list: {}".format(self.list_path))
        print("\nLoaded TFRecords: {}".format(self.data_size))
        print("\nLoaded Samples: {}\n\n\n\n".format(self.data_size * 24))
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.files = tf.data.Dataset.from_tensor_slices(self.img_paths)

        # # shuffle `num_samples` blocks of files and repeat them `num_epochs`
        # if self.split == "train":
        #     num_epochs = 5 # dummy values
        #     num_samples = 10 # dummy values
        #     self._shuffle_and_repeat_lists(num_epochs, num_samples)
        # create data set
        data = self.files.flat_map(lambda ds: tf.data.TFRecordDataset(ds, compression_type='GZIP'))
        # parse data set
        data = data.map(self._parse_sequence, num_parallel_calls=4)

        ''' prefetch `buffer_size` batches of elements of the dataset'''
        data = data.prefetch(buffer_size=10)
        data = data.map(self._map_pick_all, num_parallel_calls=4)
        data = data.apply(tf.data.experimental.unbatch())
        data = data.map(self.sampler, num_parallel_calls=4)
        if self.split == 'train':
            data = data.shuffle(self.batch_size * 10, reshuffle_each_iteration=True).repeat(-1)  # len(image_list)
        data = data.batch(self.batch_size)
        return data


def show_data_details(img, depth, pts, occ, name):
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
    # plt.show()

    fig2 = plt.figure(2)
    plt.imshow(img)
    plt.show()


def load_images_repeat(FLAGS):
    dataset = DataLoader('train', FLAGS).load_dataset()
    batch = tf.data.Dataset.make_one_shot_iterator(dataset).get_next()

    with tf.Session() as sess:
        batch_no = 0
        while (batch_no < 1000):
            try:
                print("\nEpoch no is: {}\n".format(batch_no))
                batch_val = sess.run(batch)
                batch_val.keys()
                print("\nKeys: ", batch_val.keys())
                print("Image: ", batch_val['image'].shape)
                print("Depth: ", batch_val['depth'].shape)
                print("Point ", batch_val['point'].shape)
                print("Occupancies: ", batch_val['point_label'].shape)
                print("Name: ", batch_val['name'])
                print("Path: ", batch_val['path'])
                # show_data_details(batch_val['image'][0], batch_val['depth'][0], batch_val['point'][0], batch_val['point_label'][0],batch_val['name'][0] )
                batch_no += 1
                # end of batch
            except tf.errors.OutOfRangeError:
                print("Batch has been ended")
                break


            # Image => (16, 224, 224, 3)       dtype=uint8          should be      dtype=float32
            # Depth => (16, 224, 224, 1)       dtype=float32
            # Point => (16, 2048, 3)           dtype=float32
            # Point_label => (16, 2048, 1)     dtype=float32
            # Name => (16, 1)                  dtype=object         should be       (16,)

def main(unused_argv):
    flags = tf.app.flags
    tf.logging.set_verbosity(tf.logging.INFO)
    # Model flags
    flags.DEFINE_float("sharpness", 75., "Sharpness term.")
    flags.DEFINE_integer("n_parts", 50, "Number of convexes uesd.")
    flags.DEFINE_integer("n_half_planes", 25, "Number of half spaces used.")
    flags.DEFINE_integer("latent_size", 256, "The size of latent code.")
    flags.DEFINE_integer("dims", 3, "The dimension of query points.")
    flags.DEFINE_bool("image_input", False, "Use color images as input if True.")
    flags.DEFINE_float("vis_scale", 1.3,
                       "Scale of bbox used when extracting meshes.")
    flags.DEFINE_float("level_set", 0.5,
                       "Level set used for extracting surfaces.")

    # Dataset flags
    flags.DEFINE_integer("image_h", 224, "The height of the color images.")  # 137
    flags.DEFINE_integer("image_w", 224, "The width of the color images.")  # 137
    flags.DEFINE_integer("image_d", 3, "The channels of color images.")
    flags.DEFINE_integer("depth_h", 224, "The height of depth images.")
    flags.DEFINE_integer("depth_w", 224, "The width of depth images.")
    flags.DEFINE_integer("depth_d", 20, "The number of depth views.")
    flags.DEFINE_integer("n_views", 24, "The number of color images views.")
    flags.DEFINE_string("data_dir", None, "The base directory to load data from.")
    flags.mark_flag_as_required("data_dir")
    flags.DEFINE_string("obj_class", "*", "Object class used from dataset.")

    # Training flags
    flags.DEFINE_float("lr", 1e-4, "Start learning rate.")
    flags.DEFINE_string(
        "train_dir", None, "The base directory to save training info and"
                           "checkpoints.")
    flags.DEFINE_integer("save_every", 20000,
                         "The number of steps to save checkpoint.")
    flags.DEFINE_integer("max_steps", 800000, "The number of steps of training.")
    flags.DEFINE_integer("batch_size", 4, "Batch size.")
    flags.DEFINE_integer("sample_bbx", 1024,
                         "The number of bounding box sample points.")
    flags.DEFINE_integer("sample_surf", 1024,
                         "The number of surface sample points.")
    flags.DEFINE_float("weight_overlap", 0.1, "Weight of overlap_loss")
    flags.DEFINE_float("weight_balance", 0.01, "Weight of balance_loss")
    flags.DEFINE_float("weight_center", 0.001, "Weight of center_loss")
    flags.mark_flag_as_required("train_dir")

    # Eval flags
    flags.DEFINE_bool("extract_mesh", False,
                      "Extract meshes and set to disk if True.")
    flags.DEFINE_bool("surface_metrics", False,
                      "Measure surface metrics and save to csv if True.")
    flags.DEFINE_string("mesh_dir", None, "Path to load ground truth meshes.")
    flags.DEFINE_string("trans_dir", None,
                        "Path to load pred-to-target transformations.")
    flags.DEFINE_bool("eval_once", True, "default: False     Evaluate the model only once if True.")

    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS = flags.FLAGS
    load_images_repeat(FLAGS)


if __name__ == "__main__":
    tf.app.run(main)


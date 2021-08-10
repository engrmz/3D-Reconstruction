'''

Tasks:

	dataset:
		Create folder as mentioned in "self.category_id"
		name of the mesh file should be like:  *_onet.obj
	
	1: Create a .json file 
		set =>	self.dataset_dir = 'path/to/meshes/'
		
		path to GT and gt.json files [keep it same]
		self.gt_dir = '/media/mz/mz/Datasets/pix3d/pix3d_full'
        	self.pix3d_json_file = '/media/mz/mz/Datasets/pix3d/pix3d_full/pix3d.json'
        	
        	output: *.json file in self.dataset_dir
        	
        	
        	>>> After creating -> run performance_matrices.py


'''




import sys, os
import numpy as np
import glob
import json
import shutil
#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()
import trimesh
from tqdm import tqdm

class DataLoader:
    def __init__(self):
        self.dataset_dir = '/media/mz/mz/3DV_Results/Output_Meshes/ONet_SS/onet_ss_pix3d/ShapeNet'
        # self.dataset_dir = '/data/mzohaib/code/dataset/comparison_table2/occnet_ShapeNet_real'
        self.category_id = ['03001627','04256520','04379243']
        self.category_name = ['chair', 'sofa', 'table']
            # '/data/mzohaib/code/dataset/mini_pix3d_all_segments/ShapeNet_masked/04379243'

        ##self.gt_dir = '/media/mz/mz/ICIP2021_ServerResults/Dataset/GT_pix3d'

        self.gt_dir = '/media/mz/mz/Datasets/pix3d/pix3d_full'
        self.pix3d_json_file = '/media/mz/mz/Datasets/pix3d/pix3d_full/pix3d.json'

##        self.gt_dir = '/media/mz/mz/Datasets/pix3d_full'
##        self.pix3d_json_file = '/media/mz/mz/ICIP2021_ServerResults/Dataset/mini_pix3d_all_segments/ShapeNet_masked/pix3d.json'
        # self.gt_dir = '/data/mzohaib/code/dataset/GT_pix3d/'
        # self.pix3d_json_file = '/data/mzohaib/code/dataset/mini_pix3d_all_segments/ShapeNet_masked/pix3d.json'
        self.pix3d_data = json.load(open(self.pix3d_json_file))
        self.model_lst = self.get_modellist()  # 'dataset/mini_shapenet/tfrecord/list/listall.txt'
        self.out_put = "{}{}".format(self.gt_dir,self.dataset_dir.split('/')[-1])

    # list of images in shapenet dataset - 1 class only
    def get_modellist(self):
        lst_chair = []
        lst_sofa = []
        lst_table = []
        dataset_chair = sorted(glob.glob('{}/{}/*/*/*'.format(self.dataset_dir,self.category_id[0])))
        dataset_sofa = sorted(glob.glob('{}/{}/*/*/*'.format(self.dataset_dir, self.category_id[1])))
        dataset_table = sorted(glob.glob('{}/{}/*/*/*'.format(self.dataset_dir, self.category_id[2])))
        

  
        for path in dataset_chair:
            pt = path.split('/')
            lst_chair.append(pt[len(pt) - 1].split('_')[0])

        for path in dataset_sofa:
            pt = path.split('/')
            lst_sofa.append(pt[len(pt) - 1].split('_')[0])

        for path in dataset_table:
            pt = path.split('/')
            lst_table.append(pt[len(pt) - 1].split('_')[0])

        return [lst_chair, lst_sofa, lst_table]


    # copy the models w.r.t img_lst
    def create_model_json(self):
        # out_put = "{}{}".format(self.gt_dir, self.dataset_dir.split('/')[-1])
        # chair_models = {}
        # sofa_models = {}
        # table_models = {}
        model_lst = {}
        for x in self.pix3d_data:
            if x['img'].split('/')[1] == self.category_name[0]:
                model_name = x['img'].split('/')[2].split('.')[0]
                if model_name in self.model_lst[0]:
                    model_path = "{}/{}".format(self.gt_dir, x['model'])
                    print('Copying model {}'.format(model_name))
                    model_lst.update({"{}-{}".format(self.category_id[0],model_name):  model_path})
            elif x['img'].split('/')[1] == self.category_name[1]:
                model_name = x['img'].split('/')[2].split('.')[0]
                if model_name in self.model_lst[1]:
                    model_path = "{}/{}".format(self.gt_dir, x['model'])
                    print('Copying model {}'.format(model_name))
                    model_lst.update({"{}-{}".format(self.category_id[1],model_name):  model_path})
            elif x['img'].split('/')[1] == self.category_name[2]:
                model_name = x['img'].split('/')[2].split('.')[0]
                if model_name in self.model_lst[2]:
                    model_path = "{}/{}".format(self.gt_dir, x['model'])
                    print('Copying model {}'.format(model_name))
                    model_lst.update({"{}-{}".format(self.category_id[2],model_name):  model_path})


        self.model_json = open("{}/{}_GT_list.json".format(self.dataset_dir,self.dataset_dir.split('/')[-1]),'w')
        json_object = json.dumps(model_lst, indent=2)
        self.model_json.write(json_object)
        # json_object = json.dumps(sofa_models, indent=4)
        # self.model_json.write(json_object,'\n')
        # json_object = json.dumps(table_models, indent=4)
        # self.model_json.write(json_object,'\n')

        self.model_json.close()
        
        print('files in chair: {}'.format(len(self.model_lst[0])))
        print('files in sofa: {}'.format(len(self.model_lst[1])))
        print('files in table: {}'.format(len(self.model_lst[2])))
        print(" output folder is: {}",format(self.out_put))

    def read_model_json(self):
        self.model_json = json.load(open("{}/{}_GT_list.json".format(self.out_put,self.dataset_dir.split('/')[-1])))
        return self.model_json



    # copy the models w.r.t img_lst
    def separate_pix3d_models(self):

        for i in range(len(self.category_id)):
            out_put = "{}/{}".format(self.out_put , self.category_id[i])
            if not os.path.exists(out_put):
                os.makedirs(out_put)

            print("Saving models of category: {}".format(self.category_name[i]))
            for x in self.pix3d_data:
                if x['img'].split('/')[1] == self.category_name[i]:
                    model_name = x['img'].split('/')[2].split('.')[0]
                    if model_name in self.model_lst[i]:
                        print('Copying model {}'.format(model_name))
                        shutil.copy("{}{}".format(self.gt_dir, x['model']), '{}/{}_gt.obj'
                                    .format(out_put, model_name))



    def convert_meshes_vertices_and_faces(self):
        for i in range(len(self.category_name)):
            input = "{}/model/{}".format(self.gt_dir, self.category_name[i])
            lst = sorted(glob.glob('{}/*/*.obj'.format(input)))
            for shape in tqdm(lst):
                # print('Converting model {}'.format(shape))
                mesh = trimesh.load(shape, force='mesh')
                output_path = shape.replace('GT_pix3d','GT_pix3d_vf')
                output_path = '/'+os.path.join(*output_path.split('/')[0:-1])
                mesh_name = shape.split('/')[-1]
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                mesh.export(output_path+'/'+mesh_name)

        print("\nAll the shapes of \'{}\' have been converted".format(self.category_name[i]))


    def copy_json_to_other(self, source_file, destination_file):
        img_lst = {}
        source_json = json.load(open(source_file))
        source_keys = [k for k in source_json.keys()]
        for key in source_keys:
            print("Copying files {} ".format(source_json[key]))
            img_path = source_json[key]
            path_split = img_path.split('/')
            base_name = path_split[-1].split('_')
            file_name = base_name[0] + '.' + base_name[-1].split('.')[-1]
            output_folder = os.path.join("/", *path_split[0:6], 'common_yolact_real/', *path_split[7:-1])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_img = '{}/{}'.format(output_folder, file_name)
            input_img = os.path.join("/", *path_split[0:10], 'images/', file_name)
            shutil.copy(input_img, output_img)
            img_lst.update({key: output_img})

        output_json = os.path.join("/", *path_split[0:7], destination_file)
        file_writer = open(output_json, 'w')
        json_object = json.dumps(img_lst, indent=2)
        file_writer.write(json_object)
        print('Created   {}     file in folder:    {}\n'.format(destination_file, output_json))
        file_writer.close()
            #



def main():
    # DataLoader().separate_pix3d_models()
    ## DataLoader().convert_meshes_vertices_and_faces()
    DataLoader().create_model_json()

    ### Read yolact-real images path from common_yolact_meshrcnn.json and paste in other direcotry for ONet axecution
    # source = '/data/mzohaib/code/dataset/icip_comparison/pix3d_3_categories_yolact/pix3d_3_categories_yolact.json'
    # destination = 'pix3d_3_categories_yolact_real.json'
    # DataLoader().copy_json_to_other(source, destination)


    # models_dic = DataLoader().read_model_json()
    # print(models_dic)

if __name__ == "__main__":
    main()
    # load_images()

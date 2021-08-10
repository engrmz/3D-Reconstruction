# 3D-Reconstruction


CvxNet related files:

  Create_TFrecords.py:        Creating TFRecords for of the dataset (ONet renderings)
  tf_dataloader.py:           Loading the created TFRecord version of the dataset
  train_initial.py:           Training the CvxNet using the above dataloader pipeline
  performance_matrices.py:    Comparing the meshes; predicted and GT
  separate_gt.py              Generating list of the GT meshes for performance comparison

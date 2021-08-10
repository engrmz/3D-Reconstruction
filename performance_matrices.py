

import collections
from os import path
import os
import numpy as np
import scipy as sp
from skimage import measure
import tensorflow.compat.v1 as tf
import trimesh
import matplotlib.pyplot as plt
# import open3d as o3d
import rtree
import yaml
import glob
import json
import time
from tqdm import tqdm

# count value val in tensor t
def tf_count(t, val):
  elements_equal_to_value = tf.equal(t, val)
  as_ints = tf.cast(elements_equal_to_value, tf.int32)
  count = tf.reduce_sum(as_ints)
  return count


def plot_samples_from_mesh(mesh, surface_samples):
  a = mesh.sample(surface_samples)
  fig = plt.figure(2)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(a[:, 2], a[:, 0], a[:, 1], c='r', marker='o')
  plt.show()


def compute_iou(points, gt_points):
  x = tf.constant([[1], [0], [1], [1], [0], [0]])
  y = tf.constant([[1], [0], [1], [0], [0], [1]])

  # if output>=0.5 make it 1 else 0

  i = tf.reduce_sum(x * y)
  m = tf.maximum(x, y)
  u = tf.reduce_sum(m)
  iou_ = i / u

  with tf.Session() as s:
    print("x: ", x.eval())
    print("y: ", y.eval())
    print("Intersection: ", i.eval())
    print("max: ", m.eval())
    print("Union: ", u.eval())
    print("IoU: ", iou_.eval())


# For every point in source, find its nearest - neighbor in target
# Find distance between nearest-neighbor points
def distance_field_helper(source, target):
  target_kdtree = sp.spatial.cKDTree(target)
  distances, unused_var = target_kdtree.query(source, n_jobs=-1)
  return distances


def compute_surface_metrics(mesh, mesh_gt, tau, eval_points):
  """Compute surface metrics (chamfer distance and f-score) for one example.

  Args:
    mesh: Predicted Mesh - trimesh.Trimesh, the mesh to evaluate.
    mesh_gt: Ground Truth Mesh - trimesh.Trimesh

  Returns:
    chamfer: float, chamfer distance.
    fscore: float, f-score.
  """

  # Chamfer
  # eval_points = 100000
  point_gt = mesh_gt.sample(eval_points) # points from surface of mesh
  point_gt = point_gt.astype(np.float32)
  point_pred = mesh.sample(eval_points)
  point_pred = point_pred.astype(np.float32)

  ## Distance from nearest-neighbor
  pred_to_gt = distance_field_helper(point_pred, point_gt)
  gt_to_pred = distance_field_helper(point_gt, point_pred)

  ## (a[0]**2 + a[1]**2 + ... + a[N]**2) / N
  chamfer = np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2)

  # print("chamfer : ", chamfer)

  # Fscore
  # tau = 1e-4
  eps = 1e-9

  pred_to_gt = (pred_to_gt**2)
  gt_to_pred = (gt_to_pred**2)

  ## (pred_to_gt <= tau) = [0,1,1,1,0,0,1] => (0+1+1+1+0+0+1)/7 => 0.751 * 100 = 75.1%
  prec_tau = (pred_to_gt <= tau).astype(np.float32).mean() * 100.
  recall_tau = (gt_to_pred <= tau).astype(np.float32).mean() * 100.

  fscore = (2 * prec_tau * recall_tau) / max(prec_tau + recall_tau, eps)

  # print("Tau : ", tau)
  # print("fscore : ", fscore)
  # Following the tradition to scale chamfer distance up by 10.
  return chamfer * 100., fscore


def main():

  # cloud = o3d.io.read_point_cloud("airplane.ply")  # Read the point cloud
  # o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud

  # with tf.io.gfile.GFile("dataset/input_files/Copper key.ply", "rb",) as fin:
  #   mesh_gt = trimesh.Trimesh(**trimesh.exchange.ply.load_ply(fin))
  #
  with tf.io.gfile.GFile("dataset/meshes_for_test/4_cvxnet.obj", "r",) as fin:
    cvxnet = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
  with tf.io.gfile.GFile("dataset/meshes_for_test/4_mrcnn.obj", "r",) as fin:
    meshrcnn = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
  with tf.io.gfile.GFile("dataset/meshes_for_test/4_mcvxnet.obj", "r",) as fin:
    mcvxnet = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
  with tf.io.gfile.GFile("dataset/meshes_for_test/4_gt.obj", "r",) as fin:
    gt_mesh = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))

  chamfer_cvxnet, fscore_cvxnet = compute_surface_metrics(cvxnet, gt_mesh, 0.3, 1000000)
  chamfer_meshrcnn, fscore_meshrcnn = compute_surface_metrics(meshrcnn, gt_mesh, 0.3, 1000000)
  chamfer_mcvxnet, fscore_mcvxnet = compute_surface_metrics(mcvxnet, gt_mesh, 0.3, 1000000)

  print("\nChamfer-L1 distance is : {} \& {} \& {}".format(chamfer_cvxnet, chamfer_meshrcnn, chamfer_mcvxnet))
  print("\nF-Score distance is : {} \& {} \& {}".format(fscore_cvxnet, fscore_meshrcnn, fscore_mcvxnet))

  ### Visualization of Meshes
  # mesh.show()
  # bolt1.show()

  # try:
  #   with tf.device('/device:GPU:0'):
  #     chamfer, fscore = compute_surface_metrics(mesh, gt_mesh, 1, 1000000)
  #     print("\nchamfer distance is : {}".format(chamfer))
  #     print("\nFscore is : {}".format(fscore))
  # except RuntimeError as e:
  #   print(e)


#===================== our mesh normalization ===============
def max_vertic(mesh):
  xmin = min(mesh.vertices[:, 0])
  xmax = max(mesh.vertices[:, 0])
  ymin = min(mesh.vertices[:, 1])
  ymax = max(mesh.vertices[:, 1])
  zmin = min(mesh.vertices[:, 2])
  zmax = max(mesh.vertices[:, 2])
  return max(xmax-xmin,ymax-ymin,zmax-zmin), (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2

def get_normalized_mesh(mesh):
  out_mesh = mesh.copy()

  """Step 1: Get min and max from very vertecs"""
  max_v, avg_x, avg_y, avg_z = max_vertic(out_mesh)
  m_v = out_mesh.vertices

  # Step 2: Normalization
  m_v[:, 0] = m_v[:, 0] - avg_x #np.mean(m_v[:, 0])
  m_v[:, 1] = m_v[:, 1] - avg_y #np.mean(m_v[:, 1])
  m_v[:, 2] = m_v[:, 2] - avg_z #np.mean(m_v[:, 2])

  # Step 2:
  m_v /= max_v
  out_mesh.vertices = m_v

  return out_mesh

# For testing and debugging
def debug_test_meshes():

  exp_no = 2 # 1 to 7

  with tf.io.gfile.GFile("dataset/icip_table1/3_{}_cvxnet_masked.obj".format(exp_no), "r",) as fin:
    mesh = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))

  with tf.io.gfile.GFile("dataset/icip_table1/3_{}_gt.obj".format(exp_no), "r",) as fin:
    gt = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))

  gt = get_normalized_mesh(gt)
  mesh = get_normalized_mesh(mesh)

  with tf.io.gfile.GFile("dataset/icip_table1/3_{}_cvxnet_masked_normalized_mesh.obj".format(exp_no), "w") as fout:
    mesh.export(fout, file_type="obj")


  # t1 = [[0.99, 0.05, 0.12, 0], [-0.01, 0.95, -0.32, 0.01], [-0.1, 0.3, 0.94, -0.05], [0, 0, 0, 1]]
  # t2 = [[0.77,-0.23,-0.6,0.01],[-0.04,0.91,-0.4,-0.02],[0.64,0.3,0.69,0.02],[0,0,0,1]]
  # t3 = [[1,-0.02,-0.03,-0.01],[0.01,0.96,-0.29,-0.01],[0.03,0.29,0.96,-0.02],[0,0,0,1]]
  # t4 = [[1,0.04,0.03,-0],[-0.01,0.76,-0.65,0.10],[-0.05,0.65,0.76,-0.09],[0,0,0,1]]
  # t5 = [[0.99,-0.05,-0.15,0],[-0.02,0.91,-0.42,0],[0.16,0.42,0.89,0],[0,0,0,1]]
  # t6 = [[-0.42,0.43,0.80,-0.18],[-0.06,0.87,-0.49,0.14],[-0.91,-0.25,-0.34,0.01],[0,0,0,1]]
  # t7 = [[-0.25,0.56,0.79,-0.25],[-0.12,0.79,-0.60,0.16],[-0.96,-0.25,-0.13,0.03],[0,0,0,1]]

  t1 = [[0.99, 0.05, 0.12, 0], [-0.01, 0.95, -0.32, 0], [-0.1, 0.3, 0.94, 0], [0, 0, 0, 1]]
  t2 = [[0.77, -0.23, -0.6, 0], [-0.04, 0.91, -0.4, 0], [0.64, 0.3, 0.69, 0], [0, 0, 0, 1]]
  t3 = [[1, -0.02, -0.03, 0], [0.01, 0.96, -0.29, 0], [0.03, 0.29, 0.96, 0], [0, 0, 0, 1]]
  t4 = [[1, 0.04, 0.03, 0], [-0.01, 0.76, -0.65, 0], [-0.05, 0.65, 0.76, 0], [0, 0, 0, 1]]
  t5 = [[0.99, -0.05, -0.15, 0], [-0.02, 0.91, -0.42, 0], [0.16, 0.42, 0.89, 0], [0, 0, 0, 1]]
  t6 = [[-0.42, 0.43, 0.80, 0], [-0.06, 0.87, -0.49, 00], [-0.91, -0.25, -0.34, 0], [0, 0, 0, 1]]
  t7 = [[-0.25, 0.56, 0.79, 0], [-0.12, 0.79, -0.60, 0], [-0.96, -0.25, -0.13, 0], [0, 0, 0, 1]]

  t = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
  t = np.linalg.inv(t)
  gt.apply_transform(t)
  with tf.io.gfile.GFile("dataset/icip_table1/3_{}_cvxnet_masked_normalized_GT.obj".format(exp_no), "w") as fout:
    gt.export(fout, file_type="obj")
  # mesh.apply_transform(t)
  # with tf.io.gfile.GFile("dataset/icip_table1/3_1_meshrcnn_normalized_2.obj", "w") as fout:
  #   mesh.export(fout, file_type="obj")

  chamfer_cvxnet, fscore_cvxnet = compute_surface_metrics(mesh, gt, 0.01, 100000)
  print("\nChamfer-L1 distance is : {},   F1-Score {}".format(chamfer_cvxnet, fscore_cvxnet))

  '''this transformation are valid for CvxNet => Transform GT'''
  # t = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
  # t = np.linalg.inv(t)
  # gt.apply_transform(t)
  # chamfer_cvxnet, fscore_cvxnet = compute_surface_metrics(mesh, gt, 0.001, 1000)
  # print("\nChamfer-L1 distance is : {},   F1-Score {}".format(chamfer_cvxnet, fscore_cvxnet))
  # with tf.io.gfile.GFile(path.join("test_mesh_out.obj"), "w") as fout:
  #   mesh.export(fout, file_type="obj")


  # # found t from MeshLab
  # trans = trimesh.registration.mesh_other(gt_copy,mesh)
  # gt_copy.apply_transform(trans[0])
  #
  #
  # chamfer_cvxnet, fscore_cvxnet = compute_surface_metrics(mesh, gt_copy, 0.001, 1000)
  # print("\nChamfer-L1 distance is : {},   F1-Score {}".format(chamfer_cvxnet, fscore_cvxnet))




#### =============== ICIP 2021 ====================

def icip_meshrcnn():
  t1 = [[0.99, 0.05, 0.12, 0], [-0.01, 0.95, -0.32, 0.01], [-0.1, 0.3, 0.94, -0.05], [0, 0, 0, 1]]
  t2 = [[0.77, -0.23, -0.6, 0.01], [-0.04, 0.91, -0.4, -0.02], [0.64, 0.3, 0.69, 0.02], [0, 0, 0, 1]]
  t3 = [[1, -0.02, -0.03, -0.01], [0.01, 0.96, -0.29, -0.01], [0.03, 0.29, 0.96, -0.02], [0, 0, 0, 1]]
  t4 = [[1, 0.04, 0.03, -0], [-0.01, 0.76, -0.65, 0.10], [-0.05, 0.65, 0.76, -0.09], [0, 0, 0, 1]]
  t5 = [[0.99, -0.05, -0.15, 0], [-0.02, 0.91, -0.42, 0], [0.16, 0.42, 0.89, 0], [0, 0, 0, 1]]
  t6 = [[-0.42, 0.43, 0.80, -0.18], [-0.06, 0.87, -0.49, 0.14], [-0.91, -0.25, -0.34, 0.01], [0, 0, 0, 1]]
  t7 = [[-0.25, 0.56, 0.79, -0.25], [-0.12, 0.79, -0.60, 0.16], [-0.96, -0.25, -0.13, 0.03], [0, 0, 0, 1]]
  t = [t1, t1, t2, t3, t4, t5, t6, t7]

  # # Transormation matrix for mesh-rcnn only
  # t1 = [[0.99, 0.05, 0.12, 0], [-0.01, 0.95, -0.32, 0.01], [-0.1, 0.3, 0.94, -0.05], [0, 0, 0, 1]]
  # t2 = [[0.82,-0.23,-0.53,0.01],[-0.02,0.91,-0.42,-0.02],[0.58,0.35,0.74,0.02],[0,0,0,1]]
  # # t2 = [[0.77, -0.23, -0.6, 0.01], [-0.04, 0.91, -0.4, -0.02], [0.64, 0.3, 0.69, 0.02], [0, 0, 0, 1]]
  # t3 = [[1, -0.02, -0.03, -0.01], [0.01, 0.96, -0.29, -0.01], [0.03, 0.29, 0.96, -0.02], [0, 0, 0, 1]]
  # t4 = [[1, 0.04, 0.03, -0], [-0.01, 0.76, -0.65, 0.10], [-0.05, 0.65, 0.76, -0.09], [0, 0, 0, 1]]
  # t5 = [[0.99, -0.05, -0.15, 0], [-0.02, 0.91, -0.42, 0], [0.16, 0.42, 0.89, 0], [0, 0, 0, 1]]
  # t6 = [[-0.42, 0.43, 0.80, -0.18], [-0.06, 0.87, -0.49, 0.14], [-0.91, -0.25, -0.34, 0.01], [0, 0, 0, 1]]
  # t7 = [[-0.25, 0.56, 0.79, -0.25], [-0.12, 0.79, -0.60, 0.16], [-0.96, -0.25, -0.13, 0.03], [0, 0, 0, 1]]
  #

  # t1 = [[0.99, 0.05, 0.12, 0], [-0.01, 0.95, -0.32, 0], [-0.1, 0.3, 0.94, 0], [0, 0, 0, 1]]
  # t2 = [[0.77, -0.23, -0.6, 0], [-0.04, 0.91, -0.4, 0], [0.64, 0.3, 0.69, 0], [0, 0, 0, 1]]
  # t3 = [[1, -0.02, -0.03, 0], [0.01, 0.96, -0.29, 0], [0.03, 0.29, 0.96, 0], [0, 0, 0, 1]]
  # t4 = [[1, 0.04, 0.03, 0], [-0.01, 0.76, -0.65, 0], [-0.05, 0.65, 0.76, 0], [0, 0, 0, 1]]
  # t5 = [[0.99, -0.05, -0.15, 0], [-0.02, 0.91, -0.42, 0], [0.16, 0.42, 0.89, 0], [0, 0, 0, 1]]
  # t6 = [[-0.42, 0.43, 0.80, 0], [-0.06, 0.87, -0.49, 00], [-0.91, -0.25, -0.34, 0], [0, 0, 0, 1]]
  # t7 = [[-0.25, 0.56, 0.79, 0], [-0.12, 0.79, -0.60, 0], [-0.96, -0.25, -0.13, 0], [0, 0, 0, 1]]
  #
  # t_meshrcnn = [t1, t1, t2, t3, t4, t5, t6, t7]

  for exp_no in range (1,8):  # 1 to 7

    with tf.io.gfile.GFile("dataset/icip_table1/3_{}_meshrcnn.obj".format(exp_no), "r",) as fin:
      mesh = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))

    with tf.io.gfile.GFile("dataset/icip_table1/3_{}_gt.obj".format(exp_no), "r",) as fin:
      gt = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))

    """Step 1: Get min and max from very vertecs"""
    max_v = max_vertic(mesh)
    m_v = mesh.vertices

    # Step 2: Normalization
    m_v[:, 0] = m_v[:, 0] - np.mean(m_v[:, 0])
    m_v[:, 1] = m_v[:, 1] - np.mean(m_v[:, 1])
    m_v[:, 2] = m_v[:, 2] - np.mean(m_v[:, 2])

    # Step 2:
    m_v /= max_v
    mesh.vertices = m_v
    mesh.apply_transform(t[exp_no])

    with tf.io.gfile.GFile("dataset/icip_table1/3_{}_meshrcnn_normalized_aligned.obj".format(exp_no), "w") as fout:
      mesh.export(fout, file_type="obj")

    chamfer_cvxnet, fscore_cvxnet = compute_surface_metrics(mesh, gt, 0.01, 100000)
    print("\nChamfer-L1 distance is : {},   F1-Score {}".format(chamfer_cvxnet, fscore_cvxnet))

def save_meshes(label, output_folder, exp_no, cvxnet_r, cvxnet_m, cvxnet_padded, occnet_r, occnet_m, occnet_padded, meshrcnn, gt):
  with tf.io.gfile.GFile("{}/3_{}_cvxnet_r_{}.obj".format(output_folder, exp_no,label), "w") as fout:
    cvxnet_r.export(fout, file_type="obj")
  with tf.io.gfile.GFile("{}/3_{}_cvxnet_m_{}.obj".format(output_folder, exp_no,label), "w") as fout:
    cvxnet_m.export(fout, file_type="obj")
  with tf.io.gfile.GFile("{}/3_{}_cvxnet_padded_{}.obj".format(output_folder, exp_no,label), "w") as fout:
    cvxnet_padded.export(fout, file_type="obj")
  with tf.io.gfile.GFile("{}/3_{}_occnet_r_{}.obj".format(output_folder, exp_no,label), "w") as fout:
    occnet_r.export(fout, file_type="obj")
  with tf.io.gfile.GFile("{}/3_{}_occnet_m_{}.obj".format(output_folder, exp_no,label), "w") as fout:
    occnet_m.export(fout, file_type="obj")
  with tf.io.gfile.GFile("{}/3_{}_occnet_padded_{}.obj".format(output_folder, exp_no,label), "w") as fout:
    occnet_padded.export(fout, file_type="obj")
  with tf.io.gfile.GFile("{}/3_{}_meshrcnn_{}.obj".format(output_folder, exp_no,label), "w") as fout:
    meshrcnn.export(fout, file_type="obj")
  with tf.io.gfile.GFile("{}/3_{}_gt_{}.obj".format(output_folder, exp_no,label), "w") as fout:
    gt.export(fout, file_type="obj")


def load_meshes(exp_no):


  t_occ = [[1,0,0,0.03],[0,1,0, 0.07],[0,0,1,-0.01],[0,0,0,1]]
  t_cvx = [[1, 0, 0, 0.01], [0, 1, 0, 0.04], [0, 0, 1, -0.03], [0, 0, 0, 1]]

  with tf.io.gfile.GFile("dataset/icip_table1/3_{}_gt.obj".format(exp_no), "r",) as fin:
    gt = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
  with tf.io.gfile.GFile("dataset/icip_table1/3_{}_cvxnet_real.obj".format(exp_no), "r",) as fin:
    cvxnet_r = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
    # cvxnet_r.apply_transform(t_cvx)
  with tf.io.gfile.GFile("dataset/icip_table1/3_{}_cvxnet_masked.obj".format(exp_no), "r",) as fin:
    cvxnet_m = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
    # cvxnet_m.apply_transform(t_cvx)
  with tf.io.gfile.GFile("dataset/icip_table1/3_{}_occnet_real.obj".format(exp_no), "r",) as fin:
    occnet_r = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
    # occnet_r.apply_transform(t_occ)
  with tf.io.gfile.GFile("dataset/icip_table1/3_{}_occnet_masked.obj".format(exp_no), "r",) as fin:
    occnet_m = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
    # occnet_m.apply_transform(t_occ)
  with tf.io.gfile.GFile("dataset/icip_table1/3_{}_meshrcnn.obj".format(exp_no), "r",) as fin:
    meshrcnn = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))



  return gt, cvxnet_r, cvxnet_m, occnet_r, occnet_m, meshrcnn

  #### =============== Padded meshes====================
def load_padded_meshes(exp_no):
  with tf.io.gfile.GFile("dataset/icip_table1/input_padded_shapes/exp5_{}_padding.obj".format(exp_no), "r", ) as fin:
    cvxnet_padded = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))

  with tf.io.gfile.GFile("dataset/icip_table1/input_padded_shapes/exp5_{}_padding_occnet.obj".format(exp_no), "r", ) as fin:
    occnet_padded = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))

  return cvxnet_padded, occnet_padded

def icip_table1():
  tau = 0.001  # 1e-4
  sample_points = 100000  # 100k
  exp_no = 1  # change exp 1 to 7 => Samples in table 1
  print("Tau: {}, Sample_points: {} ".format(tau,sample_points))
  shapeNet_transformation = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]

  output_folder = 'dataset/icip_table1/output_ICP'
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  while exp_no < 8:
    gt, cvxnet_r, cvxnet_m, occnet_r, occnet_m, meshrcnn = load_meshes(exp_no)
    cvxnet_padded, occnet_padded = load_padded_meshes(exp_no)

    # Normalize and Transform w.r.t GT
    cvxnet_r = get_normalized_mesh(cvxnet_r).apply_transform(shapeNet_transformation)
    cvxnet_m = get_normalized_mesh(cvxnet_m).apply_transform(shapeNet_transformation)
    cvxnet_padded = get_normalized_mesh(cvxnet_padded).apply_transform(shapeNet_transformation)
    occnet_r = get_normalized_mesh(occnet_r).apply_transform(shapeNet_transformation)
    occnet_m = get_normalized_mesh(occnet_m).apply_transform(shapeNet_transformation)
    occnet_padded = get_normalized_mesh(occnet_padded).apply_transform(shapeNet_transformation)
    meshrcnn = get_normalized_mesh(meshrcnn)  # already in pose same as GT
    gt = get_normalized_mesh(gt)

    # Debugging :: Save Meshes before ICP
    save_meshes("Before", output_folder, exp_no,
                cvxnet_r, cvxnet_m, cvxnet_padded, occnet_r, occnet_m, occnet_padded,meshrcnn, gt)



    # Sample points from each mesh for registration (ICP)
    cvxnet_r_samples = cvxnet_r.sample(1000)
    cvxnet_m_samples = cvxnet_m.sample(1000)
    cvxnet_padded_samples = cvxnet_padded.sample(1000)
    occnet_r_samples = occnet_r.sample(1000)
    occnet_m_samples = occnet_m.sample(1000)
    occnet_padded_samples = occnet_padded.sample(1000)
    meshrcnn_samples = meshrcnn.sample(1000)



    # Compute ICP transformations
    initial_t = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    cvxnet_r_trans = trimesh.registration.icp(cvxnet_r_samples, gt, initial=initial_t, threshold=1e-05,
                                              max_iterations=100, scale=False)
    cvxnet_m_trans = trimesh.registration.icp(cvxnet_m_samples, gt, initial=initial_t, threshold=1e-05,
                                              max_iterations=100, scale=False)
    cvxnet_padded_trans = trimesh.registration.icp(cvxnet_padded_samples, gt, initial=initial_t, threshold=1e-05,
                                              max_iterations=100, scale=False)
    occnet_r_trans = trimesh.registration.icp(occnet_r_samples, gt, initial=initial_t, threshold=1e-05,
                                              max_iterations=100, scale=False)
    occnet_m_trans = trimesh.registration.icp(occnet_m_samples, gt, initial=initial_t, threshold=1e-05,
                                              max_iterations=100, scale=False)
    occnet_padded_trans = trimesh.registration.icp(occnet_padded_samples, gt, initial=initial_t, threshold=1e-05,
                                              max_iterations=100, scale=False)
    meshrcnn_trans = trimesh.registration.icp(meshrcnn_samples, gt, initial=initial_t, threshold=1e-05,
                                              max_iterations=100, scale=False)


    # Apply transformations
    cvxnet_r.apply_transform(cvxnet_r_trans[0])
    cvxnet_m.apply_transform(cvxnet_m_trans[0])
    cvxnet_padded.apply_transform(cvxnet_padded_trans[0])
    occnet_r.apply_transform(occnet_r_trans[0])
    occnet_m.apply_transform(occnet_m_trans[0])
    occnet_padded.apply_transform(occnet_padded_trans[0])
    meshrcnn.apply_transform(meshrcnn_trans[0])

    # Debugging :: Save Meshes before ICP
    save_meshes("After", output_folder, exp_no,
                cvxnet_r, cvxnet_m, cvxnet_padded, occnet_r, occnet_m, occnet_padded,meshrcnn, gt)

    # Compute Chamfer Distance and F1-Score
    chamfer_cvxnet_r, fscore_cvxnet_r = compute_surface_metrics(cvxnet_r, gt, tau, sample_points)
    chamfer_cvxnet_m, fscore_cvxnet_m = compute_surface_metrics(cvxnet_m, gt, tau, sample_points)
    chamfer_occnet_r, fscore_occnet_r = compute_surface_metrics(occnet_r, gt, tau, sample_points)
    chamfer_occnet_m, fscore_occnet_m = compute_surface_metrics(occnet_m, gt, tau, sample_points)
    chamfer_cvxnet_padded, fscore_cvxnet_padded = compute_surface_metrics(cvxnet_padded, gt, tau, sample_points)
    chamfer_occnet_padded, fscore_occnet_padded = compute_surface_metrics(occnet_padded, gt, tau, sample_points)
    chamfer_meshrcnn, fscore_meshrcnn = compute_surface_metrics(meshrcnn, gt, tau, sample_points)

    print('Sample {} & {:.2f} & {:.2f} & {:.2f} &  {:.2f} & {:.2f} &  {:.2f}  &  {:.2f}  &  {:.2f}  &  {:.2f} & {:.2f} & {:.2f} & {:.2f} &'
          '  {:.2f}  &  {:.2f}  \\\\'.format(exp_no,chamfer_cvxnet_r,chamfer_cvxnet_m, chamfer_cvxnet_padded, chamfer_occnet_r, chamfer_occnet_m, chamfer_occnet_padded, chamfer_meshrcnn,
                                                    fscore_cvxnet_r,fscore_cvxnet_m,   fscore_cvxnet_padded,  fscore_occnet_r,  fscore_occnet_m, fscore_occnet_padded, fscore_meshrcnn))

    exp_no +=1


def registration_icp_debug():
  tau = 0.01  # 1e-4
  sample_points = 100000  # 100k
  exp_no = 2  # change exp 1 to 7 => Samples in table 1
  print("Tau: {}, Sample_points: {} ".format(tau,sample_points))


  while exp_no < 3:

    with tf.io.gfile.GFile("dataset/icip_table1/3_{}_gt.obj".format(exp_no), "r", ) as fin:
      gt = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
    with tf.io.gfile.GFile("dataset/icip_table1/3_{}_cvxnet_masked.obj".format(exp_no), "r", ) as fin:
      cvxnet_m = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
    with tf.io.gfile.GFile("dataset/icip_table1/3_{}_meshrcnn.obj".format(exp_no), "r", ) as fin:
      meshrcnn = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))

    cvxnet_m = get_normalized_mesh(cvxnet_m)
    meshrcnn = get_normalized_mesh(meshrcnn)
    gt = get_normalized_mesh(gt)

    output_folder = 'dataset/icip_table1/output_ICP'
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)

    t = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
    cvxnet_m.apply_transform(t)
    # t = np.linalg.inv(t)
    # gt.apply_transform(t)

    with tf.io.gfile.GFile("{}/3_{}_cvxnet_m_before.obj".format(output_folder, exp_no), "w") as fout:
      cvxnet_m.export(fout, file_type="obj")
    with tf.io.gfile.GFile("{}/3_{}_gt_before.obj".format(output_folder, exp_no), "w") as fout:
      gt.export(fout, file_type="obj")
    with tf.io.gfile.GFile("{}/3_{}_meshrcnn_before.obj".format(output_folder, exp_no), "w") as fout:
      meshrcnn.export(fout, file_type="obj")


    out = []
    out.append(gt)
    out.append(cvxnet_m)
    for i in range (3):

      a = cvxnet_m.sample(10000)
      b = gt.sample(10000)
      m = meshrcnn.sample(10000)

      # initial_transforms = trimesh.registration.procrustes(a, b, reflection=False, translation=False, scale=False,
      #                                                      return_cost=False)
      initial_t = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      cvxnet_trans=trimesh.registration.icp(a, gt, initial=initial_t, threshold=1e-05, max_iterations=100)

      initial_transforms = trimesh.registration.procrustes(m, b, reflection=False, translation=False, scale=False,
                                                           return_cost=False)
      mesh_trans=trimesh.registration.icp(m, gt, initial=initial_t, threshold=1e-05, max_iterations=100)


      # meshrcnn_trans = trimesh.registration.icp(m, gt, initial=initial_transforms, threshold=1e-05, max_iterations=100)
      # print(mesh_trans[0])
      cvxnet_m.apply_transform(cvxnet_trans[0])
      meshrcnn.apply_transform(mesh_trans[0])

      #
      # t = trimesh.registration.mesh_other(meshrcnn, gt, samples=1000, scale=False, icp_first=50, icp_final=250)
      # meshrcnn.apply_transform(t[0])
      # t = trimesh.registration.mesh_other(cvxnet_m, gt, samples=1000, scale=False, icp_first=50, icp_final=250)
      # cvxnet_m.apply_transform(t[0])

      out.append(cvxnet_m)
      with tf.io.gfile.GFile("{}/3_{}_cvxnet_m_after.obj".format(output_folder, i), "w") as fout:
        cvxnet_m.export(fout, file_type="obj")
      with tf.io.gfile.GFile("{}/3_{}_meshrcnn_after.obj".format(output_folder, i), "w") as fout:
          meshrcnn.export(fout, file_type="obj")

    # out_samples = []
    # for i in range (5):
    #   out_samples.append(out[i].sample(1000))
    #
    # fig = plt.figure(1)
    # plt.axis('off')
    # ax = fig.add_subplot(151, projection='3d')
    # ax.scatter(out_samples[0][:, 2], out_samples[0][:, 0], out_samples[0][:, 1], c='r', marker='o')
    # ax = fig.add_subplot(152, projection='3d')
    # ax.scatter(out_samples[1][:, 2], out_samples[1][:, 0], out_samples[1][:, 1], c='b', marker='o')
    # ax = fig.add_subplot(153, projection='3d')
    # ax.scatter(out_samples[2][:, 2], out_samples[2][:, 0], out_samples[2][:, 1], c='b', marker='o')
    # ax = fig.add_subplot(154, projection='3d')
    # ax.scatter(out_samples[3][:, 2], out_samples[3][:, 0], out_samples[3][:, 1], c='b', marker='o')
    # ax = fig.add_subplot(155, projection='3d')
    # ax.scatter(out_samples[4][:, 2], out_samples[4][:, 0], out_samples[4][:, 1], c='b', marker='o')
    # ax.scatter(out_samples[0][:, 2], out_samples[0][:, 0], out_samples[0][:, 1], c='r', marker='o')
    #
    # plt.show()

    # with tf.io.gfile.GFile("{}/3_{}_cvxnet_m.obj".format(output_folder, exp_no), "w") as fout:
    #   cvxnet_m.export(fout, file_type="obj")


    exp_no +=1



####    Table 2 ======================

folder_name = 'table_predictions_07'

class DataLoader:
  def __init__(self):
    # self.dataset_dir = '/data/mzohaib/code/datasets/3dv_results/cvxnet/ShapeNet_yolact_real'
  ##  self.dataset_dir = '/media/mz/mz/ICIP2021_ServerResults/Dataset/icip_comparison/out_cvxnet_real/common_yolact_real/ShapeNet_yolact_real'
    self.dataset_dir = '/media/mz/mz/3DV_Results/Output_Meshes/ONet_SS/onet_ss_pix3d/ShapeNet'

    # self.dataset_dir = '/data/mzohaib/code/dataset/icip_comparison/out/common_yolact_masked/ShapeNet'
    # self.dataset_dir = '/data/mzohaib/code/dataset/icip_comparison/pix3d_3_categories_meshrcnn/ShapeNet'
    # self.dataset_dir = '/data/mzohaib/code/dataset/comparison_table2/occnet_ShapeNet_real'
    # self.gt_dir = '/data/mzohaib/code/dataset/comparison_table2/GT_occnet_ShapeNet_masked/'
    self.gt_lst =  json.load(open(glob.glob("{}/ShapeNet_GT_list.json".format(self.dataset_dir))[0]))
    directory = '/{}/{}/{}'.format(os.path.join(*self.dataset_dir.split('/')[0:-1]), self.dataset_dir.split('/')[-1], folder_name)
    if not os.path.exists(directory):
      os.makedirs(directory)
    self.history = open('/{}/{}/{}/_history.txt'.format(os.path.join(*self.dataset_dir.split('/')[0:-1]), self.dataset_dir.split('/')[-1], folder_name), 'w')
    self.results = open(
      '/{}/{}/{}/_chair_results.txt'.format(os.path.join(*self.dataset_dir.split('/')[0:-1]), self.dataset_dir.split('/')[-1], folder_name), 'w')
    self.file_overall = open(
      '/{}/{}/{}/_chair_overall.txt'.format(os.path.join(*self.dataset_dir.split('/')[0:-1]), self.dataset_dir.split('/')[-1], folder_name), 'w')
    self.file_table = open(
      '/{}/{}/{}/_chair_table.txt'.format(os.path.join(*self.dataset_dir.split('/')[0:-1]), self.dataset_dir.split('/')[-1], folder_name), 'w')
    self.file_chair = open(
      '/{}/{}/{}/_chair_chair.txt'.format(os.path.join(*self.dataset_dir.split('/')[0:-1]), self.dataset_dir.split('/')[-1], folder_name), 'w')
    self.file_sofa = open(
      '/{}/{}/{}/_chair_sofa.txt'.format(os.path.join(*self.dataset_dir.split('/')[0:-1]), self.dataset_dir.split('/')[-1], folder_name), 'w')
    self.invalid_mesh_list = open('/{}/{}_chair_invalid_mesh_list.txt'.format(os.path.join(*self.dataset_dir.split('/')[0:-1]), self.dataset_dir.split('/')[-1]),
      'w')
    # self.out_put = "{}{}".format(self.gt_dir, self.dataset_dir.split('/')[-1])
    # self.metadata_file = 'metadata_selected.yaml'
    # self.image_folder = 'img_choy2016'



  def load_predictions(self):
    pred_sofa = []
    pred_table = []
    pred_chair = []
    #temp_sofa = sorted(glob.glob('{}/04256520/sofa_background/{}/*.obj'.format(self.dataset_dir, folder_name)))
    temp_table = sorted(glob.glob('{}/04379243/table_background/{}/*.obj'.format(self.dataset_dir, folder_name)))
    # temp_chair = sorted(glob.glob('{}/03001627/chair_background/{}/*.obj'.format(self.dataset_dir, folder_name)))

#    for x in temp_sofa:
 #      key = x.split('/')[-4] + '-' + x.split('/')[-1].split('_')[0]
 #      if key in self.gt_lst.keys():
 #        pred_sofa.append(x)
 #      else:
 #        self.invalid_mesh_list.write("SOFA -- Mesh is not in YOLACT: {}".format(x))

    for x in temp_table:
       key = x.split('/')[-4] + '-' + x.split('/')[-1].split('_')[0]
       if key in self.gt_lst.keys():
         pred_table.append(x)
       else:
         self.invalid_mesh_list.write("TABLE -- Mesh is not in YOLACT: {}".format(x))

#    for x in temp_chair:
#      key = x.split('/')[-4] + '-' + x.split('/')[-1].split('_')[0]
#      if key in self.gt_lst.keys():
#        pred_chair.append(x)
#      else:
#        self.invalid_mesh_list.write("CHAIR -- Mesh is not in YOLACT: {}\n".format(x))

    print("Total number of loaded shapes:     Chair = {},   Sofa = {},  Table = {}\n".format(len(pred_chair), len(pred_sofa), len(pred_table)))
    self.results.write("\nTotal number of loaded shapes:     Chair = {},   Sofa = {},  Table = {}\n".format(len(pred_chair), len(pred_sofa), len(pred_table)))

    return [pred_sofa, pred_table, pred_chair]


  def load_pred_gt(self, prediction):
    # keys = [i for i in self.gt_lst.keys()]

    folders = prediction.split('/')
    gt_path = self.gt_lst[folders[-4]+'-'+folders[-1].split('_')[0]]

    # gt_path = "{}{}/{}_gt.obj".format(self.gt_dir, folders[7],folders[-1].split('_')[0])
    self.history.write(prediction + ' --> ' + gt_path + '\n')

    # with tf.io.gfile.GFile(prediction, "r", ) as fin:
    #   mesh = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
    # with tf.io.gfile.GFile(gt_path, "r", ) as fin:
    #   gt = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))

    mesh = trimesh.load(prediction, force='mesh')
    try:
      gt = trimesh.load(gt_path, force='mesh')
    except:
      gt = mesh
      print("GT not found:: {}", gt_path)

    return mesh, gt

  def update_results(self, cd_table, cd_sofa, cd_chair, f_table, f_sofa, f_chair, count_table, count_sofa, count_chair):
    if count_table != 0:
      cd_table /= count_table
      f_table /= count_table

    if count_chair != 0:
      cd_chair /= count_chair
      f_chair /= count_chair

    if count_sofa != 0:
      cd_sofa /= count_sofa
      f_sofa /= count_sofa


    cd_overall = (cd_table + cd_sofa + cd_chair)/3
    f_overall = (f_table + f_sofa + f_chair)/3
    print('cd_overall: {:.2f} , cd_table: {:.2f}, cd_sofa: {:.2f}, cd_chair {:.2f}'.
          format(cd_overall, cd_table, cd_sofa, cd_chair))
    print('f_overall: {:.2f} , f_table: {:.2f}, f_sofa: {:.2f}, f_chair {:.2f}'.
          format(f_overall, f_table, f_sofa, f_chair))

    self.results.write('cd_overall: {:.2f} , cd_table: {:.2f}, cd_sofa: {:.2f}, cd_chair {:.2f}    f_overall: {:.2f} , f_table: {:.2f}, f_sofa: {:.2f}, f_chair {:.2f}'.
          format(cd_overall, cd_table, cd_sofa, cd_chair, f_overall, f_table, f_sofa, f_chair) + '\n')

    self.results.write(
            'count_table: {:.2f} , count_chair: {:.2f}, count_sofa: {:.2f} \n'.
            format(count_table, count_chair, count_sofa))

    self.history.close()
    self.results.close()
    # self.detailed_results.close()
    self.file_overall.close()
    self.file_table.close()
    self.file_chair.close()
    self.file_sofa.close()
    self.invalid_mesh_list.close()

  def update_files(self, cd, fscore, mesh_name,  category, count):

    if category == '04256520':
      self.file_sofa.write('No:   {}     Mesh:  {}      CD:   {:.2f}   F1:    {:.2f}'.format(count, mesh_name, cd, fscore) + '\n')
    elif category == '03001627':
      self.file_chair.write('No:   {}   Mesh:  {}      CD:   {:.2f}   F1:    {:.2f}'.format(count, mesh_name, cd, fscore) + '\n')
    elif category == '04379243':
      self.file_table.write('No:   {}   Mesh:  {}      CD:   {:.2f}   F1:    {:.2f}'.format(count, mesh_name, cd, fscore) + '\n')
    self.file_overall.write('Mesh:  {}      CD:   {:.2f}   F1:    {:.2f}'.format( mesh_name, cd, fscore) + '\n')

  def update_invalid_mesh_list(self, prediction, mesh, gt):
    self.invalid_mesh_list.write("\nInvalide mesh: {}, mesh.vertices: {}, GT.vertices {}".format(prediction, len(mesh.vertices), len(gt.vertices)))


def icip_table2():
  tau = 0.001  # 1e-4
  sample_points = 100000  # 100k
  exp_no = 1  # change exp 1 to 7 => Samples in table 1
  print("Tau: {}, Sample_points: {} ".format(tau,sample_points))
  shapeNet_transformation = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]

  file1 = open("MyFile.txt", "a")
  count_table = 0; count_sofa = 0; count_chair =0
  cd_table = 0.0; cd_sofa=0.0; cd_chair=0.0
  f_table = 0.0;  f_sofa = 0.0; f_chair = 0.0
  cd_overall = 0.0; f_overall = 0.0
  dataset_loader = DataLoader()
  dataset = dataset_loader.load_predictions()
  initial_time = time.time()
  for category in dataset:  # 3 times  pred_sofa, pred_table, pred_chair
    for prediction in tqdm(category): # traverse all the predictions in this category
      start_time = time.time()
      # mesh, gt = dataset_loader.load_pred_gt(prediction)

      try:
        mesh, gt = dataset_loader.load_pred_gt(prediction)
      except:
        with tf.io.gfile.GFile(prediction, "r", ) as fin:
          mesh = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
        folders = prediction.split('/')
        gt_path = dataset_loader.gt_lst[folders[-4] + '-' + folders[-1].split('_')[0]]
        with tf.io.gfile.GFile(gt_path, "r", ) as fin:
          gt = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
        print("\nInvalide mesh: {}, mesh.vertices: {}, GT.vertices {}".format(prediction, len(mesh.vertices),
                                                                              len(gt.vertices)))
        dataset_loader.invalid_mesh_list.write(
          "\nInvalide mesh: {}, mesh.vertices: {}, GT.vertices {}".format(prediction, len(mesh.vertices),
                                                                          len(gt.vertices)))
        continue


      # Normalize and Transform w.r.t GT
      try:
        mesh = get_normalized_mesh(mesh).apply_transform(shapeNet_transformation)
        gt = get_normalized_mesh(gt)
      except:
        print('Error in mesh: ')
        continue
        
      # Sample points from each mesh for registration (ICP)
      mesh_samples = mesh.sample(1000)

      # Compute ICP transformations
      initial_t = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      try:
        mesh_trans = trimesh.registration.icp(mesh_samples, gt, initial=initial_t, threshold=1e-05,
                                                max_iterations=100, scale=False)
      except:
        continue
      # Apply transformations
      mesh.apply_transform(mesh_trans[0])

      # Compute Chamfer Distance and F1-Score
      chamfer_distance, fscore_score = compute_surface_metrics(mesh, gt, tau, sample_points)


      if prediction.split('/')[-4] == '04256520':
        cd_sofa += chamfer_distance
        f_sofa += fscore_score
        count_sofa += 1
        dataset_loader.update_files(chamfer_distance, fscore_score, prediction,'04256520',count_sofa)
      elif prediction.split('/')[-4] == '03001627':
        cd_chair += chamfer_distance
        f_chair += fscore_score
        count_chair += 1
        dataset_loader.update_files(chamfer_distance, fscore_score, prediction, '03001627', count_chair)
      elif prediction.split('/')[-4] == '04379243':
        cd_table += chamfer_distance
        f_table += fscore_score
        count_table +=1
        dataset_loader.update_files(chamfer_distance, fscore_score, prediction, '04379243', count_table)
      else:
        print("Unknown category... Pleas check:: {}\n\n\n ".format(prediction.split('/')[-4] ))

      # print("comparing mesh: {}         Time:  {:.2f} Sec.      Total time: {:.2f} Min."
      #       .format(prediction.split('/')[-3]+'-'+prediction.split('/')[-1], time.time() - start_time, (time.time() - initial_time) / 60))

      ## # save resutls for every iteration
      ## dataset_loader.update_detailed_results(cd_table, cd_sofa, cd_chair, f_table, f_sofa, f_chair)

  # save resutls in the end and close files
  dataset_loader.update_results(cd_table, cd_sofa, cd_chair, f_table, f_sofa, f_chair, count_table, count_sofa, count_chair)

def is_valid_prediction():
  dataset_loader = DataLoader()
  dataset = dataset_loader.load_predictions()
  flag = True
  for category in dataset:  # 3 times  pred_sofa, pred_table, pred_chair
    for prediction in category: # traverse all the predictions in this category
      print("comparing mesh: {}".format(prediction))
      try:
        mesh, gt = dataset_loader.load_pred_gt(prediction)
      except:
        with tf.io.gfile.GFile(prediction, "r", ) as fin:
          mesh = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
        folders = prediction.split('/')
        gt_path = dataset_loader.gt_lst[folders[-4] + '-' + folders[-1].split('_')[0]]
        with tf.io.gfile.GFile(gt_path, "r", ) as fin:
          gt = trimesh.Trimesh(**trimesh.exchange.obj.load_obj(fin))
        print("\nInvalide mesh: {}, mesh.vertices: {}, GT.vertices {}".format(prediction, len(mesh.vertices),
                                                                              len(gt.vertices)))
        dataset_loader.invalid_mesh_list.write(
          "\nInvalide mesh: {}, mesh.vertices: {}, GT.vertices {}".format(prediction, len(mesh.vertices),
                                                                          len(gt.vertices)))

        flag = False
        continue
      # check if any mesh has no vertices
      #
      # if len(mesh.vertices) == 0 or len(gt.vertices)==0:
      #   print("\nInvalide mesh: {}, mesh.vertices: {}, GT.vertices {}".format(prediction, len(mesh.vertices), len(gt.vertices)))
      #   dataset_loader.invalid_mesh_list.write(
      #     "\nInvalide mesh: {}, mesh.vertices: {}, GT.vertices {}".format(prediction, len(mesh.vertices),
      #                                                                     len(gt.vertices)))
      #   flag = False
      #   continue
  return flag

if __name__ == "__main__":
  # debug_test_meshes()
  # main()
  # icip_table1() # transform GT
  # icip_meshrcnn()
  # registration_icp_debug()  # Debug ICP for sample 2 only
  #
  # if is_valid_prediction(): #test the predicted meshes first
  #   print("\n\nYes, meshes are complete, you can run performance measurement task")
  # else:
  #   print("\n\nSome fo the meshes are incorrect, please find the list in *_invalid_mesh_list.txt in dataset_dir")

  icip_table2()

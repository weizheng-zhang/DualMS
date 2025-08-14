import argparse
import os
import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import mcubes
import pyvista as pv
import trimesh
from concurrent.futures import ProcessPoolExecutor

import models
from generate_rb_points import *

# wirte mesh
def write_ply_triangle(name, vertices, triangles):    
    fout = open(name, "w")
    for ii in range(len(vertices)):
        fout.write("v " + str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f " +str(int(triangles[ii,0]) + 1)+" "+str(int(triangles[ii,1]) + 1)+" "+str(int(triangles[ii,2]) + 1)+"\n")
    fout.close()
    print('file saved already!')

def save_result(model, val, count, args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    file_name = args.out + 'train' + str(count) + '.obj'
    # generate mesh
    resolution = args.resolution
    grid_size = 1/resolution
    d = 1 # cube size
    x = np.arange(0,d, grid_size)
    y = np.arange(0,d, grid_size)
    z = np.arange(0,d, grid_size)
    sample_x = np.array(np.meshgrid(x,y,z))
    sample_x = sample_x .reshape((3, resolution * resolution * resolution))
    sample_x = sample_x .transpose() # shape = (resolution * resolution * resolution, 3)
    sample_x = torch.from_numpy(sample_x).to(device).to(torch.float32)
    grid_fx = np.empty([sample_x.size()[0],1], dtype = np.float32) 
    print('start sample in network...')
    for i in range(resolution):
        with torch.no_grad():
            t = model(sample_x[resolution * resolution * i: resolution * resolution * (i + 1), :], only_f = True)
            t = t.detach().cpu().numpy()
        grid_fx[resolution * resolution * i: resolution * resolution * (i + 1), ...] = t     
    grid_fx = grid_fx.reshape((resolution, resolution, resolution))

    print('start marching cubes...')
    _vertices, _triangles = mcubes.marching_cubes(grid_fx, 0.0)
    _vertices[:, [0, 1]] = _vertices[:, [1, 0]]
    _vertices = _vertices/resolution #normalize to [0,1]
    write_ply_triangle(file_name, _vertices, _triangles) # save mesh

    faces = np.concatenate([np.full((_triangles.shape[0], 1), 3), _triangles], axis=1).astype(np.int64)
    mesh = pv.PolyData(_vertices, faces)
    shape = pv.read(args.shape) # get the shape of the boundary
    mesh = mesh.clip_surface(shape, invert=True)
    mesh.save(args.out + 'train' + str(count) + '.stl')
    print('saved in ' + args.out + 'train' + str(count) + '.stl')
    print()

def sample_volume(mesh, num_points):
    return trimesh.sample.volume_mesh(mesh, num_points)

def sample_points_with_volume_mesh(mesh, n, num_threads=8, batch_size=128**3):
    try:

        points = []
        points_per_thread = min(batch_size, n // num_threads)
        total_batches = (n + points_per_thread - 1) // points_per_thread

        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(total_batches):
                futures = [executor.submit(sample_volume, mesh, points_per_thread) for _ in range(num_threads)]
                results = [future.result() for future in futures]
                for sublist in results:
                    points.extend(sublist)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break

        return np.array(points[:n])
    except Exception as e:
        print(f"Error in process: {e}")
        raise

def main(args):

    torch.set_float32_matmul_precision('high') # TF32 optimization

    out_dir = '../out/'
    dataset_dir = args.dataset
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = models.SurfaceModel(rff_sigma=args.rff_sigma)
    model.to(device)
    model.train()

    optimizer = optim.Adam( [{'params': model.parameters()}], lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)

    # tensorboard init
    cur_time = datetime.datetime.now().strftime("%m-%d@%H-%M-%S")
    tbdir= os.path.join(out_dir,'log', cur_time)
    writer= SummaryWriter(tbdir)

    red_points,blue_points = get_rb_points_from_txt(dataset_dir, args.line_resolution)
    red_points = torch.from_numpy(red_points).to(torch.float32).to(device)
    blue_points = torch.from_numpy(blue_points).to(torch.float32).to(device)

    # prepaired for skeleton loss
    red_points_num, blue_points_num = red_points.shape[0], blue_points.shape[0]
    red_neg_ones_tensor = torch.full((red_points_num,1), -1.0).to(device)
    blue_ones_tensor = torch.full((blue_points_num,1), 1.0).to(device)
    
    epoch = args.n_iterations
    minibatch_size = args.n_samples//256

    # random_points = torch.empty(args.n_samples, 3).uniform_(0, 1) # random sample points if the boundary is cube
    random_points = sample_points_with_volume_mesh(trimesh.load(args.shape), args.n_samples) # sample points from the boundary mesh
    print("random points sampled done!")
    random_points = torch.from_numpy(random_points).float()

    dataset = TensorDataset(random_points)
    dataloader = DataLoader(dataset, batch_size=minibatch_size, num_workers=16, pin_memory=True, shuffle=True)

    for e in tqdm(range(epoch)):
        for idx,batch in enumerate(dataloader):
            count = e*(args.n_samples//minibatch_size)+idx
            optimizer.zero_grad()
            
            x = batch[0].to(device)
            x.requires_grad = True

            gauss_noise_eps = args.gauss_noise_eps
            red_gauss_noise = torch.randn(red_points.shape).to(device) * gauss_noise_eps
            blue_gauss_noise = torch.randn(blue_points.shape).to(device) * gauss_noise_eps
            out = model(x,red_points + red_gauss_noise,blue_points + blue_gauss_noise)
            
            red_tensor = out['red_res']
            blue_tensor = out['blue_res']
            skeleton_loss = (torch.abs(red_tensor-red_neg_ones_tensor).sum() + torch.abs(blue_tensor-blue_ones_tensor).sum()) / (red_points_num + blue_points_num)
            
            smooth_error = (out['df'].norm(p=2, dim=-1)).mean()

            alpha = args.alpha
            beta = args.beta
            loss = alpha*skeleton_loss + beta*smooth_error
            
            writer.add_scalar('Loss/train', loss.item(), count) 
            writer.add_scalar('Error/skeleton_loss', skeleton_loss.item(), count)
            writer.add_scalar('Error/smoothness', smooth_error.item(), count) 

            tqdm_str= '[Epoch={:<2d} | count={:<2d} | loss={:+.6f} | skeleton={:+.6f} | smooth={:+.6f}'.format(e,count,loss.item(),skeleton_loss.item(),smooth_error.item())
            pbar = tqdm()
            pbar.set_description(tqdm_str)

            loss.backward()
            optimizer.step()
            if count<=60000:
                scheduler.step() # update lr

            # save result
            if count>0 and count%20000==0:
                save_result(model, 0.0, count, args)

    torch.save(model.state_dict(),os.path.join(args.out,'model.pt'))
    print('model saved already!')

    # generate mesh
    resolution = args.resolution
    grid_size = 1/resolution
    d = 1 # cube size
    x = np.arange(0,d, grid_size)
    y = np.arange(0,d, grid_size)
    z = np.arange(0,d, grid_size)
    sample_x = np.array(np.meshgrid(x,y,z))
    sample_x = sample_x .reshape((3, resolution * resolution * resolution))
    sample_x = sample_x .transpose() # shape = (resolution * resolution * resolution, 3)
    sample_x = torch.from_numpy(sample_x).to(device).to(torch.float32)
    grid_fx = np.empty([sample_x.size()[0],1], dtype = np.float32) 
    print('start sample in network...')
    for i in range(resolution):
        with torch.no_grad():
            t = model(sample_x[resolution * resolution * i: resolution * resolution * (i + 1), :], only_f = True)
            t = t.detach().cpu().numpy()
        grid_fx[resolution * resolution * i: resolution * resolution * (i + 1), ...] = t     
    grid_fx = grid_fx.reshape((resolution, resolution, resolution))

    print('start marching cubes...')
    _vertices, _triangles = mcubes.marching_cubes(grid_fx, 0.0)
    _vertices[:, [0, 1]] = _vertices[:, [1, 0]]
    _vertices = _vertices/resolution #normalize to [0,1]
    write_ply_triangle(args.out + 'zero_mc0.obj', _vertices, _triangles) # save mesh
    faces = np.concatenate([np.full((_triangles.shape[0], 1), 3), _triangles], axis=1).astype(np.int64)
    mesh = pv.PolyData(_vertices, faces)
    shape = pv.read(args.shape) # get the shape of the boundary
    mesh = mesh.clip_surface(shape, invert=True)
    mesh.save(args.out + 'zero_mc.stl')
    print('vertices num:' + str(len(_vertices)) + ' triangles num:' + str(len(_triangles)))
    print('saved in ' + args.out + 'zero_mc.stl')
    print()
    return 

if __name__ == '__main__':
    for i in range(1):
        parser = argparse.ArgumentParser()
        parser.add_argument('--out', type=str, default='../out/generation/u_shape1/')
        parser.add_argument('--dataset', type=str, default='../dataset/dual_flow_skeleton/u_shape_skeleton300.txt') # dual skeleton
        parser.add_argument('--shape', type=str, default='../dataset/boundary_shape/u_shape.obj') # dual skeleton
        parser.add_argument('--line_resolution', type=float, default=1/256)
        parser.add_argument('--resolution', type=float, default=256)
        parser.add_argument('--gauss_noise_eps', type=int, default=2e-3)
        parser.add_argument('--n_samples', type=int, default=128**3*4)
        parser.add_argument('--lr', type=float, default=3e-5)
        parser.add_argument('--alpha', type=float, default=5000)
        parser.add_argument('--beta', type=float, default=1)
        parser.add_argument('--n_iterations', type=int, default=200)
        parser.add_argument('--seed', type=int, default=1)
        parser.add_argument('--boundary', type=str, default='../bdry/empty.txt') 
        parser.add_argument('--rff_sigma', type=float, default=2) # 2 4 6 8 

        args = parser.parse_args()
        print(args)
        main(args)
        print(args)

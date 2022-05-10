# TODO list
# Particle simulation -> Pixel-based (now), Convex hull (paper, extension)
# Least squares data -> Uses CVXOPT (paper, maybe extension)
# Come up with sensible action limits for theta, move_distance (from Sarvesh's simulator)
# Tune the variance of the Gaussians in Chi-Square to fit data from PyBullet
# Replace exhaustive search with BO for faster minimization

import copy
import torch
import numpy as np
from matplotlib import pyplot as plt
import imageio
import argparse
import os

from yaml import parse

class ObjectCentricTransport:

    def __init__(self, target_shape="square", target_size = 10, start_board = None):
        num_particles = 400 # approx
        self.target_shape = target_shape
        self.target_size = target_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.knife_half_length = 16
        if start_board is None:
            self.board_shape = np.array([64,96])
            board_size = self.board_shape[0] * self.board_shape[1]
            self.board = 1.0 * (torch.rand(self.board_shape[0], self.board_shape[1]) > 0.5*2*(board_size - num_particles)/board_size)
            self.board = self.board.to(self.device)
        else:
            self.board = start_board.to(self.device)
            self.board_shape = np.array([start_board.shape[0], start_board.shape[1]])


    def step(self, x, y, theta, move_distance, curr_board):
        board = copy.deepcopy(curr_board) # M x N
        coords = torch.nonzero(board).to(self.device) # P x 2
        R = torch.Tensor([[-np.sin(theta),-np.cos(theta)],
                        [np.cos(theta),-np.sin(theta)]]).to(self.device) # 2 x 2 x T
        transformed_coords = coords.unsqueeze(1).unsqueeze(1).float() @ R.permute(2,0,1) # P x T x 1 x 2
        apply_at = torch.Tensor([[x,y]]).to(self.device).permute(2,0,1) @ R.permute(2,0,1) #  T x 1 x 2  @ T x 2 x 2 = T x 1 x 2 

        indices_of_interest = torch.logical_and((apply_at[:,0,0] + move_distance).unsqueeze(0) > transformed_coords[:,:,0, 0], transformed_coords[:,:,0, 0]> apply_at[:, 0, 0].unsqueeze(0)) 
        # apply_at[:,0,0] is dim T which we unsqueeze to 1 x T, transformed_coords[:,:,0,0] is P x T hence indices of interest is P x T
        indices_of_interest = torch.logical_and(indices_of_interest, ((apply_at[:,0,1] + self.knife_half_length).unsqueeze(0) > transformed_coords[:,:,0,1]))
        indices_of_interest = torch.logical_and(indices_of_interest, (transformed_coords[:,:,0, 1]> (apply_at[:, 0,1] - self.knife_half_length).unsqueeze(0) ))
        
        to_move = transformed_coords[indices_of_interest]  # should give T x P x 1 x 2
        to_zero = coords[indices_of_interest]
        board[to_zero[:,0], to_zero[:,1]] = 0.0

        # Adding Chi-Square noise, i.e. simply sum of 2 squared Gaussian random variables
        # TODO - Tune the variance of the gaussian to fit data from PyBullet
        to_move[:,0] = apply_at[0,0] + move_distance + (torch.randn_like(to_move[:,0])**2 + torch.randn_like(to_move[:,0])**2)
        to_move = (to_move@ R.T).round().long()

        # indices_of_interest = torch.logical_and(to_move[:,0] >= 0, to_move[:,1] >= 0)
        # indices_of_interest = torch.logical_and(indices_of_interest, to_move[:,0] < self.board_shape[0])
        # indices_of_interest = torch.logical_and(indices_of_interest, to_move[:,1] < self.board_shape[1])
        # to_move = to_move[indices_of_interest]
        to_move = torch.maximum(to_move, torch.zeros_like(to_move).to(self.device))
        to_move = torch.minimum(to_move, torch.Tensor([[board.shape[0]-1, board.shape[1]-1]]).repeat((to_move.shape[0],1)).to(self.device))
        to_move = to_move.round().long()

        # occupied = to_move[board[to_move[:,0], to_move[:,1]] == 1.0]
        # board[to_move[:,0], to_move[:,1]][board[to_move[:,0], to_move[:,1]] == 0.0] = 1.0
        for x,y in to_move:
            self.board_recursion(x, y, board)
        # board[to_move[:,0], to_move[:,1]] = 1.0
        return board, self.lyapunov_function(board)
    
    def board_recursion(self, x,y, board):
        move = [[-1,-1], [-1,1], [1,-1], [1,1], [-1, 0], [0,-1], [1,0], [0,1]]
        if board[x,y] == 0.0:
            board[x,y] = 1.0
            return
        else:
            row = np.random.choice(8, 1)
            new_x, new_y = move[row[0]]
            self.board_recursion( min(x+new_x, board.shape[0]-1) , min(y+new_y, board.shape[1]-1), board)
    
    def lyapunov_function(self, board):
        assert self.target_shape in ["circle", "square"]
        object_locs = torch.nonzero(board)
        vec_values = board[object_locs[:,0], object_locs[:,1]]
        if self.target_shape == "circle":
            s1 = (object_locs - torch.Tensor(self.board_shape).to(self.device)/2).float()
            s2 = torch.norm(s1, dim = 1)
            vec_distances = torch.clamp( s2- self.target_size, min=0)
        elif self.target_shape == "square":
            x_min, x_max = (self.board_shape[0]-self.target_size)/2, (self.board_shape[0]+self.target_size)/2
            y_min, y_max = (self.board_shape[1]-self.target_size)/2, (self.board_shape[1]+self.target_size)/2

            vec_distances = torch.logical_and(object_locs[:,0] < x_min, object_locs[:,1] < y_min) * torch.norm((object_locs - torch.Tensor([x_min, y_min]).to(self.device)).float(), dim = 1)
            vec_distances += torch.logical_and(object_locs[:,0] < x_min, object_locs[:,1] > y_max) * torch.norm((object_locs - torch.Tensor([x_min, y_max]).to(self.device)).float(), dim = 1)
            vec_distances += torch.logical_and(object_locs[:,0] > x_max, object_locs[:,1] > y_max) * torch.norm((object_locs - torch.Tensor([x_max, y_max]).to(self.device)).float(), dim = 1)
            vec_distances += torch.logical_and(object_locs[:,0] > x_max, object_locs[:,1] < y_min) * torch.norm((object_locs - torch.Tensor([x_max, y_min]).to(self.device)).float(), dim = 1)

            vec_distances += torch.logical_and(torch.logical_and(object_locs[:,0] >= x_min, object_locs[:,0] <= x_max), object_locs[:,1] >= y_max) * (object_locs[:,1] - y_max)
            vec_distances += torch.logical_and(torch.logical_and(object_locs[:,0] >= x_min, object_locs[:,0] <= x_max), object_locs[:,1] <= y_min) * (y_min - object_locs[:,1])
            vec_distances += torch.logical_and(torch.logical_and(object_locs[:,1] >= y_min, object_locs[:,1] <= y_max), object_locs[:,0] >= x_max) * (object_locs[:,0] - x_max)
            vec_distances += torch.logical_and(torch.logical_and(object_locs[:,1] >= y_min, object_locs[:,1] <= y_max), object_locs[:,0] <= x_min) * (x_min - object_locs[:,0])
            
        else:
            raise NotImplementedError
        
        # # For visualizing the distances that contribute to the lyapunov function
        # viz = torch.zeros_like(board)
        # viz[object_locs[:,0], object_locs[:,1]] = vec_distances
        # X, Y = np.meshgrid(range(self.board_shape[0]), range(self.board_shape[1]))
        # hf = plt.figure()
        # ha = hf.add_subplot(111, projection='3d')
        # ha.plot_surface(X, Y, viz.cpu().numpy())
        # plt.show()
        
        return (vec_distances @ vec_values).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--gif', action='store_true')
    parser.add_argument('--gif_name', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--target_shape', type=str, default="square")
    parser.add_argument('--target_size', type=int, default=10)
    parser = parser.parse_args(None)

    if parser.gif and parser.gif_name is None:
        print("Bitch, GIF mangta hai par naam nahi specify kiya? ", parser.gif_name)
        exit()
        
    torch.set_grad_enabled(False)

    dynamics = ObjectCentricTransport( parser.target_shape, parser.target_size)
    # print("Number of particles on the board: ", torch.sum(dynamics.board))
    # plt.imshow(dynamics.board.cpu())
    # plt.show()

    # dynamics.lyapunov_function(dynamics.board, "square")
    # board, _ = dynamics.step(32,16, theta = np.pi/6, move_distance=10, curr_board = dynamics.board)
    # plt.imshow(board.cpu())
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Cutting Board')
    ax1.imshow(dynamics.board.cpu())
    ax1.set_title("Before")

    rend = []
    lyapunov_scores = []
    

    curr_lyp_score = dynamics.lyapunov_function(dynamics.board)
    print("initial score: ", curr_lyp_score)
    lyapunov_scores.append(curr_lyp_score)
    iter = 0
    while curr_lyp_score > 0:
        iter += 1
        best_board = None
        best_lyp_score = curr_lyp_score
        # This is exhaustive search -> Needs to be replaced with BO for faster performance
        for x in np.linspace(0,dynamics.board.shape[0],20):
            for y in np.linspace(0,dynamics.board.shape[1],20):
                for theta in [0., np.pi/2, np.pi, -np.pi/2]:
                    for move_distance in [20]: # np.linspace(2,32,5):
                        board, lyp_score = dynamics.step(x,y, theta, move_distance, dynamics.board)
                        if lyp_score < curr_lyp_score and lyp_score < best_lyp_score:
                            best_board = board
                            best_lyp_score = lyp_score
        if best_lyp_score >= curr_lyp_score or iter >= parser.max_iter:
            break
        dynamics.board = best_board
        print("curre particle numbers: ", torch.nonzero(best_board).shape[0] )
        rend.append((255*best_board.cpu().detach().numpy()).astype(np.uint8))
        curr_lyp_score = best_lyp_score
        lyapunov_scores.append(best_lyp_score)
        print("Step #{}: ".format(iter), best_lyp_score)
    
    if parser.gif:
        if not os.path.exists("gif_out"):
            os.makedirs("gif_out")
        imageio.mimsave("gif_out/"+parser.gif_name+".gif", rend, fps=2)


    ax2.imshow(dynamics.board.cpu())
    ax2.set_title("After")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(range(iter), lyapunov_scores)
    plt.title("Lyapunov score vs. knife swipes")
    plt.xlabel("# of knife swipes")
    plt.ylabel("Lyapunov score")
    plt.savefig("./lyapunov_graph.jpg")

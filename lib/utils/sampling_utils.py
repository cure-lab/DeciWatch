import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.distributions as distributions
import numpy as np
import random


def make_random_mask(input, ratio, rand=True):
    N, T, C = input.shape
    rand_mat = torch.rand(N, T)
    k = round((1-ratio) * T)
    k_th_quant = torch.topk(rand_mat, k, largest = False)[0][:,-1:]
    bool_tensor = rand_mat <= k_th_quant
    desired_tensor = torch.where(bool_tensor,torch.tensor(1),torch.tensor(0)).cuda()
    mask_input = desired_tensor.unsqueeze(dim=-1)*input
    return mask_input

def make_random_seq_mask(input, ratio, seq, rand=True):
    N, T, C = input.shape
    rand_mat = torch.rand(N, T)
    k = round((1-ratio) * T)
    k_th_quant = torch.topk(rand_mat, k, largest = False)[0][:,-1:]
    bool_tensor = rand_mat <= k_th_quant
    desired_tensor = torch.where(bool_tensor,torch.tensor(1),torch.tensor(0)).cuda()
    mask_input = desired_tensor.unsqueeze(dim=-1)*input
    return mask_input

def make_random_out(input, ratio, rand=True):
    N, T, C = input.shape
    rand_mat = torch.rand(N, T)
    k = int((1-ratio) * T)
    mask_input=torch.empty(0,k,C).cuda()
    for i in range(N):
        choose=random.sample(range(0, T), k)
        choose.sort()
        mask_input=torch.cat((mask_input,input[i,choose,:].unsqueeze(0)),0)
    
    return mask_input

def make_uni_mask(input, ratio, rand=True):
    N, T, C = input.shape
    gap = int(10*ratio)
    if rand:
        start_ind = torch.randint(gap, size=(N, 1)).cuda() #control the start point
    else:
        start_ind = torch.zeros((N,1)).long().cuda()
    desired_tensor = torch.zeros_like(input[:, :, 0]).long().cuda()
    for i in range(N):
        choose=range(start_ind[i],T,gap)
        desired_tensor[i][choose]=torch.LongTensor([1]).cuda()
    print('mask ratio:',1-torch.sum(desired_tensor)/(N*T))
    mask_input = desired_tensor.unsqueeze(dim=-1)*input
    return mask_input

def make_uni_seq_mask(input, ratio, seq, rand=True): 
    N, T, C = input.shape
    gap = int(10*ratio)
    if rand:
        start_ind = torch.randint(gap, size=(N, 1)).cuda() #control the start point
    else:
        start_ind = torch.zeros((N,1)).long().cuda()
    desired_tensor = torch.zeros_like(input[:, :, 0]).long().cuda()
    for i in range(N):
        for j in range(seq):
            choose=range(start_ind[i]+j,T,gap+seq)
            desired_tensor[i][choose]=torch.LongTensor([1]).cuda()
    mask_input = desired_tensor.unsqueeze(dim=-1)*input
    print('mask ratio:',1-torch.sum(desired_tensor)/(N*T))
    return mask_input


def make_uni_out(input, ratio, rand=True):
    N, T, C = input.shape
    gap = int(10*ratio)
    if rand:
        start_ind = torch.randint(gap, size=(N, 1)).cuda() #control the start point
    else:
        start_ind = torch.zeros((N,1)).long().cuda()
    desired_tensor = torch.zeros_like(input[:, :, 0]).long().cuda()
    for i in range(N):
        choose=range(start_ind[i],T,gap)
        desired_tensor[i][choose]=torch.LongTensor([1]).cuda()
    bool_tensor = desired_tensor.unsqueeze(dim=-1).repeat(1,1,C).bool()
    mask_input = torch.reshape(torch.masked_select(input, bool_tensor.cuda()),(N,-1,C))
    return mask_input
 
def sort_error_by_thres(inputs_2d, inputs_3d, thre=0.05):
    # to control the error for each input of interpolator
    # it may be influenced by the gap of train/test errors, and imbalance error distribution for each sequence.
    B, T, K, C = inputs_2d.shape
    assert inputs_2d.shape == inputs_3d.shape
    error = torch.mean(torch.norm((inputs_2d - inputs_3d), dim=-1), dim=-1) #[b, t]
    bool_tensor = error <= thre

    desired_tensor = torch.where(bool_tensor, torch.tensor(1).cuda(), torch.tensor(0).cuda())
    print('mask ratio:',1-torch.sum(desired_tensor)/(B*T))
    mask_input = desired_tensor.unsqueeze(dim=-1)*inputs_2d.reshape(B, T, -1)

    return mask_input.reshape(B, T, K, C)

def sort_error_by_prob(inputs_2d, inputs_3d, ratio=0.1):
    # prob: the masked proportion is (ratio)
    # to control the ratio for each sequence!
    B, T, K, C = inputs_2d.shape
    assert inputs_2d.shape == inputs_3d.shape
    error = torch.mean(torch.norm((inputs_2d - inputs_3d), dim=-1), dim=-1) #[b, t]
    k = int((1-ratio) * T)
    # sort errors
    sorted_error, index = torch.sort(error, dim=1, descending=True) #from largest to smallest 
    select_index = index[:, :k]
    mask_input = torch.zeros_like(inputs_2d).cuda()

    for i in range(B):
        mask_input[i, select_index[i]] = inputs_2d[i, select_index[i]]
    assert mask_input.shape == inputs_2d.shape
    return mask_input

def make_uni_rand_mask(inputs_2d, uni_ratio, rand_ratio):
    # inputs_2d:[B,T,K*C]
    uni_out = make_uni_mask(inputs_2d, ratio=uni_ratio, rand=False)
    final_out = make_random_mask(uni_out, ratio=rand_ratio)
    print('final mask ratio:',torch.sum(torch.sum(final_out, dim=1)==0)/(final_out.shape[0]*final_out.shape[1]))
    return final_out

def make_uni_rand_out(inputs_2d, uni_ratio, rand_ratio):
    # inputs_2d:[B,T,K*C]
    uni_out = make_uni_out(inputs_2d, ratio=uni_ratio, rand=False) 
    final_out = make_random_out(uni_out, ratio=rand_ratio)
    return final_out
    
def velocity_select(input, strategy, select_rate):
    '''
    input: (batch, frame_num * joint_num * dim)
    strategy: "max" for finding the joint that moves fastest, "average" for averaging all joints, "norm" for L2 norm
    select_rate: the percentage of frames that is selected
    
    ps: the velocity is cauculated using L2 norm
    
    '''
    B, T, K, C = input.shape
    velocity = input[:, 1:,:,:]-input[:, :-1,:,:]
    velocity_norm=torch.norm(velocity, dim=-1)    

    if strategy=="max":
        velocity_selected=torch.max(velocity_norm,dim=-1)[0]
    elif strategy=="mean":
        velocity_selected=torch.mean(velocity_norm,dim=-1)
    elif strategy=="norm":
        velocity_selected=torch.linalg.norm(velocity_norm,dim=-1)  
    frame_num = len(velocity_selected[0])
    
    select_framenum=T*select_rate

    velocity_selected_sum=torch.sum(velocity_selected, axis=-1)
    velocity_selected_sum_average=velocity_selected_sum/select_framenum

    selected_frame = torch.zeros((B, T)).cuda()
    present_sum = torch.zeros((B, 1)).cuda()
    
    for j in range(B):
        for i in range(0,frame_num):
            present_sum[j] += velocity_selected[j, i]
            if present_sum[j] >= velocity_selected_sum_average[j]:
                selected_frame[j][i] = torch.LongTensor([1]).cuda()
                # print('bbbb',j,i)
                present_sum[j] -= velocity_selected_sum_average[j]

    mask_output = selected_frame.unsqueeze(dim=-1).unsqueeze(dim=-1)*input
    print('mask ratio:',1-torch.sum(selected_frame)/(B*T))      
    return mask_output    
    
# if __name__=="__main__":
#     dict=np.load("result_train.npz")
#     joint_info=dict["pred_joints"][0:498]
    
#     frames=select(joint_info,"average",0.2)

import numpy as np
import torch

def set_seed(seed):
    torch.manual_seed(seed) # Set seed for Python's built-in random number generator
    np.random.seed(seed) # Set seed for numpy
    if torch.cuda.is_available(): # Set seed for CUDA if available
        torch.cuda.manual_seed_all(seed)
        # Set cuDNN's random number generator seed for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def f1_score(outputs, targets=None, M=400, mult=1, offset=0, min_num=1, max_num=1000, device=torch.device("cpu")):
    num_classes = outputs.shape[-1]
    prob_ord = torch.argsort(outputs, dim=-1, descending=True)
    log_prob = torch.gather(outputs, -1, prob_ord)
    sample = torch.bernoulli(torch.sigmoid(log_prob).repeat(M, 1, 1))
    cum_sum = torch.cat([torch.zeros([M,outputs.shape[0],1],device=device), torch.cumsum(sample, -1)], -1)
    rev_cum_sum = torch.cat([torch.flip(torch.cumsum(torch.flip(sample, [-1]), -1), [-1]), torch.zeros([M,outputs.shape[0],1],device=device)], -1)
    f1_values = cum_sum / (cum_sum + 0.5*(torch.arange(num_classes+1,device=device)-cum_sum) + 0.5*rev_cum_sum)
    f1_values[:,:,0] = 0
    f1_expected = torch.nanmean(f1_values, 0)
    # print(f1_expected)
    pred_num = torch.argmax(f1_expected, -1)
    pred_num = torch.maximum(torch.minimum(torch.round(mult*pred_num+offset), torch.Tensor([outputs.shape[-1]]).to(device)), torch.zeros([1],device=device))
    pred_num = torch.maximum(torch.minimum(pred_num, torch.tensor(max_num)), torch.tensor(min_num))
    
    pred_num = pred_num.int()  
    if targets is None:
        pred_list = [prob_ord[i,:pred_num[i]] for i in range(outputs.shape[0])]
        return pred_list
    else:
        pred_list = [torch.cat([torch.ones([pred_num[i]],device=device), torch.zeros([outputs.shape[-1]-pred_num[i]],device=device)]) for i in range(outputs.shape[0])]
        targets_ordered = torch.gather(targets, -1, prob_ord)
        pred_ordered = torch.stack(pred_list)
        f1 = torch.sum(torch.logical_and(targets_ordered, pred_ordered), -1) / torch.sum(torch.logical_or(targets_ordered, pred_ordered), -1)
        return f1
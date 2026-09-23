# unified consistent algorithm
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import argparse
import sys


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run unified consistent algorithm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        choices=[64, 128, 256],
        help="Hidden layer width.",
    )
    parser.add_argument(
        "--check-intersection-space",
        type=str,
        default="heuristic",
        choices=["imaginary", "heuristic"],
        help="Method to search intersection space.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="consistent-right",
        choices=["span-recovery", "consistent-right", "consistent-wrong"],
        help="Program to run.",
    )
    return parser


# No-arg execution is treated as "help mode" to avoid running expensive experiments accidentally.
if __name__ == "__main__" and len(sys.argv) == 1:
    build_arg_parser().print_help()
    sys.exit(0)

import torch
import torch.nn as nn
import time
import numpy as np
import scipy

# use float64 for torch
torch.set_default_dtype(torch.float64)

# deep neural network
class DeepDNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, depth=9):
        # This implementation assumes a fixed pretrained architecture (depth=9).
        super(DeepDNN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, output_dim)
        
        # ====== random init ======
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        #         # nn.init.zeros_(m.bias)
        # ====================
        
        # ====== load model ======
        if hidden_dim == 64:
            state_dict = torch.load("68db5deee325620a018a1ba1/state_dict.pt", map_location="cpu")  # 64 dim
        elif hidden_dim == 128:
            state_dict = torch.load("68db5dfb8a418b4f3ec1e70c/state_dict.pt", map_location="cpu")  # 128 dim
        elif hidden_dim == 256:
            state_dict = torch.load("68d77c2b1e8717de2410cf3d/state_dict.pt", map_location="cpu")  # 256 dim
        else:
            raise ValueError(f"Unsupported hidden_dim: {hidden_dim}. Choose one of [64, 128, 256].")


        with torch.no_grad():
            self.hidden_layers[0].weight.copy_(state_dict["model.model.1.weight"])
            self.hidden_layers[0].bias.copy_(state_dict["model.model.1.bias"])
            
            self.hidden_layers[1].weight.copy_(state_dict["model.model.3.weight"])
            self.hidden_layers[1].bias.copy_(state_dict["model.model.3.bias"])
            
            self.hidden_layers[2].weight.copy_(state_dict["model.model.5.weight"])
            self.hidden_layers[2].bias.copy_(state_dict["model.model.5.bias"])
            
            self.hidden_layers[3].weight.copy_(state_dict["model.model.7.weight"])
            self.hidden_layers[3].bias.copy_(state_dict["model.model.7.bias"])
            
            self.hidden_layers[4].weight.copy_(state_dict["model.model.9.weight"])
            self.hidden_layers[4].bias.copy_(state_dict["model.model.9.bias"])
            
            self.hidden_layers[5].weight.copy_(state_dict["model.model.11.weight"])
            self.hidden_layers[5].bias.copy_(state_dict["model.model.11.bias"])
            
            self.hidden_layers[6].weight.copy_(state_dict["model.model.13.weight"])
            self.hidden_layers[6].bias.copy_(state_dict["model.model.13.bias"])
            
            self.hidden_layers[7].weight.copy_(state_dict["model.model.15.weight"])
            self.hidden_layers[7].bias.copy_(state_dict["model.model.15.bias"])
            
            self.hidden_layers[8].weight.copy_(state_dict["model.model.17.weight"])
            self.hidden_layers[8].bias.copy_(state_dict["model.model.17.bias"])

            self.output.weight.copy_(state_dict["model.model.19.weight"])
            self.output.bias.copy_(state_dict["model.model.19.bias"])
        # ====================
        
    def forward(self, x):
        #x = x.float()
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output(x)
        return x

# Normalize each row of W and compensate the next layer so end-to-end function is preserved.
def normalize_model(model):
    layers = list(model.hidden_layers) + [model.output]

    for i in range(len(layers) - 1):
        layer = layers[i]
        next_layer = layers[i+1]

        W = layer.weight.data
        b = layer.bias.data
        
        row_norms = W.norm(dim=1, keepdim=True)
        row_norms[row_norms == 0] = 1.0

        # row normalization
        W = W / row_norms
        b = b / row_norms.squeeze(1)

        layer.weight.data = W
        layer.bias.data = b

        # adjust next layer
        next_layer.weight.data = next_layer.weight.data * row_norms.T

    return model

# generate two random inputs with different label
def get_two_different_class_inputs(model, max_tries=10000):
    model.eval()
    for trial in range(max_tries):
        x1 = generate_random_input()
        x2 = generate_random_input()
        with torch.no_grad():
            y1 = torch.argmax(model(x1), dim=1).item()
            y2 = torch.argmax(model(x2), dim=1).item()
        if y1 != y2:
            return x1, x2
    return None, None

# get activation pattern associated with x_input (whitebox)
def get_relu_masks(model, x_input):
    x = x_input.view(1, -1)
    masks = []
    layers = []
    layers.extend(model.hidden_layers)

    for layer in layers:
        x = layer(x)
        mask = (x > 0).to(x_input.dtype)          # activation pattern
        masks.append(torch.diag(mask.squeeze()))  # store as diagonal matrix
        x = torch.relu(x)

    return masks
def extract_diag_list(tensor_list):
    return [t.diag().tolist() for t in tensor_list]

# get normal vector of decision boundary (whitebox)
def compute_theoretical_normal_full(model, x_boundary, D):
    model.eval()
    x = x_boundary.view(1, -1)

    weights = []
    for layer in model.hidden_layers:
        weights.append(layer.weight.to(x.dtype))
    weights.append(model.output.weight.to(x.dtype)) 

    # Effective linear map inside the current ReLU region: W_eff = W_out * D_{L-1} * ... * D_0 * W_0
    W_eff = weights[-1] 
    for i in reversed(range(len(D))): 
        W_eff = W_eff @ D[i] @ weights[i]

    output = model(x)
    top2 = torch.topk(output, 2, dim=1).indices[0]
    i, j = top2[0].item(), top2[1].item()
    n = W_eff[i] - W_eff[j]
    n_unit = n / n.norm()

    return n_unit, extract_diag_list(D)
def compute_theoretical_normal_given_tops(model, x_boundary, top1, top2, D):
    model.eval()
    x = x_boundary.view(1, -1)

    weights = []
    for layer in model.hidden_layers:
        weights.append(layer.weight.to(x.dtype))
    weights.append(model.output.weight.to(x.dtype)) 

    W_eff = weights[-1] 
    for i in reversed(range(len(D))): 
        W_eff = W_eff @ D[i] @ weights[i]

    n = W_eff[top1] - W_eff[top2]
    n_unit = n / n.norm()

    return n_unit, extract_diag_list(D)

# binary search
def generate_random_input():
    return torch.randn(1, 1, 28, 28)  # (batch=1, channels=1, height=28, width=28)
def find_decision_boundary(model, x1, x2, tol=1e-20, max_iter=100):
    x1 = x1.clone().detach().requires_grad_(False).view(1, -1)
    x2 = x2.clone().detach().requires_grad_(False).view(1, -1)

    label1 = torch.argmax(model(x1), dim=1).item()
    label2 = torch.argmax(model(x2), dim=1).item()

    if label1 == label2:
        raise ValueError("error, x1 and x2 have the same label.")

    for iter in range(max_iter):
        midpoint = (x1 + x2) / 2.0
        mid_label = torch.argmax(model(midpoint), dim=1).item()

        if mid_label == label1:
            x1 = midpoint
        else:
            x2 = midpoint

        # if norm is less than tol, break
        if torch.norm(x1 - x2).item() < tol:
            break

        # convert numpy
        x1_np = x1.detach().cpu().numpy()
        x2_np = x2.detach().cpu().numpy()

        # Stop if midpoint updates are below floating-point resolution.
        next_np = np.nextafter(x1_np, x2_np)
        if np.all(next_np == x2_np):
           iter = -1
           break

    return x1, x2, iter

# compute target neuron
def compute(x, target_round, target_index):
    x = x.reshape(-1)
    for r in range(target_round + 1):
        y = model.hidden_layers[r].weight @ x + model.hidden_layers[r].bias
        x = torch.relu(y)
    return y[target_index]

# search intersection points (whitebox)
def finding_relu_boundary(model, init, target_round, target_neuron):

    def diff_num_place(net, net_walk):
        find = 0
        for i in range(len(net)):
            for j in range(len(net[i])):
                if abs(net[i][j] - net_walk[i][j]) > 1e-10:
                    find = find + 1
        return find
        
    def diff_place(net, net_walk):
        for i in range(len(net)):
            for j in range(len(net[i])):
                if abs(net[i][j] - net_walk[i][j]) > 1e-10:
                    return i,j
        return None, None
    
    
    # need customize, if the target inputs of the neuron is far from the boundary, give up
    x = compute(init, target_round, target_neuron)
    if abs(x) > 0.01:
        return False

    # get activation pattern associated with init
    current_D = get_relu_masks(model, init.reshape(-1))
    current_decision_normal, current_net = compute_theoretical_normal_full(model, init, current_D)

    # check consistency with prob_info
    flag = True
    for i, row in enumerate(prob_info):
        for j, val in enumerate(row):
            if (val == 1) & (current_net[i][j] == 0):
                flag = False
                break
            elif (val == 0) & (current_net[i][j] == 1):
                flag = False
                break
        if flag == False:
            break
    if flag == False:
        return False
    
    # linear map up to target_round
    M0 = torch.eye(input_dim)
    for round in range(target_round):
        M0 = current_D[round] @ model.hidden_layers[round].weight @ M0

    # Move direction is projected to be tangent to the class boundary (orthogonal to decision normal).
    dir = M0.T @ model.hidden_layers[target_round].weight[target_neuron]
    dir /= dir.norm()
    move_dir = dir - torch.dot(dir, current_decision_normal) * current_decision_normal
    move_dir /= move_dir.norm()

    # if x is posiive, move in the negative direction.
    if x > 0:
        move_dir *= -1

    # search points, where only one activation pattern changes
    dist1 = 1e-12
    max_iter = 100
    for iter_idx in range(max_iter):
        y = init + dist1 * move_dir
        new_D = get_relu_masks(model, y.reshape(-1))
        new_net = extract_diag_list(new_D)
        
        k = diff_num_place(new_net, current_net)
        if k == 0:
            dist1 = 2 * dist1
        elif k == 1:
            depth, index = diff_place(new_net, current_net)
            if depth == target_round and index == target_neuron:
                # print(f"finding_relu_boundary: iter={iter_idx + 1}/{max_iter} (hit target)")
                return True
            else:
                # print(f"finding_relu_boundary: iter={iter_idx + 1}/{max_iter} (single change but non-target)")
                return False
        else:
            dist1 = dist1 / 3
    # print(f"finding_relu_boundary: iter={max_iter}/{max_iter} (max_iter reached, last dist1={dist1:.3e})")
    return False

# get intersection space
def get_intersection_space_random(model, target_round, target_index, Ws, bs, Recovereds, method = "heuristic"):
    """
    get_intersection_space_random:
      - model, target_round. target_index, Ws (list), bs (list), Recovereds (list), method
      - method is heuristic or imaginary

    Returns:
      - M, X, intersection_point
      - M is the convert matrix
      - X is the input basis
      - intersection point is the intersection point
    """

    # check the target bit is possible
    if (prob_info[target_round][target_index] == 1) | (prob_info[target_round][target_index] == 0):
        print("we can't find this intersection space")
        return 
    
    trial = 0
    while True:
        trial += 1

        x1, x2 = get_two_different_class_inputs(model)
        y1, y2, iter = find_decision_boundary(model, x1, x2)
        boundary_point = (y1 + y2) / 2
        
        # search intersection point 
        flag = True
        if method == "heuristic":
            flag = finding_relu_boundary(model, boundary_point, target_round, target_index)
        if flag == False:
            continue

        intersection_point = boundary_point.clone()
        D = get_relu_masks(model, intersection_point)
        n1, net1 = compute_theoretical_normal_full(model, intersection_point, D)

        label1 = torch.argmax(model(y1), dim=1).item()
        label2 = torch.argmax(model(y2), dim=1).item()

        # check prob info
        flag = True
        for i, row in enumerate(prob_info):
            for j, val in enumerate(row):
                if (val == 1) & (net1[i][j] == 0):
                    flag = False
                    break
                elif (val == 0) & (net1[i][j] == 1):
                    flag = False
                    break
            if flag == False:
                break
        if flag == False:
            continue

        break


    # Flip the target ReLU bit
    if D[target_round][target_index][target_index] > 0.5:
        D[target_round][target_index][target_index] = 0
    else:
        D[target_round][target_index][target_index] = 1
    n2, net2 = compute_theoretical_normal_given_tops(model, intersection_point, label1, label2, D)

    # get null space
    A = torch.stack([n1, n2], dim=0)
    _, _, Vh = torch.linalg.svd(A)
    X = Vh[-(input_dim - 2):].T

    # Build local activation masks under recovered neuron states.
    x = intersection_point.reshape(-1)
    myD = []
    for layer in range(len(Ws)):
        x = Ws[layer] @ x + bs[layer]
        mask = (x > 0).to(x.dtype)
        persistent_mask = torch.tensor([r == "persistent" for r in Recovereds[layer]], dtype=torch.bool, device=x.device)
        mask = torch.where(persistent_mask, torch.tensor(1, dtype=x.dtype, device=x.device), mask)
        myD.append(torch.diag(mask.squeeze()))
        x_relu  = torch.relu(x)
        x = torch.where(persistent_mask, x, x_relu)

    # Local linear map
    M = torch.eye(input_dim)
    for round in range(target_round):
        M = myD[round] @ Ws[round] @ M
    
    return M, X, intersection_point.reshape(-1)
def get_intersection_space_random_crosslayer(model, target_round, target_index, Ws, bs, Recovereds, method = "heuristic"):
    """
    get_intersection_space_random_crosslayer:
      - model, target_round. target_index, Ws (list), bs (list), Recovereds (list), method
      - method is heuristic or imaginary

    Returns:
      - M, X, intersection_point
      - M is the convert matrix
      - intersection_point is the intersection point
    """

    # check the target bit is possible
    if (prob_info[target_round][target_index] == 1) | (prob_info[target_round][target_index] == 0):
        print("we can't find this intersection space")
        return 
    
    trial = 0
    while True:
        trial += 1

        x1, x2 = get_two_different_class_inputs(model)
        y1, y2, iter = find_decision_boundary(model, x1, x2)
        boundary_point = (y1 + y2) / 2
        label1 = torch.argmax(model(y1), dim=1).item()
        label2 = torch.argmax(model(y2), dim=1).item()


        # search intersection point 
        flag = True
        if method == "heuristic":
            flag = finding_relu_boundary(model, boundary_point, target_round, target_index)

        if flag == False:
            continue

        D = get_relu_masks(model, boundary_point)
        n1, net1 = compute_theoretical_normal_full(model, boundary_point, D)

        # check prob info
        flag = True
        for i, row in enumerate(prob_info):
            for j, val in enumerate(row):
                if (val == 1) & (net1[i][j] == 0):
                    flag = False
                    break
                elif (val == 0) & (net1[i][j] == 1):
                    flag = False
                    break
            if flag == False:
                break
        if flag == False:
            continue

        #
        break


    # Flip the target ReLU bit
    if D[target_round][target_index][target_index] > 0.5:
        D[target_round][target_index][target_index] = 0
    else:
        D[target_round][target_index][target_index] = 1
    n2, net2 = compute_theoretical_normal_given_tops(model, boundary_point, label1, label2, D)

    # get null space
    A = torch.stack([n1, n2], dim=0)  # shape = (2, 784)
    _, _, Vh = torch.linalg.svd(A)
    X = Vh[-(input_dim - 2):].T

    # Build local activation masks under recovered neuron states.
    x = boundary_point.reshape(-1)
    myD = []
    for layer in range(len(Ws)):
        x = Ws[layer] @ x + bs[layer]
        mask = (x > 0).to(x.dtype) 
        persistent_mask = torch.tensor([r == "persistent" for r in Recovereds[layer]], dtype=torch.bool, device=x.device)
        mask = torch.where(persistent_mask, torch.tensor(1, dtype=x.dtype, device=x.device), mask) 
        myD.append(torch.diag(mask.squeeze()))  
        x_relu  = torch.relu(x)
        x = torch.where(persistent_mask, x, x_relu)

    # Local linear map
    M0 = torch.eye(input_dim)
    for round in range(target_round - 1):
        M0 = myD[round] @ Ws[round] @ M0
    M1 = myD[target_round - 1] @ Ws[target_round - 1] @ M0
    M = torch.cat([M0, M1], dim=0) 


    
    return M, X, boundary_point.reshape(-1)









############################
# unified consistent 
############################
def check_unified(right_set = True, method = "heuristic"):


    
    ###########################################
    # 4th round
    ###########################################
    # print("Recovering the 4th weight", flush=True)
    target_round = 3

    # suppose we have all weight/biases except for persistent / dead neurons
    W3 = model.hidden_layers[target_round].weight.detach().clone()
    b3 = model.hidden_layers[target_round].bias.detach().clone()
    recovered3 = ["true"] * hidden_dim

    for i in range(len(prob_info[3])):
        if (prob_info[3][i] == 1) or (prob_info[3][i] == 0):
            W3[i] = torch.zeros_like(W3[i])
            b3[i] = 0
            recovered3[i] = "false"



    # ##########################################
    # 3rd round (cross layer)
    # ##########################################
    target_round = 4
    target_neuron = 7
    if hidden_dim == 256:
        target_neuron = 8
        target_neuron = 0

    torch.set_printoptions(edgeitems=1000)


    target_neuron0 = target_neuron
    if right_set == False:
        target_neuron0 = target_neuron + 1

    # 
    num_collect = 4
    print(f"Use {num_collect} intersecttion spaces", flush=True)

    #
    for trial in range(1,1001):

        for cnt in range(num_collect):
            if cnt == 0:
                Mi, Xi, _ = get_intersection_space_random(model, target_round, target_neuron0, [W0, W1, W2, W3], [b0, b1, b2, b3], [recovered0, recovered1, recovered2, recovered3], method)
            else:
                Mi, Xi, _ = get_intersection_space_random(model, target_round, target_neuron, [W0, W1, W2, W3], [b0, b1, b2, b3], [recovered0, recovered1, recovered2, recovered3], method)
            Li = Mi @ Xi

            if cnt == 0:
                M = Mi @ Mi.T
                L = Li @ Li.T
            else:
                M += Mi @ Mi.T
                L += Li @ Li.T
        
        row_norm = torch.linalg.norm(M, dim=1)
        keep = torch.nonzero(row_norm != 0).squeeze(1)
        M_reduced = M[keep][:, keep]
        L_reduced = L[keep][:, keep]            
        S_L, Q_L_reduced = torch.linalg.eigh(L_reduced)
        
        recovered = torch.zeros_like(L[:,0])
        recovered[keep] = Q_L_reduced[:,0]

        recovered = recovered / torch.linalg.norm(recovered, ord = 2)
        correct = model.hidden_layers[target_round].weight[target_neuron]

        mask = torch.abs(recovered) > 1e-8
        mask_f = mask.to(correct.dtype)
        correct = correct * mask_f
        correct = correct / correct.norm()

        ratio2 = S_L[0] / S_L[1] 

        print(ratio2.item(), flush=True)




############################
# unified consistent for cross layer
############################
def check_unified_crossLayer(right_set = True, method = "heuristic"):


    
    ###########################################
    # 4th round
    ###########################################
    # print("Recovering the 4th weight", flush=True)
    target_round = 3

    # suppose we have all weight/biases except for persistent / dead neurons
    W3 = model.hidden_layers[target_round].weight.detach().clone()
    b3 = model.hidden_layers[target_round].bias.detach().clone()
    recovered3 = ["true"] * hidden_dim

    mask = torch.tensor(prob_info[target_round - 1], device=b3.device, dtype=b3.dtype) == 1
    b3 = b3 + W3 @ (model.hidden_layers[target_round - 1].bias * mask)

    for i in range(len(prob_info[3])):
        if (prob_info[3][i] == 1) or (prob_info[3][i] == 0):
            W3[i] = torch.zeros_like(W3[i])
            b3[i] = 0
            recovered3[i] = "false"



    # ##########################################
    # 3rd round (cross layer)
    # ##########################################
    target_round = 4
    target_neuron = 0
    if hidden_dim == 64:
        target_neuron = 4
    if hidden_dim == 256:
        target_neuron = 8
        target_neuron = 0

    torch.set_printoptions(edgeitems=1000)


    target_neuron0 = target_neuron
    if right_set == False:
        target_neuron0 = target_neuron + 1

    # 
    num_collect = 4
    print(f"use at most {num_collect} intersection spaces", flush=True)
    print(f"score using 2 intersection spaces, gap using 2 intersection spaces, score using 3, gap using 3, score using 4, gap using 4")

    for trial in range(1,1001):

        for cnt in range(num_collect):
            if cnt == 0:
                Mi, Xi, _ = get_intersection_space_random_crosslayer(model, target_round, target_neuron0, [W0, W1, W2, W3], [b0, b1, b2, b3], [recovered0, recovered1, recovered2, recovered3], method)
            else:
                Mi, Xi, _ = get_intersection_space_random_crosslayer(model, target_round, target_neuron, [W0, W1, W2, W3], [b0, b1, b2, b3], [recovered0, recovered1, recovered2, recovered3], method)
            Li = Mi @ Xi

            if cnt == 0:
                M = Mi @ Mi.T
                L = Li @ Li.T
                D = (torch.norm(Mi, p=2, dim=1) > 1e-12)
            else:
                M += Mi @ Mi.T
                L += Li @ Li.T
                D = D & (torch.norm(Mi, p=2, dim=1) > 1e-12)
            
            if cnt > 0:

                # Solve generalized eigenproblem M v = λ (L + eps I) v; score uses Rayleigh quotient ratio.
                def direction_exists_score(L, M, eps_scale=1e-10):
                    L_np = L.detach().cpu().numpy()
                    M_np = M.detach().cpu().numpy()

                    n = L_np.shape[0]
                    B_np = L_np + eps_scale * np.eye(n, dtype=L_np.dtype)

                    w, U = scipy.linalg.eigh(M_np, B_np)

                    lam1 = w[-1].real
                    lam2 = w[-2].real if n > 1 else 0.0
                    v_np = U[:, -1]

                    v_np = v_np / np.linalg.norm(v_np)
                    score = (v_np.conj() @ L_np @ v_np).real / (v_np.conj() @ M_np @ v_np).real
                    gap = float(lam1 / lam2) if lam2 > 0 else float("inf")

                    v = torch.from_numpy(v_np).to(L.device)
                    return float(score), gap, v, eps_scale * lam1
                
                score, gap, _, _ = direction_exists_score(L, M, 1e-5)

                print(score, gap, sep = ",", end = "")

                if cnt < num_collect - 1:
                    print(end = ",")


        print("", flush=True)




############################
# span recovery for cross layer
############################
def span_recovery_crossLayer(method = "heuristic"):

    ###########################################
    # 4th round
    ###########################################
    # print("Recovering the 4th weight", flush=True)
    target_round = 3

    # suppose we have all weight/biases except for persistent / dead neurons
    W3 = model.hidden_layers[target_round].weight.detach().clone()
    b3 = model.hidden_layers[target_round].bias.detach().clone()
    recovered3 = ["true"] * hidden_dim

    mask = torch.tensor(prob_info[target_round - 1], device=b3.device, dtype=b3.dtype) == 1
    b3 = b3 + W3 @ (model.hidden_layers[target_round - 1].bias * mask)

    for i in range(len(prob_info[3])):
        if (prob_info[3][i] == 1) or (prob_info[3][i] == 0):
            W3[i] = torch.zeros_like(W3[i])
            b3[i] = 0
            recovered3[i] = "false"

    persistent_indices = []
    for i in range(len(prob_info[3])):
        if (prob_info[3][i] == 1):
            persistent_indices.append(i)
    



    # ##########################################
    # 3rd round (cross layer)
    # ##########################################
    target_round = 4

    torch.set_printoptions(edgeitems=1000)

    #
    num_collect = 400 
    num_trial = 30
    num_collect_basis = 6
    if hidden_dim == 256:
        num_collect = 2000
        num_collect_basis = 15

    

    print(f"use at most {num_collect} intersection spaces", flush=True)

    #    
    for trial in range(0, num_trial):
        print(f"{trial + 1} trial", flush=True)

        basis = []
        for target_neuron in range(0, hidden_dim):
            
            if (prob_info[target_round][target_neuron] < 0.25) | (prob_info[target_round][target_neuron] > 0.75):
                continue

            start_time = time.perf_counter()
            for cnt in range(0, num_collect):
                cnt += 1

                Mi, Xi, _ = get_intersection_space_random_crosslayer(model, target_round, target_neuron, [W0, W1, W2, W3], [b0, b1, b2, b3], [recovered0, recovered1, recovered2, recovered3], method)
                Li = Mi @ Xi

                # DAND: always nonzero rows across samples, DOR: ever nonzero rows across samples.
                if cnt == 1:
                    M = Mi @ Mi.T
                    L = Li @ Li.T
                    DAND = (torch.norm(Mi, p=2, dim=1) > 1e-14)
                    DOR = (torch.norm(Mi, p=2, dim=1) > 1e-14)
                else:
                    M += Mi @ Mi.T
                    L += Li @ Li.T
                    DAND = DAND & (torch.norm(Mi, p=2, dim=1) > 1e-14)
                    DOR = DOR | (torch.norm(Mi, p=2, dim=1) > 1e-14)
                    
                # Count local persistent neurons
                num_local_persistent = 0
                for i in range(hidden_dim):
                    if (recovered3[i] == "true") & (DAND[hidden_dim + i] == True):
                        num_local_persistent += 1
                        
                # Count inactive neurons 
                sufficient_check = 0
                for i in range(hidden_dim):
                    if (recovered2[i] == "true") & (DOR[i] == False):
                        sufficient_check += 1

                # print(num_local_persistent, sufficient_check, flush=True)
                if (num_local_persistent == 0) and (sufficient_check == 0):
                    elapsed = time.perf_counter() - start_time
                    print(f"index {target_neuron} use {cnt} intersection spaces ({elapsed:.3f}s)", flush=True)
                    break
            
            # insufficient
            if cnt == num_collect:
                continue

            # Solve generalized eigenproblem M v = λ (L + eps I) v; score uses Rayleigh quotient ratio.
            def direction_exists_score(L, M, eps_scale=1e-13):
                L_np = L.detach().cpu().numpy()
                M_np = M.detach().cpu().numpy()

                n = L_np.shape[0]
                B_np = L_np + eps_scale * np.eye(n, dtype=L_np.dtype)

                w, U = scipy.linalg.eigh(M_np, B_np)

                lam1 = w[-1].real
                lam2 = w[-2].real if n > 1 else 0.0
                v_np = U[:, -1]

                v_np = v_np / np.linalg.norm(v_np)
                score = (v_np.conj() @ L_np @ v_np).real / (v_np.conj() @ M_np @ v_np).real
                gap = float(lam1 / lam2) if lam2 > 0 else float("inf")

                v = torch.from_numpy(v_np).to(L.device)
                return float(score), gap, v, eps_scale * lam1

            score, gap, rec, p = direction_exists_score(L, M)                
            print("||v^T L|| =", (rec @ L).norm().item(), "||v^T M|| =", (rec @ M).norm().item())

            rec1 = rec[hidden_dim:]
            rec2 = rec[:hidden_dim]

            rec1 = rec1 / rec1.norm()
            rec2 = rec2 / rec2.norm()


            # the target weight in the target layer
            mask1 = torch.abs(rec1) > 1e-10
            rec1 = torch.where(mask1, rec1, torch.tensor(0.0))  

            vec1 = model.hidden_layers[target_round].weight[target_neuron].clone()
            vec1 = torch.where(mask1, vec1, torch.tensor(0.0)) 
            vec1 = vec1 / vec1.norm()

            print(torch.dot(rec1, vec1), min((rec1 + vec1).norm(), (rec1 - vec1).norm()), flush=True)


            # basis of the persistent neurons
            mask2 = torch.abs(rec2) > 1e-10
            rec2 = torch.where(mask2, rec2, torch.tensor(0.0)) 
            
            vec2 = torch.zeros_like(model.hidden_layers[target_round - 1].weight[0])
            for i in range(hidden_dim):
                if (prob_info[3][i] == 1):
                    vec2 = vec2 + model.hidden_layers[target_round].weight[target_neuron][i] * model.hidden_layers[target_round - 1].weight[i]

            vec2 = torch.where(mask2, vec2, torch.tensor(0.0))
            vec2 = vec2 / vec2.norm()
            
            print(torch.dot(rec2, vec2), min((rec2 + vec2).norm(), (rec2 - vec2).norm()), flush=True)
            print("", flush=True)


            # 
            basis.append(rec2)

            # 
            if len(basis) >= num_collect_basis:
                break





        # check the rank
        V = torch.stack(basis).T
        V_rank = torch.linalg.matrix_rank(V.detach(), 1e-8)
        U, S, _ = torch.linalg.svd(V, full_matrices=False)

        # singular value
        print("singular values of the recovered basis = ", S)
        print("", flush=True)
        recovered_basis = U[:, :V_rank]
        


        W = model.hidden_layers[target_round - 1].weight
        mask = torch.tensor(prob_info[target_round - 2], device=W.device, dtype=W.dtype) != 0
        true_basis = (W[persistent_indices] * mask).T
        true_basis, _ = torch.linalg.qr(true_basis)



        sigma = torch.linalg.svdvals(recovered_basis.T @ true_basis)
        theta = torch.acos(torch.clamp(sigma, -1.0, 1.0)) 
        print("principal angles (rad):", theta)
        print("# persistent neurons = ", V_rank)
        print("", flush=True)

        # # If we observe a rank deficient,
        # if V_rank < len(basis):
        #     W = model.hidden_layers[target_round - 1].weight
        #     mask = torch.tensor(prob_info[target_round - 2], device=W.device, dtype=W.dtype) != 0
        #     true_basis = (W[persistent_indices] * mask).T
        #     true_basis, _ = torch.linalg.qr(true_basis)



        #     sigma = torch.linalg.svdvals(recovered_basis.T @ true_basis)
        #     theta = torch.acos(torch.clamp(sigma, -1.0, 1.0)) 
        #     print("principal angles (rad):", theta)
        #     print("# persistent neurons = ", V_rank)
        #     break








############################
# Entry point: parse args, build model/statistics, then run selected routine.
############################
def parse_args():
    return build_arg_parser().parse_args()


def main(hidden_dim_arg=64, check_intersection_space="heuristic", run_mode="consistent-right"):
    global input_dim, hidden_dim, output_dim, depth, num_intersection_points, checkIntersectionSpace
    global model, prob_info
    global W0, b0, recovered0
    global W1, b1, recovered1
    global W2, b2, recovered2

    # # open log file
    # logfile = open("consistent-128.log", "w")
    # sys.stdout = logfile
    

    # parameter
    input_dim = 784
    hidden_dim = hidden_dim_arg
    output_dim = 10
    depth = 9
    num_intersection_points = 1000

    checkIntersectionSpace = check_intersection_space

    print("Simulation parameter")
    print("input_dim  = ", input_dim)
    print("hidden_dim = ", hidden_dim)
    print("output_dim = ", output_dim)
    print("depth      = ", depth)
    print(flush=True)


    # load model
    model = DeepDNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, depth=depth)
    model = normalize_model(model)
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(0)



    ###########################################
    # Prepare prob_info 
    ###########################################
    sizes = [hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim]
    prob_info = [ [0]*n for n in sizes ]
    for _ in range(1000):
        x1, x2 = get_two_different_class_inputs(model)         # prepare two points randomly

        y1, y2, iter = find_decision_boundary(model, x1, x2)   # binary search
        boundary_point = (y1 + y2) / 2
        label1 = torch.argmax(model(y1), dim=1).item()
        label2 = torch.argmax(model(y2), dim=1).item()

        D = get_relu_masks(model, boundary_point)
        net1 = extract_diag_list(D)

        prob_info = [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(prob_info, net1)]

    #
    th_dead = 0.02
    th_persistent = 0.98
    if hidden_dim == 128:
        th_dead = 0.01
        th_persistent = 0.99
    if hidden_dim == 256:
        th_dead = 0.005
        th_persistent = 0.995

    # modify prob info
    print("activation probability of each neuron with 1000 random input")
    prob_info = [[x / 1000 for x in row_a] for row_a in prob_info]
    for i, row in enumerate(prob_info):
        print(i + 1, prob_info[i])
        for j, index in enumerate(row):
            if prob_info[i][j] < th_dead:
                prob_info[i][j] = 0
                print(f" - {i + 1}-th layer, {j}-th index, dead", flush=True)
            elif prob_info[i][j] > th_persistent:
                prob_info[i][j] = 1
                print(f" - {i + 1}-th layer, {j}-th index, persistent", flush=True)
    print(flush=True)

    
    # rand reset
    torch.manual_seed(int(time.time()))

    ###########################################
    # 1st layer
    ###########################################
    # print("Recovering the 1st weight", flush=True)
    target_round = 0
    W0 = model.hidden_layers[target_round].weight.detach().clone()
    b0 = model.hidden_layers[target_round].bias.detach().clone()
    recovered0 = ["true"] * hidden_dim  # persistent, dead, true, false




    ###########################################
    # 2nd layer
    ###########################################
    # print("Recovering the 2nd weight", flush=True)
    target_round = 1
    W1 = model.hidden_layers[target_round].weight.detach().clone()
    b1 = model.hidden_layers[target_round].bias.detach().clone()
    recovered1 = ["true"] * hidden_dim




    ###########################################
    # 3rd layer
    ###########################################
    # print("Recovering the 3rd weight", flush=True)
    target_round = 2
    W2 = model.hidden_layers[target_round].weight.detach().clone()
    b2 = model.hidden_layers[target_round].bias.detach().clone()
    recovered2 = ["true"] * hidden_dim
    
    for i in range(len(prob_info[2])):
        if (prob_info[2][i] == 1):
            b2[i] = 0
            recovered2[i] = "persistent"
        if (prob_info[2][i] == 0):
            W2[i] = torch.zeros_like(W2[i])
            b2[i] = 0
            recovered2[i] = "dead"



    ###########################################
    # 4th layer
    ########################################### 
    if run_mode == "span-recovery":
        print("span recovery")
        span_recovery_crossLayer(checkIntersectionSpace)
    elif run_mode == "consistent-right":
        print("unified consistent (right set)")
        check_unified_crossLayer(right_set = True, method = checkIntersectionSpace)
    elif run_mode == "consistent-wrong":
        print("unified consistent (wrong set)")
        check_unified_crossLayer(right_set = False, method = checkIntersectionSpace)
    else:
        raise ValueError(f"Unknown run mode: {run_mode}")


if __name__ == "__main__":
    args = parse_args()
    main(
        hidden_dim_arg=args.hidden_dim,
        check_intersection_space=args.check_intersection_space,
        run_mode=args.run,
    )

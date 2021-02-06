
import torch 
import numpy as np
import torch.nn.utils.prune as prune

def rem_pruned_weights(AE_model):
    prune.remove(AE_model.E_conv1, 'weight')
    prune.remove(AE_model.E_conv2, 'weight')
    prune.remove(AE_model.E_conv3, 'weight')
    prune.remove(AE_model.E_conv4, 'weight')

    prune.remove(AE_model.A_conv1, 'weight')
    prune.remove(AE_model.A_conv2, 'weight')
    prune.remove(AE_model.A_conv3, 'weight')
    prune.remove(AE_model.A_conv4, 'weight')
    prune.remove(AE_model.A_conv5, 'weight')

    prune.remove(AE_model.lstm1, 'weight_ih_l0')
    prune.remove(AE_model.lstm1, 'weight_hh_l0')
    prune.remove(AE_model.lstm1, 'weight_ih_l0_reverse')
    prune.remove(AE_model.lstm1, 'weight_hh_l0_reverse')

    prune.remove(AE_model.fc1, 'weight')
    prune.remove(AE_model.fc2, 'weight')
    prune.remove(AE_model.fc3, 'weight')



def prune_model(AE_model, sparse_percent):
    parameters_to_prune = ((AE_model.E_conv1, 'weight'),
                            (AE_model.E_conv2, 'weight'),
                            (AE_model.E_conv3, 'weight'),
                            (AE_model.E_conv4, 'weight'),
                            (AE_model.A_conv1, 'weight'),
                            (AE_model.A_conv2, 'weight'),
                            (AE_model.A_conv3, 'weight'),
                            (AE_model.A_conv4, 'weight'),
                            (AE_model.A_conv5, 'weight'),
                            (AE_model.lstm1, 'weight_ih_l0'),
                            (AE_model.lstm1, 'weight_hh_l0'),
                            (AE_model.lstm1, 'weight_ih_l0_reverse'),
                            (AE_model.lstm1, 'weight_hh_l0_reverse'),
                            (AE_model.fc1, 'weight'),
                            (AE_model.fc2, 'weight'),
                            (AE_model.fc3, 'weight'))
    
    prune.global_unstructured(parameters_to_prune, 
                          pruning_method=prune.L1Unstructured, amount=sparse_percent)

def get_mask(AE_model):
    mask_array = dict()
    mask_array['E_conv1_mask'] = list(AE_model.E_conv1.named_buffers())[0][1]
    mask_array['E_conv2_mask'] = list(AE_model.E_conv2.named_buffers())[0][1]
    mask_array['E_conv3_mask'] = list(AE_model.E_conv3.named_buffers())[0][1]
    mask_array['E_conv4_mask'] = list(AE_model.E_conv4.named_buffers())[0][1]

    mask_array['A_conv1_mask'] = list(AE_model.A_conv1.named_buffers())[0][1]
    mask_array['A_conv2_mask'] = list(AE_model.A_conv2.named_buffers())[0][1]
    mask_array['A_conv3_mask'] = list(AE_model.A_conv3.named_buffers())[0][1]
    mask_array['A_conv4_mask'] = list(AE_model.A_conv4.named_buffers())[0][1]
    mask_array['A_conv5_mask'] = list(AE_model.A_conv5.named_buffers())[0][1]

    mask_array['lstm1_ih_mask'] = list(AE_model.lstm1.named_buffers())[0][1]
    mask_array['lstm1_hh_mask'] = list(AE_model.lstm1.named_buffers())[1][1]
    mask_array['lstm1_ih_rev_mask'] = list(AE_model.lstm1.named_buffers())[2][1]
    mask_array['lstm1_hh_rev_mask'] = list(AE_model.lstm1.named_buffers())[3][1]

    mask_array['fc1_mask'] = list(AE_model.fc1.named_buffers())[0][1]
    mask_array['fc2_mask'] = list(AE_model.fc2.named_buffers())[0][1]
    mask_array['fc3_mask'] = list(AE_model.fc3.named_buffers())[0][1]

    return mask_array

def update_weights(AE_model, masks):
    AE_model.E_conv1.weight = torch.nn.Parameter(AE_model.E_conv1.weight * masks['E_conv1_mask'])
    AE_model.E_conv2.weight = torch.nn.Parameter(AE_model.E_conv2.weight * masks['E_conv2_mask'])
    AE_model.E_conv3.weight = torch.nn.Parameter(AE_model.E_conv3.weight * masks['E_conv3_mask'])
    AE_model.E_conv4.weight = torch.nn.Parameter(AE_model.E_conv4.weight * masks['E_conv4_mask'])

    AE_model.A_conv1.weight = torch.nn.Parameter(AE_model.A_conv1.weight * masks['A_conv1_mask'])
    AE_model.A_conv2.weight = torch.nn.Parameter(AE_model.A_conv2.weight * masks['A_conv2_mask'])
    AE_model.A_conv3.weight = torch.nn.Parameter(AE_model.A_conv3.weight * masks['A_conv3_mask'])
    AE_model.A_conv4.weight = torch.nn.Parameter(AE_model.A_conv4.weight * masks['A_conv4_mask'])
    AE_model.A_conv5.weight = torch.nn.Parameter(AE_model.A_conv5.weight * masks['A_conv5_mask'])

    AE_model.lstm1.weight_ih_l0 = torch.nn.Parameter(AE_model.lstm1.weight_ih_l0 * masks['lstm1_ih_mask'])
    AE_model.lstm1.weight_hh_l0 = torch.nn.Parameter(AE_model.lstm1.weight_hh_l0 * masks['lstm1_hh_mask'])
    AE_model.lstm1.weight_ih_l0_reverse = torch.nn.Parameter(AE_model.lstm1.weight_ih_l0_reverse * masks['lstm1_ih_rev_mask'])
    AE_model.lstm1.weight_hh_l0_reverse = torch.nn.Parameter(AE_model.lstm1.weight_hh_l0_reverse * masks['lstm1_hh_rev_mask'])

    AE_model.fc1.weight = torch.nn.Parameter(AE_model.fc1.weight * masks['fc1_mask'])
    AE_model.fc2.weight = torch.nn.Parameter(AE_model.fc2.weight * masks['fc2_mask'])
    AE_model.fc3.weight = torch.nn.Parameter(AE_model.fc3.weight * masks['fc3_mask'])

    return AE_model


def print_sparse_percent(AE_model):
    print( "Sparsity in E_conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.E_conv1.weight == 0)) / float(AE_model.E_conv1.weight.nelement())))

    print("Sparsity in E_conv2.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.E_conv2.weight == 0)) / float(AE_model.E_conv2.weight.nelement())))

    print("Sparsity in E_conv3.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.E_conv3.weight == 0)) / float(AE_model.E_conv3.weight.nelement())))

    print( "Sparsity in E_conv4.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.E_conv4.weight == 0)) / float(AE_model.E_conv4.weight.nelement())))

    print("Sparsity in A_conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.A_conv1.weight == 0)) / float(AE_model.A_conv1.weight.nelement())))

    print("Sparsity in A_conv2.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.A_conv2.weight == 0)) / float(AE_model.A_conv2.weight.nelement())))

    print("Sparsity in A_conv3.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.A_conv3.weight == 0)) / float(AE_model.A_conv3.weight.nelement())))

    print("Sparsity in A_conv4.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.A_conv4.weight == 0)) / float(AE_model.A_conv4.weight.nelement())))

    print("Sparsity in A_conv5.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.A_conv5.weight == 0)) / float(AE_model.A_conv5.weight.nelement())))

    print("Sparsity in lstm1.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.lstm1.weight_ih_l0 == 0) + torch.sum(AE_model.lstm1.weight_hh_l0 == 0)
                         + torch.sum(AE_model.lstm1.weight_ih_l0_reverse == 0) +torch.sum(AE_model.lstm1.weight_hh_l0_reverse == 0))
                 / float(AE_model.lstm1.weight_ih_l0.nelement() + AE_model.lstm1.weight_hh_l0.nelement() 
                                  + AE_model.lstm1.weight_ih_l0_reverse.nelement() + AE_model.lstm1.weight_hh_l0_reverse.nelement())))

    print("Sparsity in fc1.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.fc1.weight == 0)) / float(AE_model.fc1.weight.nelement())))

    print("Sparsity in fc2.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.fc2.weight == 0)) / float(AE_model.fc2.weight.nelement())))

    print("Sparsity in fc3.weight: {:.2f}%".format(
            100. * float(torch.sum(AE_model.fc3.weight == 0)) / float(AE_model.fc3.weight.nelement())))

    print("Global sparsity: {:.2f}%".format(
            100. * float(torch.sum(AE_model.E_conv1.weight == 0)
                + torch.sum(AE_model.E_conv2.weight == 0)
                + torch.sum(AE_model.E_conv3.weight == 0)
                + torch.sum(AE_model.E_conv4.weight == 0) 
                + torch.sum(AE_model.A_conv1.weight == 0)
                + torch.sum(AE_model.A_conv2.weight == 0)
                + torch.sum(AE_model.A_conv3.weight == 0) 
                + torch.sum(AE_model.A_conv4.weight == 0)
                + torch.sum(AE_model.A_conv5.weight == 0)
                + torch.sum(AE_model.lstm1.weight_ih_l0 == 0)
                + torch.sum(AE_model.lstm1.weight_hh_l0 == 0)
                + torch.sum(AE_model.lstm1.weight_ih_l0_reverse == 0)
                + torch.sum(AE_model.lstm1.weight_hh_l0_reverse == 0)
                + torch.sum(AE_model.fc1.weight == 0)
                + torch.sum(AE_model.fc2.weight == 0)
                + torch.sum(AE_model.fc3.weight == 0))
                / 
                float(AE_model.E_conv1.weight.nelement()
                    + AE_model.E_conv2.weight.nelement()
                    + AE_model.E_conv3.weight.nelement()
                    + AE_model.E_conv4.weight.nelement()
                    + AE_model.A_conv1.weight.nelement()
                    + AE_model.A_conv2.weight.nelement()
                    + AE_model.A_conv3.weight.nelement()                    
                    + AE_model.A_conv4.weight.nelement()
                    + AE_model.A_conv5.weight.nelement()
                    + AE_model.lstm1.weight_ih_l0.nelement()
                    + AE_model.lstm1.weight_hh_l0.nelement()
                    + AE_model.lstm1.weight_ih_l0_reverse.nelement()
                    + AE_model.lstm1.weight_hh_l0_reverse.nelement()
                    + AE_model.fc1.weight.nelement()
                    + AE_model.fc2.weight.nelement()
                    + AE_model.fc3.weight.nelement())))


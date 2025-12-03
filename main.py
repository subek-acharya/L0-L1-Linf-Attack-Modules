import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from model_architecture import ResNet
import l0_attack
import linf_attack
import utils
import binary_search

def main():

    modelDir="./checkpoint/ResNet20-trOnlyBubbles-valOnlyBubbles-Grayscale.pth"
    
    # Parameters for the dataset
    batchSize = 64
    numClasses = 2
    inputImageSize = [1, 1, 40, 50]
    dropOutRate = 0.0

    # Define GPU device
    device = torch.device("cuda")

    # Create the ResNet model
    model = ResNet.resnet20(inputImageSize, dropOutRate, numClasses).to(device)
    # Load in the trained weights of the model
    checkpoint = torch.load(modelDir, weights_only=False)
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint["state_dict"])

    # Switch the model into eval model for testing
    model= model.eval()
    
    # val_OnlyBubbles_Grayscale dataset
    data = torch.load("./data/final_dataset_val_OnlyBubbles_Grayscale.pth", weights_only=False)
    images = data["data"].float()
    labels_binary = data["binary_labels"].long()
    labels_original = data["original_labels"].long()

    # Create a dataloader with only images and binary labels
    dataset = TensorDataset(images, labels_binary)
    valLoader = DataLoader(dataset, batch_size = batchSize, shuffle = False)

    # Check the clean accuracy of the model
    cleanAcc = utils.validateD(valLoader, model, device)
    print("Voter Dataset Clean Val Loader Acc:", cleanAcc)

    # Get correctly classified, classwise balanced samples to do the attack
    totalSamplesRequired = 1000
    correctLoader = utils.GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, valLoader, numClasses)

    # Check to make sure the accuracy is 100% on the correct loader
    correctAcc = utils.validateD(correctLoader, model, device)
    print("Clean Accuracy for Correct Loader:", correctAcc)

    # Check the number of samples in the correctLoader
    print("Number of samples in correctLoader:", len(correctLoader.dataset))

    # Get pixel bounds for a correctLoader
    minVal, maxVal = utils.GetDataBounds(correctLoader, device)
    print("Data Range for Correct Loader:", [round(minVal, 4), round(maxVal, 4)])

    # Attack Paramaters
    n_restarts = 10
    num_steps = 20
    step_size = 12
    sparsity=2000
    epsilon = 1
    kappa = 8
    random_start = True
    epsilonStep = epsilon/num_steps
    etaStart = 0.5 * epsilon 
    clipMin = 0.0
    clipMax = 1.0

    ########################### Linf Attack!!! #################

    # PGD attack with Cross Entropy Loss
    # advLoader_ce = linf_attack.PGDNativePytorch(device, correctLoader, model, epsilon, epsilonStep, num_steps, clipMin, clipMax, loss_type="ce")
    # advAcc_ce = utils.validateD(advLoader_ce, model, device)
    # print(f"Adversarial Accuracy (PGD with CE Loss): {advAcc_ce:.4f}")
    
    # PGD attack with DLR Loss
    # advLoader_dlr = linf_attack.PGDNativePytorch(device, correctLoader, model, epsilon, epsilonStep, num_steps, clipMin, clipMax, loss_type="dlr")
    # advAcc_dlr = utils.validateD(advLoader_dlr, model, device)
    # print(f"Adversarial Accuracy (PGD with DLR Loss): {advAcc_dlr:.4f}")

    # APGD with Cross Entropy Loss
    advLoader_apgd_ce = linf_attack.AutoAttackPytorchMatGPUWrapper(device, correctLoader, model, epsilon, etaStart, num_steps, clipMin=0, clipMax=1, loss_type="ce")
    advAcc_apgd_ce = utils.validateD(advLoader_apgd_ce, model, device)
    print(f"Adversarial Accuracy (APGD with CE Loss):  {advAcc_apgd_ce:.4f}")
    
    # APGD with DLR Loss
    # advLoader_apgd_dlr = linf_attack.AutoAttackPytorchMatGPUWrapper(device, correctLoader, model, epsilon, etaStart, num_steps, clipMin=0, clipMax=1, loss_type="dlr")
    # advAcc_apgd_dlr = utils.validateD(advLoader_apgd_dlr, model, device)
    # print(f"Adversarial Accuracy (APGD with DLR Loss): {advAcc_apgd_dlr:.4f}")

    ########################### LO Attack!!! ###################
    # print("L0_PGD: ")
    # l0_attack.L0_PGD_AttackWrapper(model, device, correctLoader, n_restarts,  num_steps, step_size, sparsity, random_start)

    # print("L0_Linf: ")
    # l0_attack.L0_Linf_PGD_AttackWrapper(model, device, correctLoader, n_restarts, num_steps, step_size, sparsity, epsilon, random_start)

    # print("L0_Sigma_PGD_AttackWrapper: ")
    # l0_attack.L0_Sigma_PGD_AttackWrapper(model, device, correctLoader, n_restarts, num_steps, step_size, sparsity, kappa, random_start)

    # Binary Search Parameters
    tau = 0.0  # Target robust accuracy threshold (0% means complete attack success)
    k_min = 1  # Minimum sparsity (minimum pixels to perturb)
    k_max = 2000  # Maximum sparsity (maximum pixels to perturb)
    tolerance = 10  # Stop when range is smaller than this
    # Uncomment the attack name to run the desired attack wrapper
    attack_name = 'L0_PGD_AttackWrapper'
    # attack_name = 'L0_Linf_PGD_AttackWrapper'
    # attack_name = 'L0_Sigma_PGD_AttackWrapper'

    # Uncomment to run biinary search to find the optimal sparsity value
    # binary_search.binary_search_optimal_k(model, device, correctLoader, n_restarts, num_steps, step_size, epsilon, kappa, random_start, tau, k_min, k_max, tolerance, attack_name)


if __name__ == '__main__':
    main()
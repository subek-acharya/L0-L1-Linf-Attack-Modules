import numpy as np
import L0_attack

def binary_search_optimal_k(model, device, correctLoader, n_restarts, num_steps, step_size, epsilon, kappa, random_start, tau, k_min, k_max, tolerance, attack_name):
    current_k_min = k_min
    current_k_max = k_max
    optimal_k = k_max
    optimal_robust_acc = 100.0
    iteration = 0
    k_mid = k_max  # Start with k_max
    
    while current_k_max - current_k_min > tolerance:
        iteration += 1

        # Call the appropriate attack function based on attack_name
        if attack_name == 'L0_PGD_AttackWrapper':
            robust_acc = l0_attack.L0_PGD_AttackWrapper(model, device, correctLoader, n_restarts, num_steps, step_size, k_mid, random_start)
        elif attack_name == 'L0_Linf_PGD_AttackWrapper':
            robust_acc = l0_attack.L0_Linf_PGD_AttackWrapper(model, device, correctLoader, n_restarts, num_steps, step_size, k_mid, epsilon, random_start)
        elif attack_name == 'L0_Sigma_PGD_AttackWrapper':
            robust_acc = l0_attack.L0_Sigma_PGD_AttackWrapper(model, device, correctLoader, n_restarts, num_steps, step_size, k_mid, kappa, random_start)
        
        print("Iteration: ", iteration, "Sparsity(k): ", k_mid, "Robust Accuracy: ", round(robust_acc, 2))
        
        if robust_acc > tau:
            current_k_min = k_mid + 1
        else:
            current_k_max = k_mid
            if k_mid < optimal_k:
                optimal_k = k_mid
                optimal_robust_acc = robust_acc
        
        k_mid = (current_k_min + current_k_max) // 2
    
    print("\nOptimal K:", optimal_k, ", Robust Accuracy:", round(optimal_robust_acc, 2), "%\n")
    
    return None
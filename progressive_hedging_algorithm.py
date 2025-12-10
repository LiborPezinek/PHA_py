import numpy as np
import matplotlib.pyplot as plt

# a dedicated solver for a single scenario for our problem with multiple products
def SolveScenarioMultiProduct(xbar, omega, purchaseCost, holdingCost, shortageCost, demand, rho):
    P = demand.shape[0] # number of products

    # Candidate points per product
    x_left = xbar - (purchaseCost - shortageCost + omega) / rho   # q <= d
    x_right = xbar - (purchaseCost + holdingCost + omega) / rho   # q >= d

    candidates = [np.zeros(P), demand, x_left, x_right]  # possible values of x (0 because x >= 0)

    # For each product, choose candidate with minimal value
    x_opt = np.zeros(P)
    for j in range(P):
        cand_j = np.unique(np.maximum(0.0, candidates))
        best_x = 0.0
        best_val = float("inf")
        
        # iterate over candidates for product j
        for x in cand_j:
            HoldingCostTotal = holdingCost[j] * max(x - demand[j], 0.0) # Excess inventory cost
            ShortageCostTotal = shortageCost[j] * max(demand[j] - x, 0.0)   # Unmet demand cost
            aug = 0.5 * rho * (x - xbar[j]) ** 2 + omega[j] * x
            val = purchaseCost[j] * x + HoldingCostTotal + ShortageCostTotal + aug
            if val < best_val:
                best_val = val
                best_x = x
        x_opt[j] = best_x

    # Compute scenario value with optimal x_opt - for logging, not used in PH
    hold_vec = holdingCost * np.maximum(x_opt - demand, 0.0)
    short_vec = shortageCost * np.maximum(demand - x_opt, 0.0)
    aug_vec = 0.5 * rho * (x_opt - xbar) ** 2 + omega * x_opt
    val = float(np.sum(purchaseCost * x_opt + hold_vec + short_vec + aug_vec))
    return x_opt, val

# progressive hedging for multiple-product inventory problem
def ProgressiveHedgingMultiProduct(purchase, holding, shortage, demands, probs,
                                   rho=1.0, eps=1e-6, max_iter=100, verbose=True
                                   ):
    # variables initialization
    S = len(demands)    # number of scenarios
    P = demands[0].shape[0]     # number of products
    x_bar = np.zeros(P)
    omegas = np.zeros((S, P))
    x_s = np.zeros((S, P))
    probs = probs / np.sum(probs)  # normalize probabilities

    print("\nPH multi-product")
    print(f"P={P}, S={S}, rho={rho}")
    print(f"probs={probs}")
    print(f"purchase={purchase}, holding={holding}, shortage={shortage}")
    print(f"eps={eps}, max_iter={max_iter}\n")


    # main PH loop
    for it in range(1, max_iter + 1):

        # Solve each scenario independently, vectorizing over products
        for s in range(S):
            x_opt, _ = SolveScenarioMultiProduct(
                xbar=x_bar, omega=omegas[s],
                purchaseCost=purchase, holdingCost=holding, shortageCost=shortage,
                demand=demands[s], rho=rho
            )
            x_s[s] = x_opt

        # Weighted average across scenarios for each product
        x_bar_new = np.einsum('s,sp->p', probs, x_s)  # shape (P,)
        
        # Dual update per scenario/product
        for s in range(S):
            omegas[s] += rho * (x_s[s] - x_bar_new)

        # Expected true cost (without augmentation)
        exp_cost = 0.0
        for s in range(S):
            x = x_s[s]
            d = demands[s]
            hold = holding * np.maximum(x - d, 0.0)
            short = shortage * np.maximum(d - x, 0.0)
            exp_cost += probs[s] * float(np.sum(purchase * x + hold + short))

        # Combined PH criterion:
        # sqrt( ||x_bar_prev - x_bar_new||^2 + sum_s p_s ||x_s[s] - x_bar_new||^2 )
        term1 = (np.linalg.norm(x_bar_new - x_bar, 2))**2
        converged = False
        for s in range(S):
            # per-scenario squared norms ||x_s[s] - x_bar_new||^2
            per_s_norm2 = (np.linalg.norm(x_s[s] - x_bar_new))**2
            term2 = float(np.sum(probs[s] * per_s_norm2))
            phi = float(np.sqrt(term1 + term2))
            
            if phi <= eps:
                converged = True
                break

        # Update x_bar
        x_bar = x_bar_new

        # log iteration info
        if verbose:
            print(f"Iter {it:2d}; x_bar={x_bar}; E[cost]={exp_cost:.2f}; phi={phi:.4e}; \n x_s=\n{x_s}; \n omegas=\n{omegas}\n\n")

        if converged:
            print(f"Converged in {it} iterations (phi ≤ eps).")
            break

    return x_bar, x_s, omegas, exp_cost


def DemoMulti():


    purchase = np.array([5.0, 6.0, 8.0])
    holding = np.array([2.0, 5.5, 12.0])
    shortage = np.array([13.0, 11.0, 10.0])

    demands = [
        np.array([50.0, 30.0, 20.0]),   # scenario 1
        np.array([100.0, 40.0, 30.0]),  # scenario 2
        np.array([150.0, 60.0, 40.0]),   # scenario 3
        np.array([80.0, 50.0, 70.0])    # scenario 4
    ]
    probs = np.array([0.3, 0.1, 0.4, 0.2])
    rho = 1.0


    # # the same input data as in the presentation
    # purchase = np.array([5.0, 6.0, 8.0])
    # holding = np.array([2.0, 1.5, 3.0])
    # shortage = np.array([130.0, 90.0, 110.0])

    # demands = [
    #     np.array([50.0, 30.0, 20.0]),   # scenario 1
    #     np.array([100.0, 40.0, 30.0]),  # scenario 2
    #     np.array([150.0, 60.0, 40.0]),   # scenario 3
    #     np.array([80.0, 50.0, 70.0])    # scenario 4
    # ]
    # probs = np.array([0.3, 0.1, 0.4, 0.2])
    # rho = 1.0

    ######################################################################

    plot = True
    S = len(demands)    # number of scenarios
    P = demands[0].shape[0]     # number of products

    x_bar, x_s, omegas, exp_cost = ProgressiveHedgingMultiProduct(
        purchase, holding, shortage, demands, probs,
        rho=rho, eps=1e-3, max_iter=1000, verbose=False,
    )

    print("\nFinal multi-product results:")
    print(" x_bar =", x_bar)
    print("\n E[cost] =", exp_cost, "\n")
    print(" x_s   =\n", x_s, "\n")
    #print(" omegas =\n", omegas, "\n")

    # plot PHA solutions against single-scenario optima
    if plot:
        x_opt_single = np.zeros((S, P))
        for s in range(S):
            x_opt_single[s] = SolveSingleScenarioMultiProductOptimal(purchase, holding, shortage, demands[s])
        
        PlotSolutions(x_opt_single, x_s, x_bar)




# plot PHA solutions against single-scenario optima
def PlotSolutions(x_opt_single, x_s, x_bar, title="Inventory Decisions Across Scenarios"):
    S, P = x_s.shape

    product_labels = [f"Product {i+1}" for i in range(P)]

    fig, axes = plt.subplots(1, S, figsize=(5*S, 4), sharey=True)
    if S == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14)

    for s, ax in enumerate(axes):
        ax.bar(range(P), x_opt_single[s], color="skyblue", alpha=0.8)
        ax.set_title(f"Scenario {s+1}")
        ax.set_xticks(range(P))
        ax.set_xticklabels(product_labels, rotation=0)
        ax.set_ylabel("Stock Levels")
        ax.set_xlabel("Products")

        # dashed global decision per product
        for j in range(P):
            ax.hlines(y=x_bar[j], xmin=j-0.4, xmax=j+0.4, colors="black", linestyles="--", label=None)

        # add a single legend entry for dashed line
        ax.plot([], [], linestyle="--", color="black", label="Global Decision")
        ax.legend(loc="upper left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# helper function for plot_solutions() to solve single scenario optimally without PH
def SolveSingleScenarioMultiProductOptimal(purchase, holding, shortage, demand):
        """
        Solve the scenario subproblem without augmentation terms.
        Find optimal x that minimizes: purchase*x + holding*max(x-d,0) + shortage*max(d-x,0)
        """
        P = demand.shape[0]
        x_opt = np.zeros(P)
        
        for j in range(P):
            # Candidates: 0 and demand[j]
            candidates = [0.0, demand[j]]
            best_x = 0.0
            best_val = float("inf")
            
            for x in candidates:
                hold_cost = holding[j] * max(x - demand[j], 0.0)
                short_cost = shortage[j] * max(demand[j] - x, 0.0)
                val = purchase[j] * x + hold_cost + short_cost
                
                if val < best_val:
                    best_val = val
                    best_x = x
            
            x_opt[j] = best_x
        
        return x_opt



if __name__ == '__main__':
    DemoMulti()

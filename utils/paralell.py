import math
import numpy as np

from itertools import combinations, combinations_with_replacement, product

def agent_name(eta_agent):
    """Agent abbreviation given agent's risk attitude."""
    if eta_agent == -1:
        return "rse"
    if eta_agent == 0:
        return "lin"
    if eta_agent == 1:
        return "log"
    return str(eta_agent)

def isoelastic_utility(x, eta):
    """Isoelastic utility for a given wealth.
    
    Args:
        x (array):
            Wealth vector.
        eta (float):
            Risk-aversion parameter.
    
    Returns:
        Vector of utilities corresponding to wealths. For log utility if wealth 
        is less or equal to zero, smallest float possible is returned. For other
        utilites if wealth is less or equal to zero, smallest possible utility, 
        i.e., specicfic lower bound is returned.
        
    Note:
        Not implementef for eta > 1.
    """
    u = np.zeros_like(x, dtype=float)
    if eta > 1:
        return ValueError("eta should be less than 1")
    if np.isclose(eta, 1):
        u[x > 0] = np.log(x[x > 0])
        u[x <= 0] = np.finfo(float).min
    else:
        bound = (-1) / (1 - eta)
        u[x > 0] = (np.power(x[x > 0], 1-eta) - 1) / (1 - eta)
        u[x <= 0] = bound
    return u
    
def inverse_isoelastic_utility(u, eta):
    """Inverse isoelastic utility function mapping from utility to wealth.
    
    Args:
        u (array):
            Utility vector.
        eta (float):
            Risk-aversion parameter.
    
    Returns:
        Vector of wealths coresponding to utilities. For 
    """
    if eta > 1:
        return ValueError("eta should be less than 1")
    if np.isclose(eta, 1):
        return np.exp(u)        
    else:
        bound = (-1) / (1 - eta)
        x = np.zeros_like(u, dtype=float)
        x[u > bound] = np.power(u[u > bound] * (1 - eta) + 1, 1 / (1 - eta))
        return x

def wealth_change(x, gamma, eta):
    """Apply isoelastic wealth change.
    
    Args:
        x (float):
            Initial wealth vector.
        gamma (float):
            Growth rate.
        eta (float):
            Wealth dynamic parameter.
    """
    return inverse_isoelastic_utility(isoelastic_utility(x, eta) + gamma, eta)
    
def shuffle_along_axis(a, axis):
    """Randomly shuffle multidimentional array along specified axis."""
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)

def create_gambles(c, n_fractals=9):
    """Create list of all gambles. 
    
    Args:
        c (float):
            Max growth rate for gamble space.
        n_fractals (int):
            Number of growth rate samples.
    Returns:
        List of arrays. Each gamble is represented as (2, ) array with growth 
        rates. For n fractals, n(n+1)/2 gambles are created. Order of growth 
        rates doesn't matter since probabilities are assigned equally to both 
        wealth changes.
    """
    gamma_range = np.linspace(-c, c, n_fractals)
    return [
        np.array([gamma_1, gamma_2]) 
        for gamma_1, gamma_2 
        in combinations_with_replacement(gamma_range, 2)
    ]    

def create_gamble_pairs(gambles):
    """Create list of all unique gamble pairs.
    
    Args:
        gambles (list of arrays):
            List of gambles.
    
    Returns:
        List of arrays. Each gamble pair is represented as (2, 2) array with 
        four growth rates for both gambles. Rows corresponds to gambles, columns 
        correspond to individual growth rates within a gamble. All pairs contain
        two unique gambles. For n gambles, n(n-1)/2 gamble pairs are created.
    """    
    return [
        np.concatenate((g1[np.newaxis], g2[np.newaxis]), axis=0) 
        for g1, g2 in combinations(gambles, 2)
        ]

def create_trial_order(n_simulations, n_gamble_pairs, n_trials):
    """Generates randomized trial order for paralell simulations.
    
    Args:
        n_simulations (int):
            Number of paralell simulations.
        n_gamble_pairs (int):
            Number of unique, available gamble pairs.
        n_trials (int):
            Number of experimental trials.
            
    Returns:
        Array of shape n_trials x n_simulations with indices corresponding to 
        gamble pairs.
    """
    # indicates how many times gamble pairs should be repeated to span n_trials
    repetition_factor = math.ceil(n_trials / n_gp)

    basic_order = np.arange(n_gamble_pairs)[:, np.newaxis]
    trial_order = np.repeat(basic_order, n_simulations, axis=1)
    trial_order = shuffle_along_axis(trial_order, axis=0)
    trial_order = np.tile(trial_order, (repetition_factor, 1))
    trial_order = trial_order[:n_trials]
    trial_order = shuffle_along_axis(trial_order, axis=0)
    
    return trial_order

def create_experiment(gamble_pairs):
    """Creates experiment array.
    
    Args:
        gamble_pairs (list of arrays):
            List of gamble pairs.
    
    Returns:
        Array of size (2, 2, n_trials). First two dimensions correspond to 
        gamble pair, third dimension correspond to subsequent trials.  
    """
    return np.stack(gamble_pairs, axis=2)

def create_mag_thrs(c, n_fractals):
    """Generate bounds controlling min and max average growth rate.
    
    Args:
        c (float):
            Maximal fractal growth rate.
        n_fractals (int):
            Number of growth rate samples.
            
    Returns:
        Array of bound values. First value corresponds to lower bound, last to
        upper bound. Number of levels is 2 * n_fractals.
    """
    half_step = c / (2 * (n_fractals - 1))
    return np.linspace(-c-half_step, c+half_step, 2*n_fractals) 

def create_var_thrs(c, n_fractals):
    """Generate bounds controlling min and max gamble variance.
    
    Args:
        c (float):
            Maximal fractal growth rate.
        n_fractals (int):
            Number of growth rate samples.
            
    Returns:
        Array of bound values. First value corresponds to lower bound, last to
        upper bound. Number of levels is 2 * n_fractals.
    """
    half_step = c / (n_fractals - 1)
    return np.linspace(-half_step, 2*c+half_step, n_fractals+1)

def create_dmag_thrs(c, n_fractals):
    """Generate bounds controlling difference average growth rate.
    
    Args:
        c (float):
            Maximal fractal growth rate.
        n_fractals (int):
            Number of growth rate samples.
            
    Returns:
        Array of bound values. First value corresponds to lower bound, last to
        upper bound. Number of levels is 2 * n_fractals.
    """
    half_step = c * np.sqrt(2) / (2*n_fractals - 2)
    return np.linspace(-half_step, 2*c*np.sqrt(2)+half_step, 2*n_fractals)

def is_mixed(gp):
    """Decision if a gamble pair is composed of two mixed gambles."""
    return np.product(gp[0]) < 0 and np.product(gp[1]) < 0

def is_nobrainer(gp):
    """Decision if a gamble pair is nobrainer."""
    return len(set(gp[0]).intersection(set(gp[1]))) != 0

def is_equal_growth(gp):
    """Decision if a gamble pair is composed of two gambles with equal average 
    growth rates."""
    return len(np.unique(np.mean(gp, axis=1))) == 1

def is_g_win(g):
    """Decision if gamble is not-loosing, i.e., composed of win/null fractals.

    Args:
        g (np.array):
            Gamble array of shape (2, 0).

    Returns:
        Boolean decision value.
    """
    return np.all(g >= 0) and not np.all(g == 0)

def is_g_loss(g):
    """Decision if gamble is not-winning, i.e., composed of loss/null fractals.

    Args:
        g (np.array):
            Gamble array of shape (2, 0).

    Returns:
        Boolean decision value.
    """
    return np.all(g <= 0) and not np.all(g == 0)

def is_g_mixed(g):
    """Decision if gamble is mixed, i.e., composed of both win and loss.

    Args:
        g (np.array):
            Gamble array of shape (2, 0).

    Returns:
        Boolean decision value.
    """
    return np.any(g < 0) and np.any(g > 0)

def disagreement(p1, p2):
    """Disagreement between agents.
    
    Args:
        p1 (array; n_trials x n_simulations):
            Choice / choice probability array for agent 1. 
        p2 (array):
            Choice / choice probability array for agent 2.
            
    Returns:
        Disagreement measure. If inputs are softmax choice probability, output
        interpetation is average distance in choice probability. If inputs are
        choices, output interpretation is probability of choosing different 
        gamble pairs.
    """
    return np.nansum(np.abs(p1 - p2)) / np.prod(p1.shape)

def bankruptcy_chance(x):
    """Proportion of simulations that went bankrupt.
    
    Args:
        x (array; n_trials x n_simulations):
            Wealth trajectories array.
            
    Returns:
        Proportion of trajectories that ended up with 0 wealth.
    """
    return np.mean(x[-1] == 0)

def richness_chance(x, x_limit):
    """Proportion of simulations that exceeded wealth limit.
    
    Args:
        x (array; n_trials x n_simulations):
            Wealth trajectories array.
            
    Returns:
        Proportion of trajectories that ended up with wealth higher or equal to 
        x_limit.
    """
    return np.mean(x[-1] >= x_limit)

def experiment_duration(x, x_limit):
    """Average experiment duration.
    
    Args:
        x (array; n_trials x n_simulations):
            Wealth trajectories array.
        x_limit (float):
            Wealth upper bound.
    
    Returns:
        Average number of trials before bankruptcy or upper bound exceedance.
    """
    return np.mean((x != 0) & (x != x_limit)) * x.shape[0] - 1

def payout_stats(x):
    payouts_all = x[-1]
    payouts_act = x[-1][x[-1] > 0]
    return {
        p_name: {
            "mean": np.mean(p),
            "median": np.median(p),
            "min": np.min(p),
            "max": np.max(p), 
            "std": np.std(p),
        } 
        for p_name, p 
        in zip(["all", "active"], [payouts_all, payouts_act])
    }

def run_simulation(experiment, trial_order, eta_dynamic, eta_agent, x_0, 
                   x_limit, beta, wealth_dependency=False):
    """Simulate wealth trajectories for gamble experiment. 
    
    Args:
        experiment:
            It has shape (2, 2, n_trials).
        trial_order:
            It has shape (n_trials, n_simulations).
        eta_dynamic (float):
            Risk-aversion for time-optimal gent in a given dynamic.
        eta_agent (float):
            Risk-aversion of the agent.
        x0 (float):
            Initial wealth.
        x_limit (float):
            Wealth upper-bound.
        beta (float):
            Softmax sensitivity for isoelastic agent.
        wealth_dependency (bool):
            Decision whether current wealth should affect agent's decisions. If 
            False, initial wealth is used to calculate all utilities.
            
    Returns:
        3-tuple of wealth, choice probability and choice arrays. Choice and 
        choice probability arrays are of size n_trials x n_simulations, whereas
        wealth proability array is of size n_trials + 1 x n_simulations. 
    """
    # initialize wealth array x, choice probability array p, and choice array y
    n_trials, n_simulations = trial_order.shape
    p = np.full((n_trials, n_simulations), fill_value=np.nan)
    y = np.full((n_trials, n_simulations), fill_value=np.nan)
    x = np.zeros((n_trials + 1, n_simulations), dtype=np.float64)
    x[0] = x_0

    for t in range(n_trials):
        # filter active simulations i.e, agent is not broke 
        is_broke = x[t] <= 0
        is_rich = x[t] >= x_limit
        active = ~is_broke & ~is_rich
        n_active = np.sum(active)

        # draw gamble pair
        gp = experiment[:, :, trial_order[t, active]]

        # compute expected utility difference
        t_eff = t if wealth_dependency else 0 
        u = isoelastic_utility(x[t_eff, active], eta_agent)
        u_gp = isoelastic_utility(wealth_change(x[t_eff, active], gp, eta_dynamic), eta_agent)
        du_gp = u - u_gp
        gp_diff = np.mean(du_gp[0], axis=0) - np.mean(du_gp[1], axis=0)

        # compute agents choice (coded 0, 1 for left, right)
        p[t, active] = softmax(gp_diff, beta)
        y[t, active] = p[t, active] > np.random.random(size=n_active)

        # update wealth
        coin_toss = np.random.randint(0, 2, size=n_active)
        chosen_gamble = gp[y[t, active].astype(int), :, np.arange(n_active)]
        realized_gamma = chosen_gamble[np.arange(n_active), coin_toss]
        x[t + 1, active] = wealth_change(x[t, active], realized_gamma, eta_dynamic)
        x[t + 1, is_rich] = x[t, is_rich]

    # trim excessive wealth
    x[x > x_limit] = x_limit
        
    return x, p, y
    
def softmax(x, beta):
    """Softmax choice function.
    
    Args:
        beta (float):
            Precision parameter (inverse temperature)
    
    Returns:
        Choice probability.
    """
    return np.power(1 + np.exp((-beta) * x), -1)  

def calculate_beta(eta_dynamic, eta_agent, c, x_0, x_limit, p_threshold):
    """Estimate normalized precision parameter.
    
    Here, a softmax precision (or inverse temperature) for isoelastic agent is 
    estimated. It is defined at precision in which an agent endowed with initial 
    wealth x0 and facing two extreme, opposite fractals with growth rates -c 
    and c would choose the winning fractal with probability p_threshold. 
    Wealth changes that exceed [0, x_limit] range are trimmed down before 
    calculation.
    
    Args:
        eta_dynamic (float):
            Wealth dynamic parameter.
        eta_agent (float):
            Isoelastic agent's risk attitude.
        c (float):
            Maximal fractal growth rate.
        x_0 (float):
            Initial wealth.
        x_limit (float):
            Wealth upper-bound.
        p_threshold (float):
            Matched probability of choosing better fractal. Should lie within 
            range (0, 1). 
        
    Returns:
        Precision parameter beta (float).
    """
    # extreme wealth changes 
    x_f = wealth_change(np.array([x_0, x_0]), np.array([-c, c]), eta_dynamic)
    x_f[x_f < 0] = 0
    x_f[x_f > x_limit] = x_limit

    # utility changes
    u_f = isoelastic_utility(x_f, eta_agent)
    du = u_f[1] - u_f[0]

    beta = np.log(p_threshold / (1 - p_threshold)) / du
    return beta

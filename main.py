import numpy as np
from numpy.random import uniform, normal, multivariate_normal, exponential, gamma
import scipy as sc
from scipy import stats
from statsmodels.tsa.stattools import acf
import pandas as pd
import matplotlib.pyplot as plt

#metropolis-hastings functions
def sample(data: np.ndarray, N: int, B: int, start_theta: tuple, search_breadth: float=0.5, adaptive: bool=False):
    """Takes N samples via the Metropolis-Hastings algorithm, with B burn-in samples."""
    theta, Y = start_theta, []
    for b in range(B): #burnin samples
        if adaptive:
            results = adaptive_step(data, theta, search_breadth, Y)
            theta, Y = results['theta'], results['Y']
        else:
            results = step(data, theta, search_breadth)
            theta = results['theta']
    
    samples = []
    for n in range(N): #real samples
        if adaptive:
            results = adaptive_step(data, theta, search_breadth, Y)
            theta, Y = results['theta'], results['Y']
        else:
            results = step(data, theta, search_breadth)
            theta = results['theta']
        samples.append(results)
    return samples

def step(data: np.ndarray, theta: tuple, search_breadth: float):
    """Takes one step in the Metropolis-Hastings algorithm by generating a new theta and comparing to a given theta."""
    theta_prime = sample_theta(theta, search_breadth) #sample a new set of parameters
    acceptance_log_prob = calc_acceptance_prob(theta, theta_prime, data, search_breadth) #calculate the probability of acceptance
    acceptance_prob = min(1,np.exp(acceptance_log_prob))
    accepted = acceptance_prob >= uniform() #probabilistically determine acceptance
    return {'accepted': accepted, 'acceptance_prob': acceptance_prob, 'theta': theta_prime if accepted else theta} #return results, update theta if samples accepted

def sample_theta(theta: tuple, search_breadth: float):
    """Samples theta parameters--slope, intercept, and standard deviation."""
    a,b,sigma = theta
    a,b = multivariate_normal([a,b], [[search_breadth**2,0],[0,search_breadth**2]])
    sigma = gamma(sigma*search_breadth*500, 1/(search_breadth*500))
    theta = a,b,sigma
    return theta

def calc_acceptance_prob(theta: tuple, theta_prime: tuple, data: np.ndarray, search_breadth: float):
    """Calculates acceptance log-probability by using a Bayesian linear model. Note: all terms are in log-values here,
    so must be exponentiated before determining acceptance against u."""    
    theta_likelihood = likelihood(theta, data)
    theta_prior = prior(theta)
    
    theta_p_likelihood = likelihood(theta_prime, data)
    theta_p_prior = prior(theta_prime)
    
    pr = proposal_ratio(theta, theta_prime, search_breadth)
    acceptance_ratio = theta_p_likelihood - theta_likelihood + theta_p_prior - theta_prior + pr
    return acceptance_ratio

#bayesian functions
def likelihood(theta: tuple, data: np.ndarray):
    """Calculates the likelihood component of our linear model by measuring our parameters theta on the given data."""
    a,b,sigma = theta
    x,y = data[0],data[1]
    likelihoods = sc.stats.norm.logpdf(y, loc=a*x+b, scale=sigma) #find the likelihood of a sample given a normal distribution specified by our parameters and the data
    return np.sum(likelihoods) #use log likelihood for stability

def prior(theta: tuple):
    """Calculates the prior component of our linear model, specified """
    a,b,sigma = theta
    ab_prior = sc.stats.multivariate_normal.logpdf([a,b], [0,0], [[100,0],[0,100]]) #cov defaults to 1
    sigma_prob = sc.stats.gamma.logpdf(sigma, 1, 1)
    return np.sum([ab_prior,sigma_prob])

def proposal_ratio(theta: tuple, theta_prime: tuple, search_breadth: float):
    """Offsets bidirectionality of chained samples."""
    a,b,sigma = theta
    a_p,b_p,sigma_p = theta_prime
    old_given_new_ab = sc.stats.multivariate_normal.logpdf([a,b],[a_p,b_p],[[search_breadth**2,0],[0,search_breadth**2]])
    old_given_new_sigma = sc.stats.gamma.logpdf(sigma, sigma_p*search_breadth*500, scale=1/(500*search_breadth))
    old_given_new = old_given_new_ab + old_given_new_sigma
    
    new_given_old_ab = sc.stats.multivariate_normal.logpdf([a_p,b_p],[a,b],[[search_breadth**2,0],[0,search_breadth**2]])
    new_given_old_sigma = sc.stats.gamma.logpdf(sigma_p, sigma*search_breadth*500, scale=1/(500*search_breadth))
    new_given_old = new_given_old_ab - new_given_old_sigma

    return old_given_new - new_given_old

#viz functions
def plot_convergence(df: pd.DataFrame):
    """Shows convergence of theta for a given results dataframe."""
    plt.plot(range(len(df)),df['a'], label='slope')
    plt.plot(range(len(df)),df['b'], label='intercept')
    plt.plot(range(len(df)),df['sigma'], label='variance')
    plt.plot(range(len(df)),df['acceptance_prob'], label='accept_prob')
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.legend()
    
#highest level functions to coordinate runs
def compare_convergences(starting_positions: list, data: np.ndarray, n_samples: int=20000, adaptive: bool=False, out_dir: str=''):
    """High-level function that coordinates many rounds of the Metropolis-Hastings algorithm,
    each starting at a different position."""
    dfs = []
    for start in starting_positions:
        print(f"working on {start}")
        samples = sample(data, n_samples, 0, start, 0.1, adaptive) #no burn-in, breadth of search should be 0.1 -- yields a 5% acceptance rate at convergence

        #aggregate this run's results into dataframe
        df = pd.DataFrame(samples)
        df[['a','b','sigma']] = pd.DataFrame(df['theta'].tolist(), index=df.index)
        df['start'] = str(start) #if this breaks just use an index and map
        df = df.drop(['theta'], axis=1)
        dfs.append(df) #append to the larger results df
        
    df = pd.concat(dfs)
    
    #estimation of parameter over time plots
    for param in ['a','b','sigma']:
        for start_val in df['start'].unique(): #there's definitely a more groupby-y way to do this but this will work!
            param_from_start = df[df['start']==start_val][param]
            plt.plot(range(len(param_from_start)), param_from_start, label=start_val)
        plt.legend()
        plt.xlabel('Time Step')
        plt.ylabel(param)
        plt.title(f"Evolution of {param.upper()} Parameter Estimate from Different Starts")
        plt.savefig(f'{out_dir}{param}_estimate_evolution_{"adaptive" if adaptive else "nonadaptive"}.jpg', bbox_inches='tight')
        plt.show()
        plt.clf()
        
    #autocorrelation plots (I use the ACF metric from statsmodels, see more here: https://github.com/statsmodels/statsmodels/blob/c22837f0632ae8890f56886460c429ebf356bd9b/statsmodels/tsa/stattools.py#L579)
    lag = n_samples/10
    for param in ['a','b','sigma']:
        for start_val in df['start'].unique(): #there's definitely a more groupby-y way to do this but this will work!
            param_from_start = acf(df[df['start']==start_val][param],nlags=lag)
            plt.plot(range(len(param_from_start)), param_from_start, label=start_val)
        plt.legend()
        plt.xlabel('Time Step')
        plt.ylabel(f'Autocorrelation with Lag {lag}')
        plt.title(f"Autocorrelation for {param.upper()} Parameter Estimate from Different Starts")
        plt.savefig(f'{out_dir}{param}_autocorrelation_{"adaptive" if adaptive else "nonadaptive"}.jpg', bbox_inches='tight')
        plt.show()
        plt.clf()
        
    df.to_csv(f'{out_dir}results.csv')
    return df

#adaptive metropolis-hastings functions (Python NEEDS multiple dispatch!)
def adaptive_step(data: np.ndarray, theta: tuple, search_breadth: float, Y: list):
    """Takes one step in the Metropolis-Hastings algorithm by generating a new theta and comparing to a given theta."""
    theta_prime = adaptive_sample_theta(theta, search_breadth, Y) #sample a new set of parameters
    acceptance_log_prob = adaptive_calc_acceptance_prob(theta, theta_prime, data, search_breadth, Y) #calculate the probability of acceptance
    acceptance_prob = min(1,np.exp(acceptance_log_prob))
    accepted = acceptance_prob >= uniform() #probabilistically determine acceptance
    if accepted:
        theta = theta_prime
        Y.append(theta_prime)
    return {'accepted': accepted, 
            'acceptance_prob': acceptance_prob,
            'theta': theta,
            'Y': Y}

def adaptive_sample_theta(theta: tuple, search_breadth: float, Y: list):
    """Samples theta parameters--slope, intercept, and standard deviation."""
    a,b,sigma = theta
    a,b = multivariate_normal([a,b], [[search_breadth**2,0],[0,search_breadth**2]])
    sigma = gamma(sigma*search_breadth*500, 1/(search_breadth*500))
    theta = a,b,sigma
    return theta

def adaptive_calc_acceptance_prob(theta: tuple, theta_prime: tuple, data: np.ndarray, search_breadth: float, Y: list):
    """Calculates acceptance log-probability by using a Bayesian linear model. Note: all terms are in log-values here,
    so must be exponentiated before determining acceptance against u."""
    avg_accepted_theta = (0,0,1)
    if len(Y) > 0:
        accepted_a,accepted_b,accepted_sigma = zip(*Y)
        avg_accepted_theta = (np.mean(accepted_a),np.mean(accepted_b),np.mean(accepted_sigma))
    
    theta_likelihood = likelihood(theta, data)
    theta_prior = prior(avg_accepted_theta) if len(Y) > 0 else prior(theta)
    
    theta_p_likelihood = likelihood(theta_prime, data)
    theta_p_prior = prior(theta_prime)
    
    pr = proposal_ratio(theta, theta_prime, search_breadth)
    acceptance_ratio = theta_p_likelihood - theta_likelihood + theta_p_prior - theta_prior + pr
    return acceptance_ratio
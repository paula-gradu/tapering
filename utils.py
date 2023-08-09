import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42})
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18 
import matplotlib.colors as mcolors

''' I. Opponent Process Class '''

class OP:
    ''' A simple system class that takes in an impulse response specification and
    tracks the history dose in order to output the current observations.'''
    
    def __init__(self, g=None, lin_sys=None):
        if g is not None:
            self.g = g.copy()
        else:
            A, b, c, T = lin_sys
            self.g = [np.dot(c, b)] + [np.dot(c, np.power(A, p) @ b) for p in range(1, T)]
        self.dose_history = []
        
    def reset(self):
        self.dose_history = []
    
    def step(self, dose, disturbance=0):
        self.dose_history.append(dose)
        np_dose_history = np.array(self.dose_history, copy=True)
        return np.dot(self.g[:len(self.dose_history)], np.flip(np_dose_history)) + disturbance
    
    def get_obs(self, disturbance=0):
        np_dose_history = np.array(self.dose_history, copy=True)
        return np.dot(self.g[:len(self.dose_history)], np.flip(np_dose_history)) + disturbance
    
    def initialize_addiction(self, dose_history):
        self.dose_history = dose_history.copy()
        
    def initialize_and_plot_addiction(self, dose_history, ax=None, \
                                      label="Allostasis Progression", color="red", linestyle='solid'):
        obs_hist = []
        self.reset()
        for t in range(len(dose_history)):
            self.dose_history.append(dose_history[t])
            np_dose_history = np.array(self.dose_history, copy=True)
            obs_hist.append(np.dot(self.g[:len(self.dose_history)], np.flip(np_dose_history)))
        if ax is None:
            plt.plot(obs_hist, color=color, label=label, linewidth=2, linestyle=linestyle)
        else:
            ax.plot(obs_hist, color=color, label=label, linewidth=2, linestyle=linestyle)
  
    def plot_impulse(self, T=30, color="blue", label="Impulse Response", linestyle="-", ax=None):
        if ax is None:
            plt.plot(self.g[:T], color=color, label=label, linestyle=linestyle, linewidth=2)
        else:
            ax.plot(self.g[:T], color=color, label=label, linestyle=linestyle, linewidth=2)
            
    def plot_blunted_impulse(self, T=30, color="purple", label="Blunted Impulse Response", linestyle="-", ax=None):
        obs_hist = []
        self.dose_history.append(1)
        for t in range(T):
            np_dose_history = np.array(self.dose_history, copy=True)
            obs_hist.append(np.dot(self.g[:len(self.dose_history)], np.flip(np_dose_history)))
            self.dose_history.append(0)
        if ax is None:
            plt.plot(obs_hist, color=color, label="Blunted Impulse Response", linestyle='--', linewidth=2)
        else:
            ax.plot(obs_hist, color=color, label="Blunted Impulse Response", linestyle='--', linewidth=2)
            
            
''' II. Specification of Protocol Classes '''

def cold_turkey(obs, disturbance=0):
    return 0

class bang_bang:
    def __init__(self, init_dose, obs_min):
        self.obs_min = obs_min
        self.init_dose = init_dose
    
    def __call__(self, obs, disturbance=0):
        return self.init_dose if obs < self.obs_min else 0
    
class linear:
    def __init__(self, init_dose, rate):
        self.init_dose = init_dose
        self.last_dose = init_dose
        self.rate = rate
        
    def __call__(self, obs, disturbance=0):
        self.last_dose = max(0, self.last_dose - self.rate * self.init_dose)
        return self.last_dose
    
    def reset(self):
        self.last_dose = self.init_dose
        
class exp:
    def __init__(self, init_dose, rate):
        self.init_dose = init_dose
        self.last_dose = init_dose
        self.rate = rate
        
    def __call__(self, obs, disturbance = 0):
        self.last_dose = self.rate * self.last_dose
        return self.last_dose
    
    def reset(self):
        self.last_dose = self.init_dose
        
class integral:
    def __init__(self, init_dose, obs_min, delta, g0_range, monotone=False):
        self.delta = delta
        self.obs_min = obs_min
        self.init_dose = init_dose
        self.last_dose = init_dose
        self.g0_lb, self.g0_ub = g0_range
        self.monotone = monotone
        
    def __call__(self, obs, disturbance = 0):
        dose_dec = (obs - self.obs_min - self.delta)/(self.g0_ub) if obs > self.obs_min else (obs - self.obs_min)/(self.g0_lb)
        dose = max(0, self.last_dose - dose_dec)
        self.last_dose = dose if not self.monotone else min(self.last_dose, dose) 
        
        return self.last_dose
    
    def reset(self):
        self.last_dose = self.init_dose
        
class MED:
    def __init__(self, obs_min, g, ynat):
        self.obs_min = obs_min
        self.g = g
        self.ynat = ynat
        self.dose_history = []
        
    def __call__(self, obs, disturbance = 0):
        dose = max(0, (self.obs_min - self.ynat[len(self.dose_history)] - \
                       np.dot(self.g[1:len(self.dose_history)+1], np.flip(self.dose_history)))/self.g[0])
        self.dose_history.append(dose)
        return dose


''' III. Noise Utils '''

def zero():
    return 0

class uniform:
    def __init__(self, low, high):
        self.low = low
        self.high = high
    def __call__(self):
        return np.random.uniform(self.low, self.high)
    
class uniform_poisson:
    def __init__(self, low, high, lam):
        self.low = low
        self.high = high
        self.lam = lam
    def __call__(self):
        return np.random.poisson(self.lam) * np.random.uniform(self.low, self.high)
    
''' IV. Testing Utils '''

def test(OP, protocol, T, noise_generator=zero, plot=False):
    obs_hist = [OP.get_obs()]
    dose_hist = []
    disturbance = 0
    for t in range(T):
        dose = protocol(obs_hist[-1], disturbance)
        disturbance = noise_generator()
        obs_hist.append(OP.step(dose, disturbance))
        dose_hist.append(dose)
        
    if plot:
        plt.plot(obs_hist[1:T], label="obs")
        plt.plot(dose_hist, label = "dose")
        plt.legend()
        try:
            plt.title(protocol.__name__ + " withdrawal");
        except:
            plt.title(protocol.__class__.__name__ + " withdrawal");
            
    return obs_hist[1:T], dose_hist

def test_pop(trials, system, noise_generator, obs_min_generator, protocol, taper_T):
    violation = []
    dose = []
    finished = []
    
    OP, addict_dose_history = system
    
    for trial in range(trials):
        
        try: # need to reset protocols per each rial
            protocol.reset()
        except: # no need to reset cold turkey
            pass
        
        np.random.seed(trial) # generate same obs_min
        obs_min = obs_min_generator()
        
        OP.initialize_addiction(addict_dose_history) # initialize addiction
        
        try: # lookahead & MED take true obs_min
            protocol.obs_min = obs_min
        except: # others don't
            pass
        
        obs, doses = test(OP, protocol, taper_T, noise_generator=noise_generator) # get individual outcomes

        # extract experiment metrics
        violation.append(np.sum(np.maximum((obs_min * np.ones(taper_T-1) - obs), 0)) / taper_T)
        dose.append(np.sum(np.array(doses)) / taper_T)
        finished.append(doses[-1] < 1e-2)
    
    return np.mean(violation), np.mean(dose), np.mean(finished)

def get_MED(trials, system, noise_generator, obs_min_generator, taper_T):
    violation = []
    dose = []
    finished = []
    
    OP, addict_dose_history = system
    
    for trial in range(trials):
        
        np.random.seed(trial) # generate same obs_min
        obs_min = obs_min_generator()
        
        OP.initialize_addiction(addict_dose_history) # initialize addiction for CT
        ct_obs, _ = test(OP, cold_turkey, taper_T+1) # get ynat via CT
        
        OP.initialize_addiction(addict_dose_history) # re-initialize addiction
        obs, doses = test(OP, MED(obs_min, OP.g, ct_obs), taper_T) # get individual outcomes

        # extract experiment metrics
        violation.append(np.sum(np.maximum((obs_min * np.ones(taper_T-1) - obs), 0)) / taper_T)
        dose.append(np.sum(np.array(doses)) / taper_T)
        finished.append(doses[-1] < 1e-2)
    
    return np.mean(violation), np.mean(dose), np.mean(finished)

'''V. Plotting Utils'''

def plot_responses(drug, addict_T, addict_dose_history, addict_dose_history_interrupted):
    fig, ax = plt.subplots(nrows=len(drug.keys())//2, ncols=2, figsize=(20, 10))
    for ax_idx, drug_idx in enumerate(drug.keys()):
        drug[drug_idx].plot_impulse(addict_T, ax=ax[ax_idx // 2, ax_idx % 2])
        drug[drug_idx].initialize_and_plot_addiction(addict_dose_history, ax=ax[ax_idx // 2, ax_idx % 2], \
                                                     label="Allostasis Progression (Continuous Dosing)")
        drug[drug_idx].initialize_and_plot_addiction(addict_dose_history_interrupted, ax=ax[ax_idx // 2, ax_idx % 2], \
                                                     label="Allostasis Progression (Interrupted Dosing)", \
                                                     color="orange", linestyle="dashed")
        ax[ax_idx // 2, ax_idx % 2].set_title(f'$g_{drug_idx}$', size=28)
        if ax_idx == 1:
            ax[ax_idx // 2, ax_idx % 2].legend(loc=1, prop={'size': 20})
        if ax_idx % 2 == 0:
            ax[ax_idx // 2, ax_idx % 2].set_ylabel('well-being', size=24) 
        if ax_idx > 1:
            ax[ax_idx // 2, ax_idx % 2].set_xlabel('timestep', size=24)

    plt.tight_layout()

def plot_comparison(x_data, y_data, param_sweep):
    lin_rates, exp_rates, deltas = param_sweep
   
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    ax[1,0].set_xlabel("avg. constraint violation", size=24)
    ax[1,1].set_xlabel("avg. constraint violation", size=24)
    if exp_rates is not None:
        ax[0,0].set_ylabel("avg. total dose", size=24)
        ax[1,0].set_ylabel("avg. total dose", size=24)

    for idx, drug_idx in enumerate(['A', 'B', 'C', 'D']):
        ax_idx = (idx // 2, idx % 2)
        ax[ax_idx].set_title(f'$g_{drug_idx}$', size=28)
        
        # 1. Plot Linear
        if lin_rates is not None:
            ax[ax_idx].plot([x_data[drug_idx]['lin-'+str(lin_rate)] for lin_rate in lin_rates[drug_idx]], \
                            [y_data[drug_idx]['lin-'+str(lin_rate)] for lin_rate in lin_rates[drug_idx]], \
                            c="red", linewidth=2)
            for i, lin_rate in enumerate(lin_rates[drug_idx]):
                if not i:
                    ax[ax_idx].scatter(x_data[drug_idx]['lin-'+str(lin_rate)], \
                                       y_data[drug_idx]['lin-'+str(lin_rate)], c="red", label="linear", linewidth=2)
                    ax[ax_idx].annotate(" " + str(lin_rate), (x_data[drug_idx]['lin-'+str(lin_rate)], \
                                                        y_data[drug_idx]['lin-'+str(lin_rate)]), \
                                                        size=18, color="darkred")
                else:
                    ax[ax_idx].scatter(x_data[drug_idx]['lin-'+str(lin_rate)], y_data[drug_idx]['lin-'+str(lin_rate)], \
                                       c="red", linewidth=2)
                    ax[ax_idx].annotate(" " + str(lin_rate), (x_data[drug_idx]['lin-'+str(lin_rate)], \
                                                        y_data[drug_idx]['lin-'+str(lin_rate)]), \
                                                        size=18, color="darkred")

        # 2. Plot Exponential
        if exp_rates is not None:
            ax[ax_idx].plot([x_data[drug_idx]['exp-'+str(exp_rate)] for exp_rate in exp_rates[drug_idx]], \
                            [y_data[drug_idx]['exp-'+str(exp_rate)] for exp_rate in exp_rates[drug_idx]], \
                            c="orange", linewidth=2)
            for i, exp_rate in enumerate(exp_rates[drug_idx]):
                if not i:
                    ax[ax_idx].scatter(x_data[drug_idx]['exp-'+str(exp_rate)], \
                                       y_data[drug_idx]['exp-'+str(exp_rate)], c="orange", label="exponential", linewidth=2)
                    ax[ax_idx].annotate(" " + str(exp_rate), (x_data[drug_idx]['exp-'+str(exp_rate)], \
                                                        y_data[drug_idx]['exp-'+str(exp_rate)]), \
                                                        size=18, color="darkorange")
                else:
                    ax[ax_idx].scatter(x_data[drug_idx]['exp-'+str(exp_rate)], \
                                       y_data[drug_idx]['exp-'+str(exp_rate)], c="orange", linewidth=2)
                    ax[ax_idx].annotate(" " + str(exp_rate), (x_data[drug_idx]['exp-'+str(exp_rate)], \
                                                        y_data[drug_idx]['exp-'+str(exp_rate)]), \
                                                        size=18, color="darkorange")

        # 3. Plot Lookahead
        if deltas is not None:
            ax[ax_idx].plot([x_data[drug_idx]['I-'+str(delta)] for delta in deltas[drug_idx]], \
                            [y_data[drug_idx]['I-'+str(delta)] for delta in deltas[drug_idx]], \
                             c='blue', linewidth=2)
            for j, delta in enumerate(deltas[drug_idx]):
                if not j:
                    ax[ax_idx].scatter(x_data[drug_idx]['I-'+str(delta)], y_data[drug_idx]['I-'+str(delta)], \
                               c='blue', label="integral", linewidth=2)
                    ax[ax_idx].annotate("  " + str(delta), (x_data[drug_idx]['I-'+str(delta)], \
                                                     y_data[drug_idx]['I-'+str(delta)]), \
                                                     size=18, color="darkblue")
                else:
                    ax[ax_idx].scatter(x_data[drug_idx]['I-'+str(delta)], \
                                       y_data[drug_idx]['I-'+str(delta)], c='blue', linewidth=2)
                    ax[ax_idx].annotate("  " + str(delta), (x_data[drug_idx]['I-'+str(delta)], \
                                                     y_data[drug_idx]['I-'+str(delta)]), \
                                                     size=18, color="darkblue")
            # Star delta = 0
            ax[ax_idx].plot(x_data[drug_idx]['I-0'], y_data[drug_idx]['I-0'], linestyle='None', marker="*", \
                        markersize=18, markerfacecolor="blue")
            
        # 4. Plot MED
        ax[ax_idx].plot(x_data[drug_idx]['MED'], y_data[drug_idx]['MED'], linestyle='None', marker="*", \
                        markersize=22, markerfacecolor="magenta", markeredgecolor="black", label="optimal")

    if exp_rates is not None:
        ax[0, 1].legend(loc=1, prop={'size': 22})
    plt.tight_layout()
    
def plot_ablation(x_data, y_data, param_sweep):
    lin_rates, exp_rates, deltas, g0_ranges = param_sweep
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

    ax[1,0].set_xlabel("avg. constraint violation", size=24)
    ax[1,1].set_xlabel("avg. constraint violation", size=24)
    ax[0,0].set_ylabel("avg. total dose", size=24)
    ax[1,0].set_ylabel("avg. total dose", size=24)

    labels = ['100%, 100%', '50%, 150%', '50%, 300%', '35%, 150%']
    g0_colors = ['green', 'blue', 'orange', 'red']
    
    for idx, drug_idx in enumerate(['A', 'B', 'C', 'D']):
        ax_idx = (idx // 2, idx % 2)
        if idx == 3:
            ax[ax_idx].plot([x_data[drug_idx]['lin-'+str(lin_rate)] for lin_rate in lin_rates[drug_idx]], \
                            [y_data[drug_idx]['lin-'+str(lin_rate)] for lin_rate in lin_rates[drug_idx]], c="black", \
                            label= "baseline", linewidth=2.5)
        else:
            ax[ax_idx].plot([x_data[drug_idx]['exp-'+str(exp_rate)] for exp_rate in exp_rates[drug_idx]], \
                            [y_data[drug_idx]['exp-'+str(exp_rate)] for exp_rate in exp_rates[drug_idx]], c="black", \
                            label="baseline", linewidth=2.5)
        
        for i, g0_range in enumerate(g0_ranges):
            if i == 0 and idx == 3:
                ax[ax_idx].plot([x_data[drug_idx]['I-'+str(g0_range)+'-'+str(delta)] for delta in deltas[drug_idx][2:]], \
                                [y_data[drug_idx]['I-'+str(g0_range)+'-'+str(delta)] for delta in deltas[drug_idx][2:]],
                                c=g0_colors[i], linewidth=2, label=labels[i])
            else:
                ax[ax_idx].plot([x_data[drug_idx]['I-'+str(g0_range)+'-'+str(delta)] for delta in deltas[drug_idx]], \
                                [y_data[drug_idx]['I-'+str(g0_range)+'-'+str(delta)] for delta in deltas[drug_idx]],
                                c=g0_colors[i], linewidth=2, label=labels[i])
            for delta in deltas[drug_idx]:
                if i == 0 and idx == 3 and delta < 0:
                    continue # do not include negative delta for perfect g(0) for D
                ax[ax_idx].scatter(x_data[drug_idx]['I-'+str(g0_range)+'-'+str(delta)], \
                                   y_data[drug_idx]['I-'+str(g0_range)+'-'+str(delta)], \
                                   c=g0_colors[i], linewidth=1, alpha=0.5)
        # Star delta = 0
        for i, g0_range in enumerate(g0_ranges):
            ax[ax_idx].plot(x_data[drug_idx]['I-'+str(g0_range)+'-0'], y_data[drug_idx]['I-'+str(g0_range)+'-0'],
                            linestyle='None', marker="*", markersize=20, markerfacecolor=g0_colors[i], alpha=0.75)

        ax[ax_idx].set_title(f'$g_{drug_idx}$', size=28)
        
        # Plot MED
        ax[ax_idx].plot(x_data[drug_idx]['MED'], y_data[drug_idx]['MED'], linestyle='None', marker="*", \
                        markersize=22, markerfacecolor="magenta", markeredgecolor="black")

    ax[0,1].legend(prop={'size': 22})
    plt.tight_layout()
    
def plot_ablation_ext(x_data, y_data, param_sweep):
    lin_rates, exp_rates, g0_ranges = param_sweep
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

    ax[1,0].set_xlabel("avg. constraint violation", size=24)
    ax[1,1].set_xlabel("avg. constraint violation", size=24)
    ax[0,0].set_ylabel("avg. total dose", size=24)
    ax[1,0].set_ylabel("avg. total dose", size=24)

    labels = ['100%, 100%', '100%, 300%', '75%, 125%', '50%, 150%', '50%, 300%', '50%, 400%', \
              '35%, 100%', '35%, 150%', '35%, 300%', '25%, 400%']
    g0_colors = ['xkcd:green', 'xkcd:teal', 'xkcd:aqua', 'xkcd:blue', 'xkcd:orange', 'xkcd:mauve', \
                 'xkcd:magenta', 'xkcd:red', 'xkcd:crimson', 'xkcd:barney']
    
    for idx, drug_idx in enumerate(['A', 'B', 'C', 'D']):
        ax_idx = (idx // 2, idx % 2)
        if idx == 3:
            ax[ax_idx].plot([x_data[drug_idx]['lin-'+str(lin_rate)] for lin_rate in lin_rates[drug_idx]], \
                            [y_data[drug_idx]['lin-'+str(lin_rate)] for lin_rate in lin_rates[drug_idx]], c="black", \
                            label= "baseline", linewidth=2.5)
        else:
            ax[ax_idx].plot([x_data[drug_idx]['exp-'+str(exp_rate)] for exp_rate in exp_rates[drug_idx]], \
                            [y_data[drug_idx]['exp-'+str(exp_rate)] for exp_rate in exp_rates[drug_idx]], c="black", \
                            label="baseline", linewidth=2.5)
        
        for i, g0_range in enumerate(g0_ranges):
            ax[ax_idx].plot(x_data[drug_idx]['I-'+str(g0_range)], y_data[drug_idx]['I-'+str(g0_range)], linestyle='None', \
                            marker=".", markersize=20, markerfacecolor=g0_colors[i], label=labels[i])
            ax[ax_idx].annotate("   " + labels[i], (x_data[drug_idx]['I-'+str(g0_range)], \
                                                     y_data[drug_idx]['I-'+str(g0_range)]), \
                                                     size=8, color=g0_colors[i], alpha=0.75)

        ax[ax_idx].set_title(f'$g_{drug_idx}$', size=28)
        
        # Plot MED
        ax[ax_idx].plot(x_data[drug_idx]['MED'], y_data[drug_idx]['MED'], linestyle='None', marker="*", \
                        markersize=22, markerfacecolor="magenta", markeredgecolor="black")

    plt.tight_layout()
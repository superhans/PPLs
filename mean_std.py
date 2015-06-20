import pymc
import numpy as np

weight_arr = np.asarray((69,42,63,81,44,39,71,62,33));
n = weight_arr.size;
mean =  1.0*sum(weight_arr)/n;
stddev = np.sqrt(sum((weight_arr - mean)**2)/(n-1));

print 'Using formula : Mean = ',mean,'Std dev = ',stddev;

# now, define priors

mean_mcmc = pymc.Uniform('mean_mcmc',lower = np.min(weight_arr), upper = np.max(weight_arr));
precision = pymc.Uniform('precision',lower = 0.0001, upper = 1);
weights = pymc.Normal('weights',mu=mean_mcmc,tau=precision, value=weight_arr, observed=True);

# now, use MCMC sampling
model = pymc.MCMC(weights);
model.sample(iter=1500);
print(model.stats())



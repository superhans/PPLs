import mean_std
import pymc
from pymc.Matplot import plot
from pylab import show

# now, use MCMC sampling
model = pymc.MCMC(mean_std);
model.sample(iter=10000);
print(model.stats())
plot(model)
show()


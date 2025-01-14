import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint, root_scalar
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynLaborFertModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 10 # time periods
        
        # preferences
        par.rho = 1/1.02 # discount factor

        par.beta_0 = 0.1 # weight on labor dis-utility (constant)
        par.beta_1 = 0.05 # additional weight on labor dis-utility (children)
        par.eta = -2.0 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.3 # human capital accumulation 
        par.w = 1.0 # wage base level
        par.tau = 0.1 # labor income tax

        # children
        par.p_birth = 0.1
        par.c_cost = 0

        # spouse
        par.y0 = 0. # constant
        par.y1 = 0. # slope
        par.spousegrid = np.array([0,1])
        par.p_spouse = 1

        # saving
        par.r = 0.02 # interest rate

        # grids
        par.a_max = 10.0 # maximum point in wealth grid
        par.a_min = -10.0 # minimum point in wealth grid
        par.Na = 100 #70 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 30 #30 # number of grid points in wealth grid    

        par.Nn = 2 # number of children

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals

        # estimate beta_1
        par.target_drop_y0 = -0.1


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. asset grid
        par.a_grid = nonlinspace(par.a_min,par.a_max,par.Na,1.1)

        # b. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # c. number of children grid
        par.n_grid = np.arange(par.Nn)

        # d. solution arrays
        shape = (par.T,2,par.Nn,par.Na,par.Nk) #2 is spouse, no spouse
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)
        sol.solved = False # keep track of whether the model has been solved

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_)
        
        # f. draws used to simulate child arrival
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape)

        #... and spouse arrival
        sim.spouse  = np.random.choice([0,1], p=(1-par.p_spouse, par.p_spouse), size=shape)

        # g. initialization
        sim.a_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)
        sim.spouse_init = np.random.choice([0,1], p=(1-par.p_spouse, par.p_spouse), size=par.simN) # initiate random spouse

        # h. vector of wages. Used for simulating elasticities
        par.w_vec = par.w * np.ones(par.T)


    ############
    # Solution #
    def solve(self, do_print=False):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        
        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):
            if do_print:
                print(f'solving period {t}...')
            for i_s, spouse in enumerate(par.spousegrid):
                if spouse == 0 and par.p_spouse == 1:
                    pass # no need to evaluate if there is always a spouse
                else:
                    # i. loop over state variables: number of children, human capital and wealth in beginning of period
                    for i_n,kids in enumerate(par.n_grid):
                        for i_a,assets in enumerate(par.a_grid):
                            for i_k,capital in enumerate(par.k_grid):
                                idx = (t,i_s,i_n,i_a,i_k)

                                # ii. find optimal consumption and hours at this level of wealth in this period t.

                                if t==par.T-1: # last period
                                    obj = lambda x: self.obj_last(x[0],assets,capital,spouse,kids)

                                    constr = lambda x: self.cons_last(x[0],assets,capital,spouse,kids)
                                    nlc = NonlinearConstraint(constr, lb=0.0, ub=np.inf,keep_feasible=False)

                                    # call optimizer
                                    hours_min = - assets / self.wage_func(capital,t) + 1.0e-5 # minimum amout of hours that ensures positive consumption
                                    hours_min = np.maximum(hours_min,2.0)
                                    init_h = np.array([hours_min]) if i_a==0 else np.array([sol.h[t,i_s,i_n,i_a-1,i_k]]) # initial guess on optimal hours

                                    res = minimize(obj,init_h,bounds=((0.0,np.inf),),constraints=nlc,method='trust-constr')

                                    # store results
                                    sol.c[idx] = self.cons_last(res.x[0],assets,capital,spouse,kids)
                                    sol.h[idx] = res.x[0]
                                    sol.V[idx] = -res.fun

                                else:
                                    
                                    # objective function: negative since we minimize
                                    obj = lambda x: - self.value_of_choice(x[0],x[1],assets,capital,spouse,kids,t)  

                                    # bounds on consumption 
                                    lb_c = 0.000001 # avoid dividing with zero
                                    ub_c = np.inf

                                    # bounds on hours
                                    lb_h = 0.0
                                    ub_h = np.inf 

                                    bounds = ((lb_c,ub_c),(lb_h,ub_h))
                        
                                    # call optimizer
                                    init = np.array([lb_c,1.0]) if (i_n == 0 & i_a==0 & i_k==0) else res.x  # initial guess on optimal consumption and hours
                                    res = minimize(obj,init,bounds=bounds,method='L-BFGS-B') 
                                
                                    # store results
                                    sol.c[idx] = res.x[0]
                                    sol.h[idx] = res.x[1]
                                    sol.V[idx] = -res.fun
        sol.solved = True
        if do_print:
            print('Model solved\n')

    # last period
    def cons_last(self,hours,assets,capital,spouse,kids):
        par = self.par

        income = self.wage_func(capital,par.T-1) * hours
        spouse_income = (par.y0 + par.y1*(par.T-1))*spouse
        childcare = kids*par.c_cost
        cons = assets + income + spouse_income - childcare
        return cons

    def obj_last(self,hours,assets,capital,spouse,kids):
        cons = self.cons_last(hours,assets,capital,spouse,kids)
        return - self.util(cons,hours,kids)    

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,spouse,kids,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. penalty for violating bounds. 
        penalty = 0.0
        if cons < 0.0:
            penalty += cons*1_000.0
            cons = 1.0e-5
        if hours < 0.0:
            penalty += hours*1_000.0
            hours = 0.0

        # c. utility from consumption
        util = self.util(cons,hours,kids)
        
        # d. *expected* continuation value from savings
        income = self.wage_func(capital,t) * hours
        spouse_income = (par.y0 + par.y1*t)*spouse
        childcare = kids*par.c_cost
        a_next = (1.0+par.r)*(assets + income + spouse_income - childcare - cons)
        k_next = capital + hours


        # spouse
        # no birth
        kids_next = kids
        V_next = sol.V[t+1,1,kids_next]
        V_next_no_birth = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        ## birth
        if (kids>=(par.Nn-1)):
            # cannot have more children
            V_next_birth = V_next_no_birth

        else:
            kids_next = kids + 1
            V_next = sol.V[t+1,1,kids_next]
            V_next_birth = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        EV_next_spouse = par.p_birth * V_next_birth + (1-par.p_birth)*V_next_no_birth

        # no spouse
        kids_next = kids
        V_next = sol.V[t+1,0,kids_next]
        V_next_no_spouse = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next) if par.p_spouse !=1 else 0 # will be nan if p_spouse == 1 bc value func  has not been computed

        # Total
        EV_next = par.p_spouse*EV_next_spouse + (1-par.p_spouse)*V_next_no_spouse

        # e. return value of choice (including penalty)
        return util + par.rho*EV_next + penalty


    def util(self,c,hours,kids):
        par = self.par

        beta = par.beta_0 + par.beta_1*kids

        return (c)**(1.0+par.eta) / (1.0+par.eta) - beta*(hours)**(1.0+par.gamma) / (1.0+par.gamma) 

    def wage_func(self,capital,t):
        # after tax wage rate
        par = self.par

        return (1.0 - par.tau )* par.w_vec[t] * (1.0 + par.alpha * capital)

    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.n[i,0] = sim.n_init[i]
            sim.a[i,0] = sim.a_init[i]
            sim.k[i,0] = sim.k_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.spouse[i,t],sim.n[i,t])
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[idx_sol],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[idx_sol],sim.a[i,t],sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    income = self.wage_func(sim.k[i,t],t)*sim.h[i,t]
                    spouse_income = (par.y0 + par.y1*t)*sim.spouse[i,t]
                    childcare = sim.n[i,t]*par.c_cost
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income + spouse_income - childcare - sim.c[i,t])
                    sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]

                    spouse_next = sim.spouse[i,t+1]
                    birth = 0 
                    if spouse_next == 1:
                        if ((sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1))):
                            birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth
                    
    
    def simulate_t(self, tt, shock_model):
        """ Simulate from period tt and onwards using policy functions from shock_model
        - used to compute age specific Marshall elasticities """

        sol = shock_model.sol
        par = shock_model.par

        sim = self.sim

        # b. loop over individuals and time
        for i in range(self.par.simN):

            # i. initialize states
            #sim.n[i,0] = sim.n_init[i]
            #sim.a[i,0] = sim.a_init[i]
            #sim.k[i,0] = sim.k_init[i]

            for t in range(tt, par.simT):

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.spouse[i,t],sim.n[i,t])
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[idx_sol],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[idx_sol],sim.a[i,t],sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    income = self.wage_func(sim.k[i,t],t)*sim.h[i,t]
                    spouse_income = (par.y0 + par.y1*t)*sim.spouse[i,t]
                    childcare = sim.n[i,t]*par.c_cost
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income + spouse_income - childcare - sim.c[i,t])
                    sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]

                    spouse_next = sim.spouse[i,t+1]
                    birth = 0 
                    if spouse_next == 1:
                        if ((sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1))):
                            birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth



###################
#   Estimation    #
###################

    def est_moment(self,beta_1, do_print=False):
        """ computes relative drop in labor hours upon child birth - used for estimation """

        par = self.par
        sim = self.sim
        sol = self.sol

        par.beta_1 = beta_1

        self.solve(do_print=do_print)
        self.simulate()

        # compute moment
        birth = np.zeros(sim.n.shape,dtype=np.int_)
        birth[:,1:] = (sim.n[:,1:] - sim.n[:,:-1]) > 0

        # time since birth
        periods = np.tile([t for t in range(par.simT)],(par.simN,1))
        time_of_birth = np.max(periods * birth, axis=1)

        I = time_of_birth>0
        time_of_birth[~I] = -1000 # never as a child
        time_of_birth = np.transpose(np.tile(time_of_birth , (par.simT,1)))
        time_since_birth = periods - time_of_birth

        # compute drop in hours in birth year
        drop_y0 = np.mean(sim.h[time_since_birth==0])/np.mean(sim.h[time_since_birth == -1]) -1

        return drop_y0 
    

    def estimate(self, do_print=False, disp=True, **kwargs):
        """estimates beta_1 to match target par.drop_y0"""

        par = self.par

        # define objective function and find root   
        obj = lambda beta: self.est_moment(beta,do_print=do_print) - par.target_drop_y0
        res = root_scalar(obj, **kwargs)
        assert res.converged
        
        if disp:
            print(res) # root is storedin par.beta_1
                    


###################################
# Solving or loading model        #
###################################

def get_model_solution(name, load, save, par):
    """Solves or loads model and simulates."""

    model = DynLaborFertModelClass(name=name, load=load, par=par)
    
    if model.sol.solved == False:
        model.solve(do_print=True)

    model.simulate()

    if save:
        model.save()

    return model


###################################
# Extra for marshall elasticities #
###################################

def marshall_age(model, shock_model):
    """Computes age specific Marshall elasticities for entire simulated population. """

    marshall_vec = np.empty(model.par.simT)
    model_sim = model.copy() #extra model for performing simulations

    for t in range(model.par.simT):
        model_sim.simulate() 
        model_sim.simulate_t(tt=t, shock_model=shock_model)
        marshall = ((model_sim.sim.h[:,t]/model.sim.h[:,t]-1)*100).mean() # response in shock period
        marshall_vec[t]=marshall

    return marshall_vec


def marshall_age_birth(model, shock_model, time):
    """computes age specific Marshall elasticities conditional on giving birth at age time"""

    sim = model.sim
    par = model.par

    birth = np.zeros(sim.n.shape)
    birth[:,1:] = np.diff(sim.n, axis=1)
    periods = np.tile([t for t in range(par.simT)],(par.simN,1))
    time_of_birth = np.max(periods * birth, axis=1)

    marshall_vec = np.empty(model.par.simT)
    model_sim = model.copy()

    for t in range(model.par.simT):
        model_sim.simulate() 
        model_sim.simulate_t(tt=t, shock_model=shock_model)
        marshall = ((model_sim.sim.h[:,t][time_of_birth==time]/model.sim.h[:,t][time_of_birth==time]-1)*100).mean() # response in shock period
        marshall_vec[t]=marshall

    return marshall_vec



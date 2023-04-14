import numpy as np
import QuantLib as ql

def data_factory():
    valuation_date = ql.Date(11, 4, 2023)
    ql.Settings.instance().evaluationDate = valuation_date
    domestic_curve = ql.FlatForward(valuation_date, ql.QuoteHandle(ql.SimpleQuote(0.01)), ql.Actual360())
    domestic_process = ql.HullWhiteProcess(ql.YieldTermStructureHandle(domestic_curve), 0.02, 0.26)
    foreign_curve = ql.FlatForward(valuation_date, ql.QuoteHandle(ql.SimpleQuote(0.035)), ql.Actual360())
    foreign_process = ql.HullWhiteProcess(ql.YieldTermStructureHandle(foreign_curve), 0.02, 0.22)
    discount_curve = ql.FlatForward(valuation_date, ql.QuoteHandle(ql.SimpleQuote(0.005)), ql.Actual360())
    discount_curve_handle = ql.YieldTermStructureHandle(discount_curve)
    correlation_matrix = np.array([[1.0,-0.06,-0.03],[-0.06,1.0,0.02],[-0.03,0.02,1.0]])
    return domestic_process, foreign_process, correlation_matrix, discount_curve_handle

class FxPathGenerator:
    def __init__(self, valuation_date, coupon_schedule, day_counter, fx_spot, fx_sigma, 
        domestic_process, foreign_process, correlation_matrix):
        
        self.valuation_date = valuation_date
        self.coupon_schedule = coupon_schedule
        self.day_counter = day_counter
        self.fx_spot = fx_spot
        self.fx_sigma = fx_sigma
        self.domestic_process = domestic_process
        self.foreign_process = foreign_process        
        self.correlation_matrix = correlation_matrix
        self.n_processes = 3
        #
        self.create_time_grids()
        
    def create_time_grids(self):
        all_coupon_dates = np.array(self.coupon_schedule)
        remaining_coupon_dates = all_coupon_dates[all_coupon_dates > self.valuation_date]
        self.time_grid = np.array([self.day_counter.yearFraction(self.valuation_date, date) 
            for date in remaining_coupon_dates])
        self.grid_steps = np.concatenate((np.array([self.time_grid[0]]), np.diff(self.time_grid)))
        self.n_steps = self.grid_steps.shape[0]
        
    def create_correlated_normal_variates(self):
        cholesky_matrix = np.linalg.cholesky(self.correlation_matrix)
        correlated_normal_variates = cholesky_matrix.dot(np.random.normal(0.0, 1.0, (self.n_processes, self.n_steps)))
        return correlated_normal_variates
        
    def next_path(self):
        e = self.create_correlated_normal_variates()
        spot_domestic = self.domestic_process.x0()
        spot_foreign = self.foreign_process.x0()
        spot_fx = self.fx_spot
        
        dw_domestic = e[0,:] * self.grid_steps
        dw_foreign = e[1,:] * self.grid_steps
        dw_fx = e[2,:] * self.grid_steps
        fx_path = np.empty(self.n_steps, dtype=float)
        
        for i in range(self.n_steps):
            dt = self.grid_steps[i]
            t = self.time_grid[i]
            spot_domestic = domestic_process.evolve(t, spot_domestic, dt, dw_domestic[i])
            spot_foreign = foreign_process.evolve(t, spot_foreign, dt, dw_foreign[i])
            spot_fx = spot_fx + ((spot_domestic - spot_foreign) * spot_fx * dt) + (self.fx_sigma * spot_fx * dw_fx[i])
            fx_path[i] = spot_fx

        return fx_path

class MonteCarloPricerPRDC:
    def __init__(self, valuation_date, coupon_schedule, day_counter, notional, discount_curve_handle, 
        payoff_function, fx_path_generator, n_paths, intro_coupon_schedule, intro_coupon_rate):

        self.valuation_date = valuation_date
        self.coupon_schedule = coupon_schedule
        self.day_counter = day_counter
        self.notional = notional
        self.discount_curve_handle = discount_curve_handle
        self.payoff_function = payoff_function
        self.fx_path_generator = fx_path_generator
        self.n_paths = n_paths
        self.intro_coupon_schedule = intro_coupon_schedule
        self.intro_coupon_rate = intro_coupon_rate
        #
        self.create_coupon_dates()
        
    def create_coupon_dates(self):
        self.all_coupon_dates = np.array(self.coupon_schedule)
        self.past_coupon_dates = self.all_coupon_dates[self.all_coupon_dates < self.valuation_date]
        n_past_coupon_dates = self.past_coupon_dates.shape[0] - 1
        self.past_coupon_rates = np.full(n_past_coupon_dates, 0.0, dtype=float)        
        self.remaining_coupon_dates = self.all_coupon_dates[self.all_coupon_dates > self.valuation_date]
        self.time_grid = np.array([self.day_counter.yearFraction(self.valuation_date, date) 
            for date in self.remaining_coupon_dates])
        self.grid_steps = np.concatenate((np.array([self.time_grid[0]]), np.diff(self.time_grid)))
        self.n_steps = self.grid_steps.shape[0]
        
        if(self.intro_coupon_schedule==None):
            self.has_intro_coupon = False
        else:
            self.intro_coupon_dates = np.array(self.intro_coupon_schedule)
            self.remaining_intro_coupon_dates = self.intro_coupon_dates[self.intro_coupon_dates > self.valuation_date]
            self.n_remaining_intro_coupon_dates = self.remaining_intro_coupon_dates.shape[0]
            if(self.n_remaining_intro_coupon_dates > 0):
                self.has_intro_coupon = True
            else:
                self.has_intro_coupon = False

    def simulate_coupon_rates(self):
        self.simulated_coupon_rates = np.zeros(self.n_steps, dtype=float)
        
        for i in range(self.n_paths):
            path = self.fx_path_generator.next_path()
            for j in range(self.n_steps):
                self.simulated_coupon_rates[j] += self.payoff_function(path[j])

        self.simulated_coupon_rates = self.simulated_coupon_rates / self.n_paths        
        if(self.has_intro_coupon): self.append_intro_coupon_rates()
        self.coupon_rates = np.concatenate((self.past_coupon_rates, self.simulated_coupon_rates))
        self.n_coupon_cash_flows = self.coupon_rates.shape[0]

    def append_intro_coupon_rates(self):
        for i in range(self.n_remaining_intro_coupon_dates):
            self.simulated_coupon_rates[i] = self.intro_coupon_rate
    
    def create_cash_flows(self):        
        self.coupon_cash_flows = np.empty(self.n_coupon_cash_flows, dtype=ql.FixedRateCoupon)
        
        for i in range(self.n_coupon_cash_flows):
            self.coupon_cash_flows[i] = ql.FixedRateCoupon(self.all_coupon_dates[i+1], self.notional, 
                self.coupon_rates[i], self.day_counter, self.all_coupon_dates[i], self.all_coupon_dates[i+1])
        
        self.coupon_leg = ql.Leg(self.coupon_cash_flows)
        redemption = ql.Redemption(self.notional, self.all_coupon_dates[-1])
        self.redemption_leg = ql.Leg(np.array([redemption]))

    def npv(self):
        self.simulate_coupon_rates()
        self.create_cash_flows()
        
        self.redemption_leg_npv = ql.CashFlows.npv(self.redemption_leg, self.discount_curve_handle, False)
        self.coupon_leg_npv = ql.CashFlows.npv(self.coupon_leg, self.discount_curve_handle, False)
        
        date = [payment.date() for payment in self.coupon_leg]
        amount = [payment.amount() for payment in self.coupon_leg]
        amount[-1] += self.notional
        pv = [ql.CashFlows.npv(np.array([payment]), self.discount_curve_handle, False) for payment in self.coupon_leg]        
        pv[-1] += self.redemption_leg_npv
        
        self.cash_flow_table = np.array([date, amount, pv])
        return self.coupon_leg_npv + self.redemption_leg_npv    

# create processes for FX path generator, correlation and discount curve for JPY
domestic_process, foreign_process, correlation_matrix, discount_curve_handle = data_factory()

# create schedules for coupon- and intro coupon payments
effectiveDate = ql.Date(3, ql.September, 2015)
terminationDate = ql.Date(3, ql.September, 2041)
coupon_schedule = ql.MakeSchedule(effectiveDate, terminationDate, ql.Period(6, ql.Months), 
    backwards=True, calendar=ql.TARGET(), convention=ql.ModifiedFollowing)

intro_coupon_termination_date = ql.Date(3, ql.September, 2016)
intro_coupon_schedule = ql.MakeSchedule(effectiveDate, intro_coupon_termination_date, ql.Period(6, ql.Months), 
    backwards=True, calendar=ql.TARGET(), convention=ql.ModifiedFollowing)

# create FX path generator
usd_jpy_spot = 133.2681
usd_jpy_vol = 0.1
fx_path_generator = FxPathGenerator(ql.Settings.instance().evaluationDate, coupon_schedule, ql.Actual360(), 
    usd_jpy_spot, usd_jpy_vol, domestic_process, foreign_process, correlation_matrix)

# create PRDC pricer
notional = 300000000.0
intro_coupon_rate = 0.022
n_paths = 10000
prdc_payoff_function = lambda fx_rate : min(max(0.122 * (fx_rate / 120.0) - 0.1, 0.0), 0.022)
prdc_pricer = MonteCarloPricerPRDC(ql.Settings.instance().evaluationDate, coupon_schedule, ql.Actual360(),notional, 
    discount_curve_handle, prdc_payoff_function, fx_path_generator, n_paths, intro_coupon_schedule, intro_coupon_rate)

# request results
npv_ccy = prdc_pricer.npv()
print('PV in CCY: {}'.format(npv_ccy))
jpy_eur = 145.3275
npv_eur = npv_ccy / jpy_eur
print('PV in EUR: {}'.format(npv_eur))
print()
print('Cash flow dates: {}'.format(prdc_pricer.cash_flow_table[0]))
print()
print('Cash flows: {}'.format(prdc_pricer.cash_flow_table[1]))
print()
print('Present values of cash flows: {}'.format(prdc_pricer.cash_flow_table[2]))







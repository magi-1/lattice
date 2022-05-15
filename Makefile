# https://opensource.com/article/18/8/what-how-makefile

pull_all:
	@pdm run python $(CURDIR)/lattice/utils/get_historical.py
	
clear_sims:
	@rm -rf $(CURDIR)/data/sim_out
	@rm -rf $(CURDIR)/data/report

backtest:
	@pdm run python lattice/simulations/backtest.py $(CONFIG) -n $(SIMS)

report:
	@pdm run python $(CURDIR)/lattice/utils/report.py

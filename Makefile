# https://opensource.com/article/18/8/what-how-makefile

pull_all_data:
	@pdm run python $(CURDIR)/lattice/utils/get_historical_data.py
	
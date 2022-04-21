# https://opensource.com/article/18/8/what-how-makefile

pull_all:
	@pdm run python $(CURDIR)/lattice/utils/get_historical.py
	
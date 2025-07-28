# Variables
MYSQL_DATABASE := YOUR_DATABASE
MYSQL_USER := YOUR_USERNAME

# Target
query:
	# Set locale in the shell before running the commands
	@echo Running query...
	cd src/query && \
	LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 ./main.sh $(MYSQL_DATABASE) $(MYSQL_USER)

M1:
	# Set locale in the shell before running the commands
	@echo Creating M1...
	cd slurm && \
	sbatch M1.slurm

M2:
	# Set locale in the shell before running the commands
	@echo Creating M2...
	cd slurm && \
	sbatch M2.slurm

M3:
	# Set locale in the shell before running the commands
	@echo Creating M3...
	cd slurm && \
	sbatch M3.slurm

model:
	# Set locale in the shell before running the commands
	@echo Creating M1...
	cd slurm && \
	sbatch M4.slurm

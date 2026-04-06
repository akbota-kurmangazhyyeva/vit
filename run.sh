tegrastats --interval 100 --logfile tegra.log &
TEGRA_PID=$!

python3.6 inf.py ./results

kill $TEGRA_PID
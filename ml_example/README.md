# Machine Learning Examples

These examples based on code from https://github.com/uber-research/deconstructing-lottery-tickets?uclick_id=eae6f0ed-8383-4f55-8f4c-f2a2489948ab
The lottery ticket example is fully functional.
The signSGD example works outside of dependent rounding (too slow to be run every epoch). It also currently runs on CPU only for stochastic and dependent rounding (had to for my setup, feel free to disable).

## SETUP EXAMPLE FOR WSL

1. Get extra dependencies

    pip install tf

2. Test

    Run one of the commands from the shell script to test.

## STRUCTURE

```
ml_example
+-- example_lottery_ticket.sh   # Example scripts to run for Lottery Ticket testing
+-- example_signsgd.sh          # Example scripts to run for SignSGD testing
+-- example.py                  # Full example implementations
+-- LICENSE.txt                 # License from Uber paper
+-- lottery_ticket_pruner.py    # Pruning utils
+-- README.md                   # README
```
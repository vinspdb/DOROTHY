# Leveraging a foundation deep neural embedding in process discovery under not-Pareto distribution

**The repository contains code referred to the work:**

*Vincenzo Pasquadibisceglie, Annalisa Appice, Giovanni Discanno, Donato Malerba*

[*Leveraging a foundation deep neural embedding in process discovery under not-Pareto distribution*](https://ieeexplore.ieee.org/document/11220731)

Please cite our work if you find it useful for your research and work.

```
@INPROCEEDINGS{11220731,
  author={Pasquadibisceglie, Vincenzo and Appice, Annalisa and Discanno, Giovanni and Malerba, Donato},
  booktitle={2025 7th International Conference on Process Mining (ICPM)}, 
  title={Leveraging a foundation deep neural embedding in process discovery under not-Pareto distribution}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={Process mining;Accuracy;Liquids;Noise;Process control;Sepsis;Recording;Mirrors;Standards;Monitoring;Process discovery;Trace deep embedding;Trace extraction strategy;Trace clustering;Pareto principle},
  doi={10.1109/ICPM66919.2025.11220731}}
```

# How to use
Process Discovery using DOROTHY
- event_log: event log name
- miner: ILP miner (ilp), Inductive Miner (im) or Split Miner (sm)

```
python -m main.py -event_log sepsis -miner im
```

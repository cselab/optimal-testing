bsub -n 36 -R "select[model==XeonGold_6150]fullnode" -W 24:00 python3 nested.py --cores 72 --nlive 100 --dlogz 0.1 --case 3

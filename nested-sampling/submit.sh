bsub -n 36 -R "select[model==XeonGold_6150]fullnode" -W 24:00 python3 nested.py --cores 72 --nlive 500 --dlogz 0.01 --case 3

import argparse
import os
import warnings
import torch
import pandas as pd
from .data import SlicePeptide as Peptide
from .comp import COREX
from .comp.sampler import get_sampler
from .comp import MaxMinCriterion
from . import _customize_print as cprint

def _get_id_name_pairs(peptide):
    _atom_df = peptide.atoms._atom_df
    res_ids = _atom_df.res_id.unique()
    res_names = [_atom_df[_atom_df.res_id==i].res_name.values[0]
                for i in res_ids]
    return list(zip(res_ids, res_names))

def _format_corex(peptide, corex):
    _pairs = _get_id_name_pairs(peptide)
    _data = [{'Residue No.': p_[0], 'Residue': p_[1], 'COREX': c_}
             for p_, c_ in zip(_pairs, corex)]
    return pd.DataFrame(_data)

def main_gpucorex_():
    parser = argparse.ArgumentParser(description='GPUCOREX Computation')
    parser.add_argument('-p',  '--protein',         required=True,  type=str,   default=None,  help='The protein PDB file path for COREX computation.')
    parser.add_argument('-o',  '--output',          required=False, type=str,   default=None,  help='The COREX output path (if it is not provided, directly print the output).')
    parser.add_argument('-ws', '--windowsize',      required=False, type=int,   default=10,    help='The window size of partition.')
    parser.add_argument('-ms', '--minsize',         required=False, type=int,   default=4,     help='The min window size of partition.')
    parser.add_argument('-ds', '--gibbsfactor',     required=False, type=float, default=-0.12, help='The delta Gibbs free energy factor.')
    parser.add_argument('-rc', '--residueconstant', required=False, type=str,   default=None,  help='The path to the residue constant file.')
    parser.add_argument('-rt', '--radiustable',     required=False, type=str,   default=None,  help='The path to the atom radius table file.')
    parser.add_argument('-w',  '--worker',          required=False, type=int,   default=10,    help='The number of processes. Set 0<= to make the program auto decide it.')
    parser.add_argument('-b',  '--batchsize',       required=False, type=int,   default=512,   help='The batch size for computation (how many samples are computed at the same time).')
    parser.add_argument('-n',  '--sample',          required=False, type=int,   default=10000, help='The sample number per partition (not work for exhaustive sampler).')
    parser.add_argument('-c',  '--cpu',             required=False, action='store_true', help='Use CPU to compute.')
    parser.add_argument('-g',  '--gpu',             required=False, action='store_true', help='Use GPU to compute.')
    parser.add_argument('-dt', '--datatype',        required=False, choices=['float64', 'float32', 'float16'], default='float64', help='The data type used to compute, float64 is suggested for precision.')
    parser.add_argument('-s',  '--sampler',         required=False, choices=['exhaustive', 'montecarlo', 'adaptive_montecarlo'], default='exhaustive', help='The sampler (if the sample number is larger than the total states of the protein, please use exhaustive).')
    parser.add_argument('-t',  '--probability',     required=False, type=float, default=0.75,  help='The probability threshold for sampler.')
    parser.add_argument('-a',  '--adaptiveratio',   required=False, type=float, default=0.05,  help='The adaptive ratio for sampler.')
    parser.add_argument('-tp', '--temperature',     required=False, type=float, default=298.15,help='The experiment temperature.')
    parser.add_argument('-bf', '--basefraction',    required=False, type=float, default=1.0,   help='The base fraction for COREX value aggregation.')
    parser.add_argument('-S',  '--silence',         required=False, action='store_true', help='Keep silence and do not show progress bar.')
    parser.add_argument('-e',  '--entropyfactor',   required=False, type=float, default=None,  help='The entropy factor. If it is not provided, it will automatically been searched.')
    
    
    
    
    args = parser.parse_args()
    
    try:
        peptide = Peptide(args.protein,
                        window_size=args.windowsize,
                        min_size=args.minsize,
                        dSbb_len_correlation=args.gibbsfactor,
                        residue_constant=args.residueconstant,
                        radius_table=args.radiustable)
        
        cprint.info(f'Peptide {args.protein} loaded, consisting of {len(peptide.residues)} residues.', silence=args.silence)
    except Exception as e: cprint.error(str(e))
    
    args.worker = args.worker if args.worker > 0 else os.cpu_count()
    if args.worker > os.cpu_count():
        args.worker = os.cpu_count()
        cprint.warn(f'The worker number ({args.worker}) is larger than CPU cores number ({os.cpu_count()}). It has been set to {os.cpu_count()}.', silence=args.silence)
        
    if args.batchsize < 1: cprint.error(f'Batch size must be larger than 1, {args.batchsize} is not acceptable.')
    
    if args.datatype == 'float16':
        dtype = torch.float16
        cprint.warn('Overflow may be caused by insufficient precision (Float16).', silence=args.silence)
    if args.datatype == 'float32':
        dtype = torch.float32
        cprint.warn('Overflow may be caused by insufficient precision (Float32).', silence=args.silence)
    if args.datatype == 'float64':
        dtype = torch.float64
    
    device = 'cpu'
    if args.gpu:
        if torch.cuda.is_available(): device = 'cuda'
        else: cprint.warn('GPU devices are not avaliable on this device, switch to CPU.', silence=args.silence)
    elif args.cpu:
        if torch.cuda.is_available(): cprint.warn('GPU devices are avaliable on this machine, add --gpu or -g to use GPU acceleration.', silence=args.silence)
        device = 'cpu'
    else: device = 'cpu'
    
    sampler = get_sampler(args.sampler)
    if args.sampler == 'exhaustive':
        cprint.info(f'Use exhausitve sampler.', silence=args.silence)
        sampler_args = {}
    if args.sampler == 'montecarlo':
        cprint.info(f'Use Monte Carlo sampler. (Probability Threshold={args.probability})', silence=args.silence)
        sampler_args = {
            'probability': args.probability,
            'temperature': args.temperature
        }
    if args.sampler == 'adaptive_montecarlo':
        cprint.info(f'Use adaptive Monte Carlo sampler. (Probability Threshold={args.probability}, Adaptive Ratio={args.adaptiveratio})', silence=args.silence)
        sampler_args = {
            'probability': args.probability,
            'adaptive_rate': args.adaptiveratio,
            'temperature': args.temperature
        }
        
    if args.entropyfactor is None: sconf_weight = MaxMinCriterion(1.0, 0.1, (0.01, 2))
    else: sconf_weight = args.entropyfactor
    
    try:
        cprint.info(f'Use {args.worker} CPU processes (cores) and {device} device to compute.', silence=args.silence)
        corex = COREX(workers=args.worker,
                    batch_size=args.batchsize,
                    samples=args.sample,
                    device=device,
                    dtype=dtype,
                    sampler=sampler,
                    sampler_args=sampler_args,
                    temperature=args.temperature,
                    base_fraction=args.basefraction,
                    silence=args.silence,
                    sconf_weight=sconf_weight)
        corex_val = corex.forward(peptide)
        corex_val_formatted = _format_corex(peptide, corex_val.cpu().tolist())
    except Exception as e: cprint.error(str(e))
    
    cprint.success(f'COREX finished, {corex.time_cost_total:.2f} seconds costed in total.', silence=args.silence)
    
    if args.output is None:
        print(corex_val_formatted.to_string())
    else:
        corex_val_formatted.to_csv(args.output)
        cprint.info(f'The COREX values are saved to {args.output}. (CSV format)', silence=args.silence)
        
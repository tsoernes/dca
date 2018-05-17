runsim=true
# runsim=false
runplot=true

# events=100000
events=470000

# logiter=5000
logiter=25000

avg=16
ext=".0"

targs=true
#targs=false
grads=true
#grads=false
hla=true
#hla=false
finalhoff=true
#finalhoff=false
finalnohoff=true
#finalnohoff=false


runargs=(--log_iter "${logiter}" --log_level 30 --avg_runs "${avg}" -i "${events}" \
   --log_file "plots/plot-log${ext}" --breakout_thresh 0.4)

## TARGETS ##
if [ "$targs" = true ] ; then
    if [ "$runsim" = true ] ; then
        # SMDP Discount
        # TODO tune beta, lr
        python3 main.py singhnet "${runargs[@]}" \
                -save_bp targ-smdp --target discount -rtype smdp_callcount \
                -phoff --beta_disc --beta 20 -lr 5e-6 || exit 1
        # MDP Discount
        python3 main.py singhnet "${runargs[@]}" \
                -phoff -save_bp targ-mdp --target discount \
                --net_lr 2.02e-7 || exit 1
        # MDP Average
        python3 main.py singhnet "${runargs[@]}" \
                -phoff -save_bp targ-avg \
                --net_lr 3.43e-06 --weight_beta 0.00368 || exit 1
    fi
    if [ "$runplot" = true ] ; then
        python3 plotter.py "targ-smdp${ext}" "targ-mdp${ext}" "targ-avg${ext}" \
                --labels 'SMDP discount' 'MDP discount' 'MDP avg. rewards' \
                --title "Target comparison (with hand-offs)" \
                --ctype new hoff tot --plot_save targets || exit 1
    fi
    echo "Finished targets"
fi

if [ "$grads" = true ] ; then
    ## GRADIENTS (no hoffs) ##
    if [ "$runsim" = true ] ; then
        # Semi-grad A-MDP (same as 'MDP Average', without hoffs)
        python3 main.py singhnet "${runargs[@]}" \
                -save_bp grads-semi  --net_lr 3.43e-06 --weight_beta 0.00368 || exit 1
        # Residual grad A-MDP
        python3 main.py residsinghnet "${runargs[@]}" \
                -save_bp grads-resid --net_lr 1.6e-05 || exit 1
        #  TDC A-MDP
        python3 main.py tftdcsinghnet "${runargs[@]}" \
                -save_bp grads-tdc || exit 1
        #  TDC MDP Discount
        python3 main.py tftdcsinghnet "${runargs[@]}" \
                -save_bp grads-tdc-gam --target discount \
                --net_lr 1.91e-06 --grad_beta 5e-09 || exit 1
    fi
    if [ "$runplot" = true ] ; then
        python3 plotter.py "grads-semi${ext}" "grads-resid${ext}" "grads-tdc${ext}" "grads-tdc-gam${ext}" \
                --labels 'Semi-grad. (A-MDP)' 'Residual grad. (A-MDP)' 'TDC (A-MDP)' 'TDC (MDP)' \
                --title "Gradient comparison (no hand-offs)" \
                --ctype tot --plot_save grads || exit 1
    fi
    echo "Finished Grads"
fi

## Exploration RS-SARSA ##
# if [ "$runsim" = true ] ; then
#     # RS-SARSA 
#     python3 main.py rs_sarsa --log_iter $logiter --avg_runs $avg $runt \
#             -save_bp hla-rssarsa --target discount -phoff 0.15 -hla || exit 1
# fi
# python3 plotter.py "exp-rssarsa-greedy${ext}" "exp-rssarsa-boltzlo${ext}" "exp-rssarsa-boltzhi${ext}" \
#         --labels 'RS-SARSA Greedy' 'RS-SARSA Boltmann Low' 'RS-SARSA Boltmann High' \
#         --title "Exploration for state-action methods" \
#         --ctype new hoff --plot_save exp-rssarsa || exit 1

## Exploration VNet ##
# if [ "$runsim" = true ] ; then
#     # VNet greedy
#     python3 main.py rs_sarsa --log_iter $logiter --avg_runs $avg $runt \
#             -save_bp exp-vnet-greedy --target discount -phoff 0.15 -hla || exit 1
# fi
# python3 plotter.py "exp-vnet-greedy${ext}" "exp-vnet-boltzlo${ext}" "exp-vnet-boltzhi${ext}" \
#         --labels 'VNet Greedy' 'VNet Boltmann Low' 'VNet Boltmann High' \
#         --title "Exploration for state methods" \
#         --ctype new hoff --plot_save exp-vnet || exit 1

if [ "$hla" = true ] ; then
    ## HLA ##
    if [ "$runsim" = true ] ; then
        #  TDC A-MDP
        python3 main.py tftdcsinghnet "${runargs[@]}" \
                -save_bp vnet -phoff || exit 1
        # TDC A-MDP HLA
        python3 main.py tftdcsinghnet "${runargs[@]}" \
                -save_bp hla-vnet -hla -phoff || exit 1
        # RS-SARSA
        python3 main.py rs_sarsa "${runargs[@]}" \
                -save_bp rssarsa --lilith -phoff || exit 1
        # RS-SARSA HLA
        python3 main.py hla_rs_sarsa "${runargs[@]}" \
                -save_bp hla-rssarsa --lilith -phoff -hla || exit 1
    fi
    if [ "$runplot" = true ] ; then
        python3 plotter.py "vnet${ext}" "hla-vnet${ext}" "rssarsa${ext}" "hla-rssarsa${ext}" \
                --labels 'VNet' 'VNet (HLA)' 'RS-SARSA' 'RS-SARSA (HLA)' \
                --title "Hand-off look-ahead" \
                --ctype new hoff tot --plot_save hla || exit 1
   fi
    echo "Finished HLA"
fi

if [ "$finalhoff" = true ] ; then
    ## Final comparison ##
    # VNet and HLA from previous run
    if [ "$runsim" = true ] ; then
        # FCA
        python3 main.py fixedassign "${runargs[@]}" \
                -save_bp final-fca -phoff || exit 1
        # RandomAssign
        python3 main.py randomassign "${runargs[@]}" \
                -save_bp final-rand -phoff || exit 1
    fi
    if [ "$runplot" = true ] ; then
        python3 plotter.py "hla-vnet${ext}" "hla-rssarsa${ext}" "final-fca${ext}" "final-rand${ext}" \
                --labels 'VNet (HLA)' 'RS-SARSA (HLA)' 'FCA' 'Random assignment' \
                --title "RL vs non-learning agents (with hand-offs)" \
                --ctype new hoff --plot_save final-whoff || exit 1
    fi
    echo "Finished Final w/hoff"
fi

if [ "$finalnohoff" = true ] ; then
    ## Final comparison, without hoffs
    if [ "$runsim" = true ] ; then
        # TDC avg.
        python3 main.py tftdcsinghnet "${runargs[@]}" \
                -save_bp final-nohoff-vnet || exit 1
        # RS-SARSA
        python3 main.py rs_sarsa "${runargs[@]}" \
                -save_bp final-nohoff-rssarsa --lilith || exit 1
        # FCA
        python3 main.py fixedassign "${runargs[@]}" \
                -save_bp final-nohoff-fca || exit 1
        # RandomAssign
        python3 main.py randomassign "${runargs[@]}" \
                -save_bp final-nohoff-rand || exit 1
    fi
    if [ "$runplot" = true ] ; then
        python3 plotter.py "final-nohoff-vnet${ext}" "final-nohoff-rssarsa${ext}" \
                "final-nohoff-fca${ext}" "final-nohoff-rand${ext}" \
                --labels 'VNet' 'RS-SARSA' 'FCA' 'Random assignment' \
                --title "RL vs non-learning agents (no hand-offs)" \
                --ctype new --plot_save final-nohoff || exit 1
    fi
    echo "Finished Final wo/hoff"
fi
echo "FINISHED ALL"

plotonly=true
# plotonly=false
runt="-i 10000"
# runt="-h 24"
logiter=1000
avg=3
ext=".0"

## TARGETS ##
if [ "$plotonly" = false ] ; then
    # SMDP Discount
    # TODO tune beta, lr
    python3 main.py singhnet --log_iter $logiter --avg_runs $avg $runt \
            -save_bp targ-smdp --target discount -rtype smdp_callcount \
            -phoff --beta_disc --beta 20 -lr 5e-6 || {exit 1}
    # MDP Discount
    python3 main.py singhnet --log_iter $logiter --avg_runs $avg $runt \
            -phoff -save_bp targ-mdp --target discount || {exit 1}
    # MDP Average
    python3 main.py singhnet --log_iter $logiter --avg_runs $avg $runt \
            -phoff -save_bp targ-avg || {exit 1}
fi
python3 plotter.py "targ-smdp${ext}" "targ-mdp${ext}" "targ-avg${ext}" \
        --labels 'SMDP discount' 'MDP discount' 'MDP avg. rewards' \
        --title "Target comparison" \
        --ctype new hoff tot --plot_save targets || {exit 1}

## GRADIENTS ##
if [ "$plotonly" = false ] ; then
    # semi-grad A-MDP
    python3 main.py singhnet --log_iter $logiter --avg_runs $avg $runt \
            -save_bp grads-semi  || {exit 1}
    # residual grad A-MDP
    python3 main.py residsinghnet --log_iter $logiter --avg_runs $avg $runt \
            -save_bp grads-resid --net_lr 1.6e-05 || {exit 1}
    #  TDC A-MDP
    python3 main.py tfdcssinghnet --log_iter $logiter --avg_runs $avg $runt \
            -save_bp grads-tdc || {exit 1}
    #  TDC MDP
    # TODO tune gam lr
    python3 main.py tfdcssinghnet --log_iter $logiter --avg_runs $avg $runt \
            -save_bp grads-tdc-gam --target discount --gamma X -lr Y || {exit 1}
fi
python3 plotter.py "grads-semi${ext}" "grads-resid${ext}" "grads-tdc${ext}" "grads-tdc-gam${ext}"\
        --labels 'Semi-gradient (A-MDP)' 'Residual (A-MDP)' 'TDC (A-MDP)' 'TDC (MDP)' \
        --title "Gradient comparison" \
        --ctype tot --plot_save grads || {exit 1}
# NOTE do note that hoffs are turned off

## Exploration RS-SARSA ##
# if [ "$plotonly" = false ] ; then
#     # RS-SARSA 
#     python3 main.py rs_sarsa --log_iter $logiter --avg_runs $avg $runt \
#             -save_bp hla-rssarsa --target discount -phoff 0.15 -hla || {exit 1}
# fi
# python3 plotter.py "exp-rssarsa-greedy${ext}" "exp-rssarsa-boltzlo${ext}" "exp-rssarsa-boltzhi${ext}" \
#         --labels 'RS-SARSA Greedy' 'RS-SARSA Boltmann Low' 'RS-SARSA Boltmann High' \
#         --title "Exploration for state-action methods" \
#         --ctype new hoff --plot_save exp-rssarsa || {exit 1}

## Exploration VNet ##
# if [ "$plotonly" = false ] ; then
#     # VNet greedy
#     python3 main.py rs_sarsa --log_iter $logiter --avg_runs $avg $runt \
#             -save_bp exp-vnet-greedy --target discount -phoff 0.15 -hla || {exit 1}
# fi
# python3 plotter.py "exp-vnet-greedy${ext}" "exp-vnet-boltzlo${ext}" "exp-vnet-boltzhi${ext}" \
#         --labels 'VNet Greedy' 'VNet Boltmann Low' 'VNet Boltmann High' \
#         --title "Exploration for state methods" \
#         --ctype new hoff --plot_save exp-vnet || {exit 1}

## HLA ##
if [ "$plotonly" = false ] ; then
    #  TDC A-MDP
    python3 main.py tfdcssinghnet --log_iter $logiter --avg_runs $avg $runt \
            -save_bp vnet -phoff || {exit 1}
    # TDC A-MDP HLA
    python3 main.py tftdcsinghnet --log_iter $logiter --avg_runs $avg $runt \
            -save_bp hla-vnet -hla -phoff || {exit 1}
    # RS-SARSA
    python3 main.py rs_sarsa --log_iter $logiter --avg_runs $avg $runt \
            -save_bp rssarsa --lilith -phoff || {exit 1}
    # RS-SARSA HLA
    python3 main.py rs_sarsa --log_iter $logiter --avg_runs $avg $runt \
            -save_bp hla-rssarsa --lilith -phoff -hla || {exit 1}
fi
python3 plotter.py "vnet${ext}" "hla-vnet${ext}" "rssarsa${ext}" "hla-rssarsa${ext}" \
        --labels 'VNet' 'VNet (HLA)' 'RS-SARSA' 'RS-SARSA (HLA)' \
        --title "Hand-off look-ahead" \
        --ctype new hoff tot --plot_save hla || {exit 1}

## Final comparison
if [ "$plotonly" = false ] ; then
    # FCA
    python3 main.py fixedassign --log_iter $logiter --avg_runs $avg $runt \
            -save_bp final-fca -phoff || {exit 1}
   # RandomAssign
    python3 main.py randomassign --log_iter $logiter --avg_runs $avg $runt \
            -save_bp final-rand -phoff || {exit 1}
fi
python3 plotter.py "hla-vnet${ext}" "hla-rssarsa${ext}" "final-fca${ext}" "final-rand${ext}"\
        --labels 'VNet (HLA)' 'RS-SARSA (HLA)' 'FCA' 'Random assignment' \
        --title "RL vs non-learning agents (with hand-offs)" \
        --ctype new hoff --plot_save final-whoff || {exit 1}

## Final comparison, without hoffs
if [ "$plotonly" = false ] ; then
    # TDC avg.
    python3 main.py tftdcsinghnet --log_iter $logiter --avg_runs $avg $runt \
            -save_bp final-nohoff-vnet || {exit 1}
    # RS-SARSA
    python3 main.py rs_sarsa --log_iter $logiter --avg_runs $avg $runt \
            -save_bp final-nohoff-rssarsa --lilith || {exit 1}
    # FCA
    python3 main.py fixedassign --log_iter $logiter --avg_runs $avg $runt \
            -save_bp final-nohoff-fca || {exit 1}
    # RandomAssign
    python3 main.py randomassign --log_iter $logiter --avg_runs $avg $runt \
            -save_bp final-nohoff-rand || {exit 1}
fi
python3 plotter.py "final-nohoff-vnet{ext}" "final-nohoff-rssarsa{ext}" \
        "final-nohoff-fca{ext}" "final-nohoff-rand{ext}"\
        --labels 'VNet' 'RS-SARSA' 'FCA' 'Random assignment' \
        --title "RL vs non-learning agents, without hand-offs" \
        --ctype new --plot_save final-nohoff || {exit 1}

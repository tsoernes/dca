plotonly=true
# plotonly=false
events=10000
logiter=1000
avg=3
ext=".0"

## TARGETS ##
if [ "$plotonly" = false ] ; then
    # SMDP Discount
    python3 main.py singhnet --log_iter $logiter --avg_runs $avg -i $events \
            -save_bp targ-smdp --target discount -rtype smdp_callcount \
            --beta_disc --beta 20 -lr 5e-6 || {exit 1}
    # MDP Discount
    python3 main.py singhnet --log_iter $logiter --avg_runs $avg -i $events \
            -save_bp targ-mdp --target discount || {exit 1}
    # MDP Average
    python3 main.py singhnet --log_iter $logiter --avg_runs $avg -i $events \
            -save_bp targ-avg || {exit 1}
fi
python3 plotter.py "targ-smdp${ext}" "targ-mdp${ext}" "targ-avg${ext}" \
        --labels 'SMDP discount' 'MDP discount' 'MDP avg. rewards' --title "Target comparison" \
        --ctype tot --plot_save grad-cmp || {exit 1}

## GRADIENTS ##
if [ "$plotonly" = false ] ; then
    # semi-grad A-MDP
    python3 main.py singhnet --log_iter $logiter --avg_runs $avg -i $events \
            -save_bp grads-semi || {exit 1}
    # residual grad A-MDP
    python3 main.py residsinghnet --log_iter $logiter --avg_runs $avg -i $events \
            -save_bp grads-resid || {exit 1}
    #  TDC A-MDP
    python3 main.py tfdcssinghnet --log_iter $logiter --avg_runs $avg -i $events \
            -save_bp grads-tdc || {exit 1}
fi
python3 plotter.py "grads-semi${ext}" "grads-resid${ext}" "grads-tdc${ext}" \
        --labels 'Semi-gradient' 'Residual' 'TDC' --title "Gradient comparison" \
        --ctype tot --plot_save grad-cmp || {exit 1}

## Exploration RS-SARSA ##
if [ "$plotonly" = false ] ; then
    # RS-SARSA HLA
    python3 main.py rs_sarsa --log_iter $logiter --avg_runs $avg -i $events \
            -save_bp hla-rssarsa --target discount -phoff 0.15 -hla || {exit 1}
fi
python3 plotter.py "exp-rssarsa-greedy${ext}" "exp-rssarsa-boltzlo${ext}" "exp-rssarsa-boltzhi${ext}" \
        --labels 'RS-SARSA Greedy' 'RS-SARSA Boltmann Low' 'RS-SARSA Boltmann High' \
        --title "Exploration for Q-Learning" \
        --ctype new hoff --plot_save grad-cmp || {exit 1}

## Exploration RS-SARSA ##
if [ "$plotonly" = false ] ; then
    # RS-SARSA HLA
    python3 main.py rs_sarsa --log_iter $logiter --avg_runs $avg -i $events \
            -save_bp hla-rssarsa --target discount -phoff 0.15 -hla || {exit 1}
fi
python3 plotter.py "exp-rssarsa-greedy${ext}" "exp-rssarsa-boltzlo${ext}" "exp-rssarsa-boltzhi${ext}" \
        --labels 'RS-SARSA Greedy' 'RS-SARSA Boltmann Low' 'RS-SARSA Boltmann High' \
        --title "Exploration for Q-Learning" \
        --ctype new hoff --plot_save grad-cmp || {exit 1}

## HLA ##
if [ "$plotonly" = false ] ; then
    # TDC avg. HLA
    python3 main.py tftdcsinghnet --log_iter $logiter --avg_runs $avg -i $events \
            -save_bp hla-vnet -hla -phoff 0.15 || {exit 1}
    # TODO set exploration
    # RS-SARSA
            # -save_bp rssarsa || {exit 1}
    # RS-SARSA HLA
    python3 main.py rs_sarsa --log_iter $logiter --avg_runs $avg -i $events \
            -save_bp hla-rssarsa --target discount -phoff 0.15 -hla || {exit 1}
fi
python3 plotter.py "grads-tdc${ext}" "hla-vnet${ext}" "rssarsa${ext}" "hla-rssarsa${ext}" \
        --labels 'VNet' 'HLA-VNet' 'RS-SARSA' 'HLA-RS-SARSA' --title "Hand-off Look-ahead" \
        --ctype new hoff tot --plot_save grad-cmp || {exit 1}

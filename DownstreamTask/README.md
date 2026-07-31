$YOURPATH=***
echo "=== Starting OS Multimodal Survival Analysis ==="
python  $YOURPATH/downstreamTasks/downstream_survival_multimodal.py \
    --model CPSformer \
    --survival_type os \
    --output_dir 0716revise/experiments/results_survival_multimodal \
    --n_folds 5 --epochs 200 --patience 30 \
    2>&1 | tee 0716revise/experiments/survival_os_mm.log

echo ""
echo "=== Starting DSS Multimodal Survival Analysis ==="
python $YOURPATH/downstreamTasks/downstream_survival_multimodal.py \
    --model CPSformer \
    --survival_type dss \
    --output_dir ./experiments/results_survival_multimodal \
    --n_folds 5 --epochs 200 --patience 30 \
    2>&1 | tee ./experiments/survival_dss_mm.log

echo ""
echo "=== All Done ==="

### Attentions ## 
## Please download ./data/TCGA/ all data and unzip them



## Extract CPSformer features, you need:
1. prepare patch images, segment files are optional.  

python extract_cps_features.py  --root_dir   ./data  --samples_per_patient  100  --cohort COAD 
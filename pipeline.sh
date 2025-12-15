echo "Assembling data..."
uv run ./analysis/01_assemble_data.py
echo "Estimating RRs..."
uv run ./analysis/02_meta_estimate_RRs.py
echo "Assembling resistance data..."
uv run ./analysis/03_assemble_resistance_data.py
echo "Calculating GC attributable HIV burden..."
uv run ./analysis/04_gc_attributable_hiv.py
echo "Running averted burden analyses..."
uv run ./analysis/05_appropriate_abx_averted_burden.py

echo "Pipeline complete."
echo "Making figures and tables."

uv run ./analysis/figure_overall_burden_gc_resistance_hiv.py
uv run ./analysis/figure_supplemental_across_africa_trends.py
uv run ./analysis/figure_gc_attributable_burden.py
uv run ./analysis/figure_supplemental_across_africa_attributable_sex.py
uv run ./analysis/figure_supplemental_across_africa_attributable_resistance.py
uv run ./analysis/figure_abx_avertable_hiv.py

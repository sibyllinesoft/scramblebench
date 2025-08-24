#!/bin/bash
# S8 Statistical Analysis Dependencies Installation Script
# Generated on: 2025-01-23
# Review this script before execution

set -euo pipefail  # Exit on any error

echo "🔍 Checking system requirements..."
echo "Python version: $(python3 --version)"
echo "Pip version: $(python3 -m pip --version)"

echo ""
echo "📦 Installing S8 Python dependencies..."
echo "Installing core data science packages..."
python3 -m pip install --user numpy>=1.21.0 pandas>=1.3.0 scipy>=1.7.0

echo "Installing statistical modeling packages..."
python3 -m pip install --user statsmodels>=0.13.0 scikit-learn>=1.0.0

echo "Installing testing packages..."
python3 -m pip install --user pytest>=6.0.0 pytest-cov>=3.0.0

echo ""
echo "🔧 Optional: Installing R integration support..."
echo "Uncomment the following lines if you want R backend support:"
echo "# python3 -m pip install --user rpy2>=3.4.0"
echo "# R -e \"install.packages(c('lme4', 'mgcv', 'segmented'))\""

echo ""
echo "✅ Verifying installation..."
echo "Checking numpy..."
python3 -c "import numpy; print(f'numpy {numpy.__version__} installed')"

echo "Checking pandas..."
python3 -c "import pandas; print(f'pandas {pandas.__version__} installed')"

echo "Checking scipy..."
python3 -c "import scipy; print(f'scipy {scipy.__version__} installed')"

echo "Checking statsmodels..."
python3 -c "import statsmodels; print(f'statsmodels {statsmodels.__version__} installed')"

echo "Checking scikit-learn..."
python3 -c "import sklearn; print(f'scikit-learn {sklearn.__version__} installed')"

echo ""
echo "🎉 S8 dependencies installation complete!"
echo ""
echo "Next steps:"
echo "1. Run the S8 component tests: python3 test_s8_basic.py"
echo "2. Try the S8 demo: python3 s8_analysis_demo.py --output-dir s8_demo_results"
echo "3. Use S8 with your data: scramblebench analyze fit --run-id <your_run_id>"
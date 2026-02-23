# Install dependencies Windows

$ENV_NAME = "NarrativeMultilingualSubtleStereotypes"

python -m venv $ENV_NAME
& "$ENV_NAME\Scripts\Activate.ps1"
pip install -r .\requirements.txt
Write-Host "Installation complete"
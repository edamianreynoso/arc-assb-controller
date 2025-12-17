$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$paperDir = Join-Path $repoRoot "paper_latex"
$outDir = Join-Path $repoRoot "release_assets"
$stagingDir = Join-Path $outDir "arxiv_upload"
$outZip = Join-Path $outDir "arxiv_upload.zip"

if (-not (Test-Path (Join-Path $paperDir "main.tex"))) {
  throw "Missing paper_latex/main.tex"
}
if (-not (Test-Path (Join-Path $paperDir "references.bib"))) {
  throw "Missing paper_latex/references.bib"
}
if (-not (Test-Path (Join-Path $paperDir "figures"))) {
  throw "Missing paper_latex/figures/"
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if (Test-Path $stagingDir) {
  Remove-Item -Recurse -Force $stagingDir
}
New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null

Copy-Item (Join-Path $paperDir "main.tex") (Join-Path $stagingDir "main.tex") -Force
Copy-Item (Join-Path $paperDir "references.bib") (Join-Path $stagingDir "references.bib") -Force
Copy-Item (Join-Path $paperDir "figures") $stagingDir -Recurse -Force
Copy-Item (Join-Path $paperDir "*.sty") $stagingDir -Force -ErrorAction SilentlyContinue

Remove-Item -Force -ErrorAction SilentlyContinue $outZip
Compress-Archive -Path (Join-Path $stagingDir "*") -DestinationPath $outZip -Force

Write-Host "Created: $outZip"

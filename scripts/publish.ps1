param([string]$Repository = 'hennyi-yin/monai-brain-extraction')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$taskGhExecutable = Join-Path (Get-Location) '.local\gh\bin\gh.exe'
if (-not (Test-Path -LiteralPath $taskGhExecutable)) {
    $taskGhExecutable = (Get-Command gh -ErrorAction Stop).Source
}
if (-not (Test-Path -LiteralPath 'results\acceptance.json')) {
    throw 'Complete the experiment and run python -m scripts.audit_delivery first.'
}
$taskAcceptance = Get-Content -LiteralPath 'results\acceptance.json' -Raw | ConvertFrom-Json
if ($taskAcceptance.R1_to_R10 -ne 'passed') { throw 'Artifact acceptance did not pass.' }
$taskAssets = @('artifacts\release\best.pt', 'artifacts\release\unet_weights.pt',
                'artifacts\release\test_predictions.tar.gz', 'artifacts\release\SHA256SUMS.json')
foreach ($taskAsset in $taskAssets) {
    if (-not (Test-Path -LiteralPath $taskAsset)) { throw "Missing release asset: $taskAsset" }
}
if (git status --porcelain) { throw 'Commit reviewed project artifacts before publishing.' }
& $taskGhExecutable auth status
if ($LASTEXITCODE -ne 0) { throw 'GitHub login required: .\.local\gh\bin\gh.exe auth login' }
& $taskGhExecutable auth setup-git --hostname github.com
if ($LASTEXITCODE -ne 0) { throw 'Could not configure GitHub authentication.' }
$taskRemotes = @(git remote)
if ($taskRemotes -contains 'origin') {
    $taskRemoteInfo = & $taskGhExecutable repo view --json nameWithOwner
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect existing origin.' }
    if (($taskRemoteInfo | ConvertFrom-Json).nameWithOwner -ne $Repository) {
        throw 'Existing remote does not match the requested repository.'
    }
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw 'Git push failed.' }
} else {
    & $taskGhExecutable repo view $Repository --json nameWithOwner 2>$null
    if ($LASTEXITCODE -eq 0) { throw 'Repository already exists; inspect it before attaching this project.' }
    & $taskGhExecutable repo create $Repository --public --source . --remote origin --push --description 'Reproducible MONAI 3D U-Net brain extraction on 25 held-out NFBS scans, compared with FSL BET.'
    if ($LASTEXITCODE -ne 0) { throw 'Repository creation or push failed.' }
}
& $taskGhExecutable release view v1.0.0 --repo $Repository 2>$null
if ($LASTEXITCODE -eq 0) { throw 'Release v1.0.0 already exists; refusing to overwrite published assets.' }
& $taskGhExecutable release create v1.0.0 @taskAssets --repo $Repository --title 'NFBS held-out brain extraction benchmark' --notes-file 'results\summary.md'
if ($LASTEXITCODE -ne 0) { throw 'Release upload failed.' }
Write-Output "Published https://github.com/$Repository"
Write-Output 'Open your GitHub profile and use Customize your pins to pin this repository.'
